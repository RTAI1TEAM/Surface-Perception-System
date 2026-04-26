from flask import jsonify
import pymysql

from services.db import get_db
from services.model_service import (
    INDOOR_FEATURES,
    OUTDOOR_FEATURES,
    predict_indoor,
    predict_outdoor,
)


MAX_PREDICTION_LOGS = 500
INDOOR_LABEL_MAP = None


def _load_indoor_label_map():
    global INDOOR_LABEL_MAP
    if INDOOR_LABEL_MAP is not None:
        return INDOOR_LABEL_MAP

    db = get_db()
    try:
        with db.cursor() as cursor:
            cursor.execute(
                """
                SELECT DISTINCT surface_encoded, surface
                FROM indoor_route_features
                WHERE surface_encoded IS NOT NULL AND surface IS NOT NULL
                """
            )
            INDOOR_LABEL_MAP = {row[0]: row[1] for row in cursor.fetchall()}
    finally:
        db.close()

    return INDOOR_LABEL_MAP


def get_robot_path_points():
    db = get_db()
    try:
        with db.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute(
                """
                SELECT
                    rp.point_id,
                    rp.sequence_no,
                    rp.pos_x,
                    rp.pos_y,
                    rp.area_type,
                    rp.surface_type,
                    irf.surface AS indoor_feature_label,
                    CASE
                        WHEN orf.source_label = 1 THEN 'pothole'
                        WHEN orf.source_label = 0 THEN 'normal_road'
                        ELSE NULL
                    END AS outdoor_feature_label
                FROM route_points rp
                INNER JOIN routes r ON r.route_id = rp.route_id
                LEFT JOIN indoor_route_features irf ON irf.point_id = rp.point_id
                LEFT JOIN outdoor_route_features orf ON orf.point_id = rp.point_id
                WHERE r.is_active = 1 AND rp.is_active = 1
                ORDER BY rp.route_id, rp.sequence_no
                """
            )
            rows = cursor.fetchall()
    finally:
        db.close()

    return [
        {
            "point_id": row["point_id"],
            "sequence_no": row["sequence_no"],
            "x": float(row["pos_x"]),
            "y": float(row["pos_y"]),
            "area_type": row["area_type"],
            "surface_type": row["surface_type"],
            "feature_label": row["indoor_feature_label"] if row["area_type"] == "Indoor" else row["outdoor_feature_label"],
        }
        for row in rows
    ]


def _fetch_point_payload(cursor, point_id):
    indoor_cols = ", ".join([f"irf.`{name}` AS `indoor__{name}`" for name in INDOOR_FEATURES])
    outdoor_cols = ", ".join([f"orf.`{name}` AS `outdoor__{name}`" for name in OUTDOOR_FEATURES])

    cursor.execute(
        f"""
        SELECT
            rp.route_id,
            rp.point_id,
            rp.sequence_no,
            rp.pos_x,
            rp.pos_y,
            rp.area_type,
            rp.surface_type,
            irf.indoor_feature_id,
            irf.surface AS indoor_feature_label,
            irf.surface_encoded,
            orf.outdoor_feature_id,
            orf.source_label,
            {indoor_cols},
            {outdoor_cols}
        FROM route_points rp
        LEFT JOIN indoor_route_features irf ON irf.point_id = rp.point_id
        LEFT JOIN outdoor_route_features orf ON orf.point_id = rp.point_id
        WHERE rp.point_id = %s
        """,
        (point_id,),
    )
    return cursor.fetchone()


def _extract_feature_dict(row, prefix, feature_names):
    return {name: row[f"{prefix}{name}"] for name in feature_names}


def _insert_prediction_log(cursor, row, prediction):
    if prediction["pred_label"] == "normal_road":
        return
    cursor.execute(
        """
        INSERT INTO prediction_logs (
            route_id, point_id, area_type, feature_table_type,
            indoor_feature_id, outdoor_feature_id,
            sequence_no, pos_x, pos_y, surface_type,
            pred_label, pred_prob, model_name, loop_no
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            row["route_id"],
            row["point_id"],
            row["area_type"],
            prediction["feature_table_type"],
            row["indoor_feature_id"],
            row["outdoor_feature_id"],
            row["sequence_no"],
            row["pos_x"],
            row["pos_y"],
            row["surface_type"],
            prediction["pred_label"],
            prediction["pred_prob"],
            prediction["model_name"],
            1,
        ),
    )

def _select_prediction_log():
    db = get_db()
    try:
        with db.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute("""
                select prediction_id, played_at, area_type, surface_type, pred_label, pred_prob, pos_x, pos_y
                from prediction_logs
                ORDER BY played_at DESC
                LIMIT 1;               
            """)
            row = cursor.fetchone()
    except Exception as e:
        print('error :', e)
    finally:
        db.close()

    if not row:
        return jsonify({"status": "empty"})

    return jsonify(
        {
            "status": "ok",
            "prediction_id": row["prediction_id"],
            "played_at": row["played_at"].strftime("%Y-%m-%d %H:%M:%S") if row["played_at"] else None,
            "area_type": row["area_type"],
            "surface_type": row["surface_type"],
            "pred_label": row["pred_label"],
            "pred_prob": f"{float(row['pred_prob']) * 100:.1f}",
            "x": float(row["pos_x"]),
            "y": float(row["pos_y"]),
        }
    )

def _prune_prediction_logs(cursor, route_id):
    cursor.execute(
        """
        DELETE FROM prediction_logs
        WHERE route_id = %s
          AND prediction_id NOT IN (
              SELECT prediction_id
              FROM (
                  SELECT prediction_id
                  FROM prediction_logs
                  WHERE route_id = %s
                  ORDER BY prediction_id DESC
                  LIMIT %s
              ) AS kept
          )
        """,
        (route_id, route_id, MAX_PREDICTION_LOGS),
    )


def process_point_prediction(payload):
    point_id = payload.get("point_id")
    if point_id is None:
        return jsonify({"status": "error", "message": "point_id is required"}), 400

    db = get_db()
    try:
        with db.cursor(pymysql.cursors.DictCursor) as cursor:
            row = _fetch_point_payload(cursor, point_id)
            if not row:
                return jsonify({"status": "error", "message": f"point_id={point_id} not found"}), 404

            if row["area_type"] == "Indoor":
                if row["indoor_feature_id"] is None:
                    return jsonify({"status": "error", "message": f"indoor feature missing for point_id={point_id}"}), 400
                feature_dict = _extract_feature_dict(row, "indoor__", INDOOR_FEATURES)
                prediction = predict_indoor(feature_dict, _load_indoor_label_map())
            else:
                if row["outdoor_feature_id"] is None:
                    return jsonify({"status": "error", "message": f"outdoor feature missing for point_id={point_id}"}), 400
                feature_dict = _extract_feature_dict(row, "outdoor__", OUTDOOR_FEATURES)
                prediction = predict_outdoor(feature_dict)

            cursor.execute(
                """
                SELECT pred_label
                FROM prediction_logs
                WHERE route_id = %s
                ORDER BY prediction_id DESC
                LIMIT 1
                """,
                (row["route_id"],),
            )
            last_log = cursor.fetchone()
            prediction_changed = last_log is None or last_log["pred_label"] != prediction["pred_label"]

            if prediction_changed:
                _insert_prediction_log(cursor, row, prediction)
                _prune_prediction_logs(cursor, row["route_id"])

        db.commit()
    finally:
        db.close()

    return jsonify(
        {
            "status": "ok",
            "point_id": row["point_id"],
            "route_id": row["route_id"],
            "sequence_no": row["sequence_no"],
            "x": float(row["pos_x"]),
            "y": float(row["pos_y"]),
            "area_type": row["area_type"],
            "surface_type": row["surface_type"],
            "feature_label": row["indoor_feature_label"] if row["area_type"] == "Indoor" else ("pothole" if row["source_label"] == 1 else "normal_road"),
            "pred_label": prediction["pred_label"],
            "pred_prob": prediction["pred_prob"],
            "logged": prediction_changed,
        }
    )
