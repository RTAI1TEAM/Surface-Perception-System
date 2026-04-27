from flask import jsonify
import pymysql
import os
import pandas as pd

from services.db import get_db
from services.model_service import (
    INDOOR_FEATURES,
    OUTDOOR_FEATURES,
    predict_indoor,
    predict_outdoor,
)


MAX_PREDICTION_LOGS = 500
INDOOR_LABEL_MAP = None
INDOOR_THRESHOLDS = None
LAST_PREDICTION_STATE = None


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


def _load_indoor_thresholds():
    global INDOOR_THRESHOLDS
    if INDOOR_THRESHOLDS is not None:
        return INDOOR_THRESHOLDS

    threshold_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data", "processed", "indoor", "indoor_3sigma_thresholds.csv"
    )
    if os.path.exists(threshold_path):
        try:
            df = pd.read_csv(threshold_path)
            # surface를 인덱스로 하여 딕셔너리로 변환
            INDOOR_THRESHOLDS = df.set_index("surface").to_dict("index")
        except Exception as e:
            print(f"Error loading thresholds: {e}")
            INDOOR_THRESHOLDS = {}
    else:
        print(f"Threshold file not found: {threshold_path}")
        INDOOR_THRESHOLDS = {}

    return INDOOR_THRESHOLDS


def _check_indoor_anomaly(surface, feature_dict):
    thresholds = _load_indoor_thresholds()
    if surface not in thresholds:
        return False, None

    limit = thresholds[surface]
    is_anomaly = False
    reason = ""

    # 1. accel_mag_max (Z축 충격) 체크
    acc_z = feature_dict.get("accel_mag_max")
    if acc_z is not None:
        if acc_z > limit["accel_mag_max_upper_bound"] or acc_z < limit["accel_mag_max_lower_bound"]:
            is_anomaly = True
            reason = "Z-axis Impact"

    # 2. accel_diff_mean (진동 변화량) 체크
    acc_diff = feature_dict.get("accel_diff_mean")
    if acc_diff is not None:
        if acc_diff > limit["accel_diff_mean_upper_bound"] or acc_diff < limit["accel_diff_mean_lower_bound"]:
            reason = "Complex Anomaly" if is_anomaly else "Vibration Anomaly"
            is_anomaly = True

    return is_anomaly, reason


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


def _build_chart_payload(row):
    if row["area_type"] == "Indoor":
        accel_diff_mean = float(row["indoor__accel_diff_mean"])
        accel_mag_max = float(row["indoor__accel_mag_max"])
        z_acc_std = float(row["indoor__linear_acceleration_Z_std"])

        scaling_down_factor = 0.5
        
        return {
            "metric": "vertical_impact",
            "labels": ["Accel Diff Mean", "Accel Mag Max", "Z-Acc Std"], 
            "x": accel_diff_mean,
            "y": accel_mag_max * scaling_down_factor,
            "z": z_acc_std,
        }
    

    multiplier = 25.0
    az_std = float(row["outdoor__az_std"]) * multiplier
    az_max = float(row["outdoor__az_max"]) * multiplier
    az_min = float(row.get("outdoor__az_min", 0))
    az_range = (az_max - (az_min * multiplier)) if az_min != 0 else (az_max * 0.8)

    return {
        "metric": "vertical_impact",
        "labels": ["Z Acc Std (S)", "Z Acc Max (S)", "Vibration Range (S)"],
        "x": az_std,
        "y": az_max,
        "z": az_range,
    }


def _insert_prediction_log(cursor, row, prediction):
    # 실외 포트홀이거나 실내 이상치인 경우 모두 로그 저장
    is_pothole = (prediction["pred_label"] == "pothole")
    is_indoor_anomaly = prediction.get("is_anomaly", False)

    if not (is_pothole or is_indoor_anomaly):
        return None
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
    return cursor.lastrowid


def _prediction_payload(row, prediction, prediction_id=None):
    return {
        "status": "ok",
        "prediction_id": prediction_id,
        "played_at": None,
        "point_id": row["point_id"],
        "route_id": row["route_id"],
        "sequence_no": row["sequence_no"],
        "x": float(row["pos_x"]),
        "y": float(row["pos_y"]),
        "area_type": row["area_type"],
        "surface_type": row["surface_type"],
        "feature_label": row["indoor_feature_label"] if row["area_type"] == "Indoor" else ("pothole" if row["source_label"] == 1 else "normal_road"),
        "pred_label": prediction["pred_label"],
        "pred_prob": float(prediction["pred_prob"]) * 100,
        "pred_prob_raw": prediction["pred_prob"],
        "is_anomaly": prediction.get("is_anomaly", False),
        "anomaly_reason": prediction.get("anomaly_reason", ""),
        "logged": prediction_id is not None,
        "chart": _build_chart_payload(row),
    }


def _select_prediction_log():
    if LAST_PREDICTION_STATE is not None:
        return jsonify(LAST_PREDICTION_STATE)

    db = get_db()
    try:
        with db.cursor(pymysql.cursors.DictCursor) as cursor:
            cursor.execute("""
                SELECT prediction_id, played_at, area_type, surface_type, pred_label, pred_prob, pos_x, pos_y
                FROM prediction_logs
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
    global LAST_PREDICTION_STATE

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

                # 3-Sigma 기반 이상감지 로직 추가
                is_anomaly, reason = _check_indoor_anomaly(prediction["pred_label"], feature_dict)
                prediction["is_anomaly"] = is_anomaly
                prediction["anomaly_reason"] = reason
                if is_anomaly:
                    prediction["pred_label"] = f"{prediction['pred_label']} ({reason})"
            else:
                if row["outdoor_feature_id"] is None:
                    return jsonify({"status": "error", "message": f"outdoor feature missing for point_id={point_id}"}), 400
                feature_dict = _extract_feature_dict(row, "outdoor__", OUTDOOR_FEATURES)
                prediction = predict_outdoor(feature_dict)
                
                # 포트홀 감지 문턱값 상향 (65% 이상일 때만 확정)
                if prediction["pred_label"] == "pothole" and prediction["pred_prob"] < 0.65:
                    prediction["pred_label"] = "normal_road"
                    prediction["pred_prob"] = 1.0 - prediction["pred_prob"]
                
                prediction["is_anomaly"] = (prediction["pred_label"] == "pothole")
                prediction["anomaly_reason"] = "Pothole" if prediction["is_anomaly"] else ""

            prediction_id = None
            is_pothole = (prediction["pred_label"] == "pothole")
            is_indoor_anomaly = prediction.get("is_anomaly", False)

            if is_pothole or is_indoor_anomaly:
                prediction_id = _insert_prediction_log(cursor, row, prediction)
                _prune_prediction_logs(cursor, row["route_id"])

            LAST_PREDICTION_STATE = _prediction_payload(row, prediction, prediction_id)

        db.commit()
    finally:
        db.close()

    return jsonify(LAST_PREDICTION_STATE)
