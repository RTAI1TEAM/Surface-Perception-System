from flask import jsonify
import pymysql
import pandas as pd
import os

# 1. 현재 파일(prediction_service.py)의 절대 경로를 가져옵니다.
# 위치: .../SSP/app/services/prediction_service.py
current_file_path = os.path.abspath(__file__)

# 2. 프로젝트의 루트 폴더(SSP)까지 위로 올라갑니다.
# services -> app -> SSP (두 단계 위로 이동)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))

# 3. 루트에서 목표 파일까지의 경로를 합칩니다.
THRESHOLDS_PATH = os.path.join(
    project_root, 
    "data", "processed", "indoor", "indoor_3sigma_thresholds.csv"
)

# 데이터 로드 로직
try:
    thresholds_df = pd.read_csv(THRESHOLDS_PATH)
    INDOOR_THRESHOLDS = thresholds_df.set_index('surface').to_dict('index')
    print(f"✅ 이상치 기준점 로드 완료: {THRESHOLDS_PATH}")
except FileNotFoundError:
    print(f"❌ 파일을 찾을 수 없습니다. 경로 확인 필요: {THRESHOLDS_PATH}")

from services.db import get_db
from services.model_service import (
    INDOOR_FEATURES,
    OUTDOOR_FEATURES,
    predict_indoor,
    predict_outdoor,
)


MAX_PREDICTION_LOGS = 500
INDOOR_LABEL_MAP = None
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
        return {
            "metric": "vertical_impact",
            "labels": ["Z Acc Std", "Z Acc Max", "Accel Diff Max"],
            "x": float(row["indoor__linear_acceleration_Z_std"]),
            "y": float(row["indoor__linear_acceleration_Z_max"]),
            "z": float(row["indoor__accel_diff_max"]),
        }

    multiplier = 20.0
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
    if prediction["pred_label"] != "pothole":
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
                # 1. 현재 지점의 실제 데이터 추출
                current_surface = row["surface_type"]  # 예: 'carpet', 'concrete'
                accel_mag_max = float(row["indoor__accel_mag_max"])
                accel_diff_mean = float(row["indoor__accel_diff_mean"])
                
                # 2. 해당 재질의 기준점 가져오기
                limit = INDOOR_THRESHOLDS.get(current_surface)
                
                is_outlier = False
                outlier_reason = None

                if limit:
                    # 3. $3\sigma$ 기준선과 비교 (임계값 이탈 확인)
                    if not (limit['accel_mag_max_lower_bound'] <= accel_mag_max <= limit['accel_mag_max_upper_bound']):
                        is_outlier = True
                        outlier_reason = "Mag_Max 이탈"
                    elif not (limit['accel_diff_mean_lower_bound'] <= accel_diff_mean <= limit['accel_diff_mean_upper_bound']):
                        is_outlier = True
                        outlier_reason = "Diff_Mean 이탈"

                # 4. 모델 예측 수행 (기존 로직)
                feature_dict = _extract_feature_dict(row, "indoor__", INDOOR_FEATURES)
                prediction = predict_indoor(feature_dict, _load_indoor_label_map())

                # 5. 이상치인 경우 라벨 강제 변경 (웹에서 핀을 꽂기 위함)
                if is_outlier:
                    prediction["pred_label"] = "outlier" 
                    prediction["outlier_reason"] = outlier_reason # 웹에 이유도 함께 전달
            else:
                if row["outdoor_feature_id"] is None:
                    return jsonify({"status": "error", "message": f"outdoor feature missing for point_id={point_id}"}), 400
                feature_dict = _extract_feature_dict(row, "outdoor__", OUTDOOR_FEATURES)
                prediction = predict_outdoor(feature_dict)

            prediction_id = None
            if prediction["pred_label"] == "pothole":
                prediction_id = _insert_prediction_log(cursor, row, prediction)
                _prune_prediction_logs(cursor, row["route_id"])

            LAST_PREDICTION_STATE = _prediction_payload(row, prediction, prediction_id)

        db.commit()
    finally:
        db.close()

    return jsonify(LAST_PREDICTION_STATE)
