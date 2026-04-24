import os
import pandas as pd
import pymysql
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
load_dotenv(dotenv_path='/Users/kongseunghyun/Desktop/project2/Surface-Perception-System/.env')

IMG_WIDTH = 1920
IMG_HEIGHT = 1080
INDOOR_LIMIT_X = 1400

# 실내 경로 좌표
indoor_route = [
    (456, 980), (472, 852), (801, 852), (805, 604),
    (452, 604), (480, 380), (825, 380), (837, 156),
    (276, 156), (276, 476), (452, 476), (452, 792),
    (124, 792), (172, 928), (328, 792), (456, 980)
]

# 실외 경로 좌표
outdoor_route = [
    (801, 1148), (801, 1360), (773, 1448), (801, 1148),
    (605, 1148), (605, 1440), (545, 1440), (545, 1256),
    (280, 1148), (280, 1396), (352, 1396), (240, 1396),
    (280, 1148), (456, 1148)
]

def sync_factory_data():
    db_config = {
        'host': os.getenv('DB_HOST'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'db': os.getenv('DB_NAME'),
        'charset': os.getenv('DB_CHARSET')
    }

    if not all(db_config.values()):
        print("Error: DB 연결 정보가 .env 파일에 없습니다.")
        return

    conn = None
    try:
        conn = pymysql.connect(**db_config)
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        
        print("1. 기준표 데이터 동기화 시작...")
        threshold_path = os.path.join(project_root, 'SSP', 'data', 'processed', 'indoor', 'indoor_3sigma_thresholds.csv')
        
        if os.path.exists(threshold_path):
            th_df = pd.read_csv(threshold_path)
            for _, row in th_df.iterrows():
                sql = """
                    INSERT INTO surface_thresholds 
                    (surface_type, z_limit_high, z_limit_low, diff_limit_high, diff_limit_low, description)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE 
                    z_limit_high=VALUES(z_limit_high), z_limit_low=VALUES(z_limit_low),
                    diff_limit_high=VALUES(diff_limit_high), diff_limit_low=VALUES(diff_limit_low),
                    description=VALUES(description)
                """
                desc = f"{row['surface']} 정상 기준치"
                cursor.execute(sql, (
                    row['surface'], 
                    row['accel_mag_max_upper_bound'], row['accel_mag_max_lower_bound'],
                    row['accel_diff_mean_upper_bound'], row['accel_diff_mean_lower_bound'],
                    desc
                ))
            conn.commit()
            print("기준표 적재 완료.")
        else:
            print("Warning: 기준표 CSV 파일을 찾을 수 없습니다.")

        cursor.execute("SELECT * FROM surface_thresholds")
        thresholds = {row['surface_type']: row for row in cursor.fetchall()}

        print("2. 실내 센서 데이터 적재 중...")
        indoor_path = os.path.join(project_root, 'SSP', 'data', 'processed', 'indoor', 'indoor_train_features.csv')

        if os.path.exists(indoor_path):
            indoor_df = pd.read_csv(indoor_path)
            for i, row in indoor_df.iterrows():
                route_idx = i % len(indoor_route)
                px = indoor_route[route_idx][0]
                py = indoor_route[route_idx][1]
                
                acc_z = row.get('accel_mag_max', row.get('accel_z_mean', 0))
                acc_diff = row.get('accel_diff_mean', 0)
                surface = row.get('surface', row.get('surface_label', 'Unknown'))
                
                cursor.execute("INSERT INTO sensor_logs (pos_x, pos_y, accel_z) VALUES (%s, %s, %s)", (px, py, acc_z))
                log_id = conn.insert_id()
                cursor.execute("INSERT INTO detection_results (log_id, area_type, surface_type) VALUES (%s, %s, %s)", (log_id, 'Indoor', surface))

                if surface in thresholds:
                    limit = thresholds[surface]
                    is_anomaly = False
                    reason = ""

                    if acc_z > limit['z_limit_high'] or acc_z < limit['z_limit_low']:
                        is_anomaly = True
                        reason = "Z축 충격 이상"
                    
                    if acc_diff > limit['diff_limit_high'] or acc_diff < limit['diff_limit_low']:
                        reason = "복합 이상" if is_anomaly else "미세 진동 이상"
                        is_anomaly = True

                    if is_anomaly:
                        cursor.execute("INSERT INTO incidents (log_id, type, severity) VALUES (%s, %s, %s)", (log_id, f'{reason} ({surface})', 'Medium'))
        else:
            print("Warning: 실내 데이터 CSV 파일이 없습니다.")

        print("3. 실외 포트홀 데이터 적재 중...")
        outdoor_path = os.path.join(project_root, 'SSP', 'data', 'processed', 'pothole', 'train_center_clip_peak_downsample.csv')

        if os.path.exists(outdoor_path):
            outdoor_df = pd.read_csv(outdoor_path)
            for i, row in outdoor_df.iterrows():
                route_idx = i % len(outdoor_route)
                px = outdoor_route[route_idx][0]
                py = outdoor_route[route_idx][1]
                
                acc_z = row.get('acc_z_max', 0)
                is_pothole = row.get('label', 0) == 1
                surface_status = 'Pothole' if is_pothole else 'Asphalt'
                
                cursor.execute("INSERT INTO sensor_logs (pos_x, pos_y, accel_z) VALUES (%s, %s, %s)", (px, py, acc_z))
                log_id = conn.insert_id()
                cursor.execute("INSERT INTO detection_results (log_id, area_type, surface_type) VALUES (%s, %s, %s)", (log_id, 'Outdoor', surface_status))
                
                if is_pothole:
                    cursor.execute("INSERT INTO incidents (log_id, type, severity) VALUES (%s, %s, %s)", (log_id, '포트홀 감지', 'High'))
        else:
            print("Warning: 실외 데이터 CSV 파일이 없습니다.")

        conn.commit()
        print("DB 데이터 동기화가 완료되었습니다.")

    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error 발생: {e}")
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    sync_factory_data()