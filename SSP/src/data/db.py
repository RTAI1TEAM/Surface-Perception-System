import pandas as pd
import pymysql
import numpy as np
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# ==========================================
# 0. 환경 변수 로드
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
env_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=env_path)

# ==========================================
# 1. 환경 설정 (우리 조감도 해상도)
# ==========================================
IMG_WIDTH = 1920        # 조감도 가로 크기 (픽셀)
IMG_HEIGHT = 1080       # 조감도 세로 크기 (픽셀)
INDOOR_LIMIT_X = 1400   # 실내 구역이 끝나는 X 좌표 (약 75% 지점)


def sync_factory_data() -> None:
    """
    전처리된 실내 및 실외 데이터를 읽어와 MySQL 데이터베이스(sensor_logs, detection_results)에 통합 적재하는 함수입니다.
    데이터베이스 연결 정보는 보안을 위해 .env 파일에서 가져옵니다.
    """
    
    db_config = {
        'host': os.getenv('DB_HOST'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'db': os.getenv('DB_NAME'),
        'charset': os.getenv('DB_CHARSET')
    }

    # DB 환경변수 누락 체크
    if not all(db_config.values()):
        print("[Error] .env 파일에 DB 연결 정보가 완벽하게 설정되지 않았습니다.")
        return

    # DB 연결
    try:
        conn = pymysql.connect(**db_config)
        cursor = conn.cursor()
    except Exception as e:
        print(f"[Error] DB 연결 실패: {e}")
        return
    
    try:
        print("[Info] 데이터 통합 적재 작업을 시작합니다...")

        # ------------------------------------------
        # 2. 실내 데이터 적재 (Indoor Section)
        # ------------------------------------------
        indoor_path = os.path.join(project_root, 'SSP', 'data', 'processed', 'indoor', 'indoor_train_features.csv')
        
        if os.path.exists(indoor_path):
            indoor_df = pd.read_csv(indoor_path)
            
            print(f"[Info] 실내 데이터({len(indoor_df)}건) 매핑 중...")
            for i, row in indoor_df.iterrows():
                # [경로 설계] 왼쪽 끝에서 문(INDOOR_LIMIT_X)까지 이동
                pos_x = (i / len(indoor_df)) * INDOOR_LIMIT_X
                pos_y = IMG_HEIGHT * 0.45 # 조감도 중앙 복도를 따라 주행
                
                # (1) sensor_logs 테이블에 원천 데이터 저장
                sql_log = """
                    INSERT INTO sensor_logs (pos_x, pos_y, accel_x, accel_y, accel_z, roll, pitch, yaw)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(sql_log, (
                    pos_x, pos_y, 
                    row.get('accel_x_mean', 0), row.get('accel_y_mean', 0), row.get('accel_z_mean', 0),
                    row.get('roll_mean', 0), row.get('pitch_mean', 0), row.get('yaw_mean', 0)
                ))
                
                # (2) 방금 저장된 로그의 ID(FK)를 가져와서 AI 판독 결과 저장
                log_id = cursor.lastrowid
                sql_res = """
                    INSERT INTO detection_results (log_id, area_type, surface_type, confidence)
                    VALUES (%s, %s, %s, %s)
                """
                surface_label = row.get('surface', row.get('surface_label', 'Unknown'))
                cursor.execute(sql_res, (log_id, 'Indoor', surface_label, row.get('confidence', 0.95)))
        else:
            print(f"[Warning] 실내 데이터 파일을 찾을 수 없습니다: {indoor_path}")

        # ------------------------------------------
        # 3. 실외 데이터 적재 (Outdoor Section)
        # ------------------------------------------
        outdoor_path = os.path.join(project_root, 'SSP', 'data', 'processed', 'pothole', 'train_center_clip_peak_downsample.csv')
        
        if os.path.exists(outdoor_path):
            outdoor_df = pd.read_csv(outdoor_path)
            
            print(f"[Info] 실외 데이터({len(outdoor_df)}건) 매핑 중...")
            for i, row in outdoor_df.iterrows():
                # [경로 설계] 문(INDOOR_LIMIT_X)에서 오른쪽 끝까지 이동
                pos_x = INDOOR_LIMIT_X + ((i / len(outdoor_df)) * (IMG_WIDTH - INDOOR_LIMIT_X))
                pos_y = IMG_HEIGHT * 0.45 
                
                # (1) sensor_logs 저장
                acc_z = row.get('acc_z_max', row.get('accel_z', 0))

                sql_log = "INSERT INTO sensor_logs (pos_x, pos_y, accel_z) VALUES (%s, %s, %s)"
                cursor.execute(sql_log, (pos_x, pos_y, acc_z))
                
                # (2) detection_results 저장
                log_id = cursor.lastrowid
                is_pothole = row.get('label', 0) == 1
                surface_status = 'Pothole' if is_pothole else 'Asphalt'
                
                sql_res = """
                    INSERT INTO detection_results (log_id, area_type, surface_type, confidence)
                    VALUES (%s, %s, %s, %s)
                """
                cursor.execute(sql_res, (log_id, 'Outdoor', surface_status, 0.99))
        else:
            print(f"[Warning] 실외 데이터 파일을 찾을 수 없습니다: {outdoor_path}")

        conn.commit()
        print("[Success] 모든 데이터가 성공적으로 통합되어 DB에 적재되었습니다.")

    except Exception as e:
        conn.rollback()
        print(f"[Error] 작업 중 오류 발생: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    sync_factory_data()
