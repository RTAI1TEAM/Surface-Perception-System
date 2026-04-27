import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import classification_report, accuracy_score, f1_score

# 1. 경로 설정
BASE_DIR = r"C:\work\mlproject\Surface-Perception-System\SSP"
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "indoor", "indoor_train_features.csv")
MODEL_PATH = os.path.join(BASE_DIR, "app", "models", "best_surface_model.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "data", "processed", "indoor", "indoor_3sigma_thresholds.csv")

def load_data_for_eval(file_path=DATA_PATH):
    df = pd.read_csv(file_path)
    target_col = 'surface_encoded'
    from sklearn.model_selection import train_test_split
    # 원본 인덱스 유지를 위해 분할
    _, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target_col])
    
    y_test = test_df[target_col]
    # 모델 입력용 데이터 (불필요 컬럼 제거)
    X_test = test_df.drop(columns=['series_id', 'group_id', 'surface', target_col, 'surface_encoded'], errors='ignore')
    
    return X_test, y_test, test_df

def evaluate_and_find_outliers():
    # 2. 로드
    print("⏳ 모델 및 데이터 로드 중...")
    model = joblib.load(MODEL_PATH)
    X_test, y_test, test_df_full = load_data_for_eval()
    thresholds = pd.read_csv(THRESHOLD_PATH)
    
    class_names = test_df_full[['surface_encoded', 'surface']].drop_duplicates().sort_values('surface_encoded')['surface'].values

    # 3. 예측 및 성능 확인
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print("\n📊 [모델 성능 요약]")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # 4. 이상치 탐지 및 세부 정보 수집
    outlier_list = []      # 요약 정보용
    outlier_indices = []   # 전체 행 데이터 추출용
    
    for i, (idx, row) in enumerate(test_df_full.iterrows()):
        surface_name = row['surface']
        target_threshold = thresholds[thresholds['surface'] == surface_name].iloc[0]
        
        is_outlier = False
        reasons = []
        
        # 3시그마 임계값 체크
        if row['accel_mag_max'] < target_threshold['accel_mag_max_lower_bound'] or \
           row['accel_mag_max'] > target_threshold['accel_mag_max_upper_bound']:
            is_outlier = True
            reasons.append("Mag_Max 이탈")
            
        if row['accel_diff_mean'] < target_threshold['accel_diff_mean_lower_bound'] or \
           row['accel_diff_mean'] > target_threshold['accel_diff_mean_upper_bound']:
            is_outlier = True
            reasons.append("Diff_Mean 이탈")
            
        if is_outlier:
            # 예측 결과 결합
            pred_class_idx = y_pred[i]
            pred_prob = np.max(y_proba[i]) * 100

            outlier_list.append({
                'Row': idx,
                'Actual': surface_name,
                'Reason': ", ".join(reasons),
                'Model_Predict': class_names[pred_class_idx],
                'Confidence': f"{pred_prob:.2f}%"
            })
            outlier_indices.append(idx)

    # 5. 결과 출력
    if outlier_list:
        # A. 요약 표 출력
        summary_df = pd.DataFrame(outlier_list)
        print("\n🚨 [이상치 요약 보고서]")
        print(summary_df.to_string(index=False))

        # B. 이상치 행의 전체 정보 데이터프레임 생성
        # 원본 데이터(test_df_full)에서 이상치 인덱스만 필터링하고 예측 정보를 붙입니다.
        outlier_full_details = test_df_full.loc[outlier_indices].copy()
        outlier_full_details['Model_Prediction'] = summary_df['Model_Predict'].values
        outlier_full_details['Model_Confidence'] = summary_df['Confidence'].values
        outlier_full_details['Outlier_Reason'] = summary_df['Reason'].values

        print("\n📋 [이상치 전체 행 데이터프레임 샘플 (상위 5개)]")
        # 컬럼이 너무 많으므로 일부만 출력하거나, 필요시 전체를 csv로 저장할 수 있습니다.
        # print(outlier_full_details.head().to_string())

        # C. 필요시 CSV로 내보내기 (추가 분석용)
        save_path = os.path.join(BASE_DIR, "reports", "figures", "indoor_performance", "outlier_detailed_analysis.csv")
        outlier_full_details.to_csv(save_path)
        print(f"\n✅ 상세 데이터가 {save_path}에 저장되었습니다.")

        # 분석 팁
        print("\n💡 분석 팁: Confidence(확률)가 높은데 틀렸다면 모델이 잘못된 특징을 학습한 것이고,")
        print("    확률이 낮다면 모델도 해당 데이터를 이상치로 느껴 판단을 망설인 것으로 해석할 수 있습니다.")
        
        return outlier_full_details # 함수 밖에서도 쓸 수 있게 리턴
    else:
        print("✅ 이상치가 없습니다.")
        return None

if __name__ == "__main__":
    outlier_df = evaluate_and_find_outliers()



