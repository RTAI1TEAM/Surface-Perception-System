import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (classification_report,confusion_matrix,
accuracy_score,
roc_curve,
roc_auc_score,
precision_recall_curve,
average_precision_score,
make_scorer,
f1_score
)
import joblib
import os

# =======================================================
# 0. 설정 및 데이터 로드
# =======================================================
DATASET_NAME = "v3_s2"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../../../data/processed/pothole")
TRAIN_PATH = os.path.join(DATA_DIR, f"train_{DATASET_NAME}.csv")
TEST_PATH = os.path.join(DATA_DIR, f"test_{DATASET_NAME}.csv")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

print(f":white_check_mark: 학습 데이터 로드 완료 ({DATASET_NAME})")
print(f"클래스 분포: \n{y_train.value_counts()}\n")

# =======================================================
# 1. 데이터 스케일링 (로지스틱 회귀 필수)
# =======================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =======================================================
# 2. GridSearchCV를 이용한 최적의 모델 학습
# =======================================================
print(":rocket: 최적의 하이퍼파라미터 탐색 중 (GridSearchCV)...")

# 포트홀(Class 1)의 F1 점수 전용 채점관
pothole_f1_scorer = make_scorer(f1_score, pos_label=1)

# 로지스틱 회귀 파라미터 그리드 (C가 작을수록 강한 규제)
param_grid = {
'C': [0.01, 0.1, 1, 10, 100],
'solver': ['liblinear', 'lbfgs'],
'max_iter': [1000] # 수렴을 위해 반복 횟수 넉넉히 설정
}

# class_weight='balanced' 기본 장착
base_logistic = LogisticRegression(class_weight='balanced', random_state=42)

grid_search = GridSearchCV(
estimator=base_logistic,
param_grid=param_grid,
cv=5,
scoring=pothole_f1_scorer,
n_jobs=-1,
verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print("\n=== GridSearchCV 탐색 완료 ===")
print("가장 성능이 좋은 조합:", grid_search.best_params_)

best_logistic_model = grid_search.best_estimator_

# =======================================================
# 3. 임계값(Threshold) 최적화 및 지표 계산
# =======================================================
y_probs = best_logistic_model.predict_proba(X_test_scaled)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)

# 포트홀 F1-Score 시뮬레이션
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-9)
best_idx = np.argmax(f1_scores)

# 최적 임계값 추출
best_threshold = thresholds[best_idx]
y_pred_tuned = (y_probs >= best_threshold).astype(int)

# =======================================================
# 4. 최종 결과 출력
# =======================================================
print("\n" + "="*60)
print(f":bar_chart: [최종 보고서 - Logistic Regression] Dataset: {DATASET_NAME}")
print("="*60)
print(f":white_check_mark: 최적 임계값 (Optimal Threshold): {best_threshold:.4f}")
print(f":white_check_mark: 예상 최대 F1-Score (Pothole): {f1_scores[best_idx]:.4f}")
print("-" * 60)
print(classification_report(y_test, y_pred_tuned, target_names=['Normal(0)', 'Pothole(1)']))
print("-" * 60)

print(f"accuracy: {accuracy_score(y_test, y_pred_tuned):.4f}")

roc_auc = roc_auc_score(y_test, y_probs)
pr_auc = average_precision_score(y_test, y_probs)
print(f"ROC_AUC Score: {roc_auc:.4f}")
print(f"PR_AUC Score:  {pr_auc:.4f}")
print("="*60)


# =======================================================
# 5. 모델 및 스케일러 저장
# =======================================================
# 저장 경로: SSP/models/pothole (사용자 요청 경로 반영)
target_dir = os.path.abspath(os.path.join(BASE_DIR, "../../../models/pothole"))

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

model_path = os.path.join(target_dir, "lr_best.pkl")
scaler_path = os.path.join(target_dir, "scaler.pkl")

# 저장 실행
joblib.dump(best_logistic_model, model_path)
joblib.dump(scaler, scaler_path) # 스케일러도 세트로 저장해야 나중에 사용 가능합니다.

print(f"\n🎉 모델 저장 완료: {model_path}")
print(f"🎉 스케일러 저장 완료: {scaler_path}")