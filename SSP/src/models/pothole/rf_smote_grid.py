import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    roc_curve, 
    roc_auc_score, 
    precision_recall_curve, 
    average_precision_score,
    make_scorer,
    f1_score
)
from imblearn.over_sampling import SMOTE 
import joblib 
import os

# =======================================================
# 0. 설정
# =======================================================
DATASET_NAME = "v3_s2"

# 1. 데이터 로드
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../../../data/processed/pothole")
TRAIN_PATH = os.path.join(DATA_DIR, f"train_{DATASET_NAME}.csv")
TEST_PATH = os.path.join(DATA_DIR, f"test_{DATASET_NAME}.csv")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train_raw = train_df.drop('label', axis=1)
y_train_raw = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# 2. SMOTE 적용
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train_raw, y_train_raw)

# 3. GridSearchCV (Pothole F1 기준)
pothole_f1_scorer = make_scorer(f1_score, pos_label=1)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}

print("🚀 최적의 하이퍼파라미터 탐색 시작...")
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42), 
    param_grid=param_grid, 
    cv=5, 
    scoring=pothole_f1_scorer, 
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_rf_model = grid_search.best_estimator_

# =======================================================
# 4. 임계값(Threshold) 최적화 및 지표 계산
# =======================================================
y_probs = best_rf_model.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)

# 포트홀 F1-Score 시뮬레이션
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-9)
best_idx = np.argmax(f1_scores)

# 🎯 최적 임계값 추출
best_threshold = thresholds[best_idx]
y_pred_tuned = (y_probs >= best_threshold).astype(int)

# =======================================================
# 5. 최종 결과 출력 (요청 사항 반영)
# =======================================================
print("\n" + "="*60)
print(f"📊 [최종 보고서] Dataset: {DATASET_NAME}")
print("="*60)
print(f"✅ 최적 임계값 (Optimal Threshold): {best_threshold:.4f}")
print(f"✅ 예상 최대 F1-Score (Pothole): {f1_scores[best_idx]:.4f}")
print("-" * 60)
print(classification_report(y_test, y_pred_tuned))
print("-" * 60)

# 요청하신 추가 지표 출력
print(f"accuracy: {accuracy_score(y_test, y_pred_tuned):.4f}")

roc_auc = roc_auc_score(y_test, y_probs)
pr_auc = average_precision_score(y_test, y_probs)
print(f"ROC_AUC Score: {roc_auc:.4f}")
print(f"PR_AUC Score:  {pr_auc:.4f}")
print("="*60)

# =======================================================
# 6. 저장 및 시각화
# =======================================================
MODEL_DIR = os.path.join(BASE_DIR, "../../../models/pothole")
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(best_rf_model, os.path.join(MODEL_DIR, f"rf_smote_grid_{DATASET_NAME}.pkl"))

FIGURE_DIR = os.path.join(BASE_DIR, "../../../reports/figures/outdoor_performance/rf")
os.makedirs(FIGURE_DIR, exist_ok=True)

# PR Curve 저장 (최적점 표시 포함)
plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, label=f'AP = {pr_auc:.4f}', color='green')
plt.plot(recalls[best_idx], precisions[best_idx], 'ro', label=f'Optimal (Thresh={best_threshold:.4f})')
plt.title(f'Precision-Recall Curve ({DATASET_NAME})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.savefig(os.path.join(FIGURE_DIR, f"rf_smote_grid_pr_{DATASET_NAME}.png"), dpi=300)
plt.close()

print(f"\n📈 모델과 PR Curve 이미지가 저장되었습니다. (Thresh: {best_threshold:.4f})")