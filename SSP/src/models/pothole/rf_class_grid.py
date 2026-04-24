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
import joblib 
import os

# =======================================================
# 0. 데이터셋 이름 설정
# =======================================================
DATASET_NAME = "v3_s2"

# 1. 데이터 로드
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

print(f"학습 데이터 클래스 분포 ({DATASET_NAME}): \n{y_train.value_counts()}\n")

# =======================================================
# 2. GridSearchCV를 이용한 최적의 RF 모델 학습 (Class Weight 적용)
# =======================================================
print("최적의 하이퍼파라미터 탐색 중 (GridSearchCV)...")

# 💡 핵심: 포트홀(Class 1)의 F1 점수만 바라보는 전용 채점관 생성
pothole_f1_scorer = make_scorer(f1_score, pos_label=1)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 'log2']
}

# 💡 핵심: 불균형 처리를 위해 class_weight='balanced'를 기본으로 설정
base_rf = RandomForestClassifier(class_weight='balanced', random_state=42)

grid_search = GridSearchCV(
    estimator=base_rf, 
    param_grid=param_grid, 
    cv=5, 
    scoring=pothole_f1_scorer, # 포트홀 F1 기준 채점
    n_jobs=-1, 
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\n=== GridSearchCV 탐색 완료 ===")
print("가장 성능이 좋은 조합:", grid_search.best_params_)

# 최적 모델 추출
best_rf_model = grid_search.best_estimator_

# =======================================================
# 3. 임계값 자동 최적화 (Pothole F1-Score 극대화)
# =======================================================
y_probs = best_rf_model.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)

# 포트홀 F1-Score 계산
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-9)

# 최적 지점 찾기
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print("\n" + "="*50)
print(f"🎯 [자동 최적화] 포트홀 F1 극대화 임계값: {best_threshold:.4f}")
print(f"🏆 해당 지점의 예상 F1-Score: {best_f1:.4f}")
print("="*50)

# 최종 예측값 생성
y_pred_tuned = (y_probs >= best_threshold).astype(int)

# =======================================================
# 4. 결과 출력 및 저장
# =======================================================
print(f"\n--- [최종 최적화 결과 - {DATASET_NAME}] ---")
print(classification_report(y_test, y_pred_tuned))
print("accuracy:", accuracy_score(y_test, y_pred_tuned))

roc_auc = roc_auc_score(y_test, y_probs)
pr_auc = average_precision_score(y_test, y_probs)
print(f"ROC_AUC Score: {roc_auc:.4f}")
print(f"PR_AUC Score:  {pr_auc:.4f}")

# 모델 저장
MODEL_DIR = os.path.join(BASE_DIR, "../../../models/pothole")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, f"rf_class_grid_{DATASET_NAME}.pkl")
joblib.dump(best_rf_model, MODEL_SAVE_PATH)

# =======================================================
# 5. 시각화 (파일명에 데이터셋 이름 포함)
# =======================================================
FIGURE_DIR = os.path.join(BASE_DIR, "../../../reports/figures/outdoor_performance/rf")
os.makedirs(FIGURE_DIR, exist_ok=True)

# 5-1. Confusion Matrix
plt.figure(figsize=(7, 5))
cm = confusion_matrix(y_test, y_pred_tuned)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Normal(0)', 'Pothole(1)'], yticklabels=['Normal(0)', 'Pothole(1)'])
plt.title(f'CM (RF Class+Grid, Thresh: {best_threshold:.2f})')
plt.savefig(os.path.join(FIGURE_DIR, f"rf_class_grid_cm_{DATASET_NAME}.png"), dpi=300)
plt.close()

# 5-2. Feature Importance
importances = best_rf_model.feature_importances_
indices = np.argsort(importances)[::-1][:10]
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=X_train.columns[indices], palette='viridis')
plt.title(f'Top 10 FI (RF Class+Grid - {DATASET_NAME})')
plt.savefig(os.path.join(FIGURE_DIR, f"rf_class_grid_fi_{DATASET_NAME}.png"), dpi=300)
plt.close()

# 5-3. ROC & PR Curve
roc_auc = roc_auc_score(y_test, y_probs)
pr_auc = average_precision_score(y_test, y_probs)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.title(f'ROC Curve ({DATASET_NAME})')
plt.legend()
plt.savefig(os.path.join(FIGURE_DIR, f"rf_class_grid_roc_{DATASET_NAME}.png"), dpi=300)
plt.close()

# PR Curve
plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, label=f'AP = {pr_auc:.4f}', color='green')
plt.plot(recalls[best_idx], precisions[best_idx], 'ro', label=f'Best F1 (Thresh={best_threshold:.2f})')
plt.title(f'PR Curve ({DATASET_NAME})')
plt.legend()
plt.savefig(os.path.join(FIGURE_DIR, f"rf_class_grid_pr_{DATASET_NAME}.png"), dpi=300)
plt.close()

print(f"\n📈 모든 분석 결과와 이미지가 저장되었습니다. ({DATASET_NAME})")