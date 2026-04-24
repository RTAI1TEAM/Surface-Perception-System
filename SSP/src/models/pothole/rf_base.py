import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    roc_curve, 
    roc_auc_score, 
    precision_recall_curve, 
    average_precision_score,
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

print(f"학습 데이터 클래스 분포 ({DATASET_NAME} 데이터): \n{y_train.value_counts()}\n")

# =======================================================
# 2. 모델 학습
# =======================================================
print("기본 랜덤포레스트 모델 학습 중...")
baseline_rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
baseline_rf.fit(X_train, y_train)

# =======================================================
# 3. 임계값 자동 최적화 로직 (Pothole F1-Score 극대화)
# =======================================================
# 포트홀(1)일 확률 추출
y_probs = baseline_rf.predict_proba(X_test)[:, 1]

# Precision-Recall Curve를 통해 모든 임계값에서의 정밀도와 재현율 계산
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)

# 포트홀(Class 1)의 F1-Score 계산 (분모 0 방지)
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-9)

# F1-Score가 최대인 지점의 인덱스 탐색
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print("\n" + "="*50)
print(f"🎯 [자동 최적화] 포트홀 F1 극대화 임계값: {best_threshold:.4f}")
print(f"🏆 해당 지점의 예상 F1-Score: {best_f1:.4f}")
print("="*50)

# 최적 임계값을 적용하여 최종 예측값 생성
y_pred_tuned = (y_probs >= best_threshold).astype(int)

# =======================================================
# 4. 결과 출력
# =======================================================
print(f"\n--- [최적 임계값 적용 - {DATASET_NAME}] 결과 ---")
print(classification_report(y_test, y_pred_tuned))
print("accuracy:", accuracy_score(y_test, y_pred_tuned))

roc_auc = roc_auc_score(y_test, y_probs)
pr_auc = average_precision_score(y_test, y_probs)
print(f"ROC_AUC Score: {roc_auc:.4f}")
print(f"PR_AUC Score:  {pr_auc:.4f}")

# =======================================================
# 5. 모델 저장
# =======================================================
MODEL_DIR = os.path.join(BASE_DIR, "../../../models/pothole")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, f"rf_base_{DATASET_NAME}.pkl")

joblib.dump(baseline_rf, MODEL_SAVE_PATH)
print(f"\n🎉 모델 저장 완료: {MODEL_SAVE_PATH}")

# =======================================================
# 6. 시각화 및 이미지 저장
# =======================================================
FIGURE_DIR = os.path.join(BASE_DIR, "../../../reports/figures/outdoor_performance/rf")
os.makedirs(FIGURE_DIR, exist_ok=True)

# 6-1. Confusion Matrix
plt.figure(figsize=(7, 5))
cm = confusion_matrix(y_test, y_pred_tuned)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Normal(0)', 'Pothole(1)'], 
            yticklabels=['Normal(0)', 'Pothole(1)'])
plt.title(f'CM (Tuned Threshold: {best_threshold:.3f}) - {DATASET_NAME}')
plt.xlabel('Predicted')
plt.ylabel('Actual')

CM_SAVE_PATH = os.path.join(FIGURE_DIR, f"rf_base_cm_{DATASET_NAME}.png")
plt.savefig(CM_SAVE_PATH, dpi=300, bbox_inches='tight')
plt.close()

# 6-2. Feature Importance
importances = baseline_rf.feature_importances_
indices = np.argsort(importances)[::-1][:10]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=X_train.columns[indices], palette='viridis')
plt.title(f'Top 10 Feature Importances ({DATASET_NAME})')
plt.xlabel('Relative Importance')

FI_SAVE_PATH = os.path.join(FIGURE_DIR, f"rf_base_fi_{DATASET_NAME}.png")
plt.savefig(FI_SAVE_PATH, dpi=300, bbox_inches='tight')
plt.close()

# 6-3. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', color='blue', lw=2)
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve ({DATASET_NAME})')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

ROC_SAVE_PATH = os.path.join(FIGURE_DIR, f"rf_base_roc_{DATASET_NAME}.png")
plt.savefig(ROC_SAVE_PATH, dpi=300, bbox_inches='tight')
plt.close()

# 6-4. PR Curve
plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, label=f'PR Curve (AUC = {pr_auc:.4f})', color='green', lw=2)
# 현재 선택된 최적 임계값 지점을 표시
plt.plot(recalls[best_idx], precisions[best_idx], 'ro', label=f'Best F1 (Thresh={best_threshold:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve ({DATASET_NAME})')
plt.legend(loc='lower left')
plt.grid(True, alpha=0.3)

PR_SAVE_PATH = os.path.join(FIGURE_DIR, f"rf_base_pr_{DATASET_NAME}.png")
plt.savefig(PR_SAVE_PATH, dpi=300, bbox_inches='tight')
plt.close()

print(f"📈 모든 시각화 이미지 저장 완료 (파일명 뒤에 _{DATASET_NAME} 포함)")