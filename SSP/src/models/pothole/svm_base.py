import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    confusion_matrix,
    roc_curve, 
    roc_auc_score, 
    precision_recall_curve, 
    average_precision_score
)
import joblib  # 모델 저장을 위해 추가
import os

# =======================================================
# 0. 데이터셋 이름 설정 (여기만 바꾸면 전체 적용됩니다!)
# =======================================================
DATASET_NAME = "v3_s2" # 예: 'raw', 'v3', 'v3_s2' 등으로 변경해서 사용하세요.

# 1. 데이터 로드
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../../../data/processed/pothole")

# DATASET_NAME을 반영하여 파일 경로 자동 생성
TRAIN_PATH = os.path.join(DATA_DIR, f"train_{DATASET_NAME}.csv")
TEST_PATH = os.path.join(DATA_DIR, f"test_{DATASET_NAME}.csv")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# 2. 특징 데이터(X)와 정답 레이블(y) 분리
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']

X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

print(f"학습 데이터 클래스 분포 ({DATASET_NAME} 데이터): \n{y_train.value_counts()}\n")

# 3. 데이터 스케일링 (SVM 필수 과정)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 4. 기본 SVM 모델 생성 및 학습 (Class Weight 적용)
# ==========================================
print("Class Weight가 적용된 SVM 모델 학습 중...")

# 불균형 처리를 위해 class_weight='balanced' 추가
# ROC & PR AUC 계산용 확률 예측을 위해 probability=True 추가
baseline_svm_model = SVC(
    kernel='rbf', 
    C=1.0,
    probability=True, 
    random_state=42
)
baseline_svm_model.fit(X_train_scaled, y_train)

# ==========================================
# 5. 예측 및 확률 계산
# ==========================================
y_pred = baseline_svm_model.predict(X_test_scaled)
y_probs = baseline_svm_model.predict_proba(X_test_scaled)[:, 1] # 포트홀(1)일 확률

# 6. 결과 평가
print(f"\n=== 베이스라인 SVM ({DATASET_NAME}) 평가 결과 ===")
print(classification_report(y_test, y_pred, target_names=['Normal(0)', 'Pothole(1)']))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# AUC 점수 계산 및 출력
roc_auc = roc_auc_score(y_test, y_probs)
pr_auc = average_precision_score(y_test, y_probs)
print(f"ROC_AUC Score: {roc_auc:.4f}")
print(f"PR_AUC Score:  {pr_auc:.4f}")

# ==========================================
# 7. 모델 저장 (models/pothole/svm_base_{DATASET_NAME}.pkl)
# ==========================================
MODEL_DIR = os.path.join(BASE_DIR, "../../../models/pothole")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(MODEL_DIR, f"svm_base_{DATASET_NAME}.pkl")
SCALER_SAVE_PATH = os.path.join(MODEL_DIR, f"svm_scaler_{DATASET_NAME}.pkl")

# 모델과 스케일러 저장
joblib.dump(baseline_svm_model, MODEL_SAVE_PATH)
joblib.dump(scaler, SCALER_SAVE_PATH)

print(f"\n🎉 모델 저장 완료: {MODEL_SAVE_PATH}")
print(f"🎉 스케일러 저장 완료: {SCALER_SAVE_PATH}")

# ==========================================
# 8. 혼동 행렬 시각화 및 이미지 저장
# ==========================================
FIGURE_DIR = os.path.join(BASE_DIR, "../../../reports/figures/outdoor_performance/svm")
os.makedirs(FIGURE_DIR, exist_ok=True)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Pothole'], 
            yticklabels=['Normal', 'Pothole'])
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title(f'Baseline SVM Confusion Matrix ({DATASET_NAME})')

CM_SAVE_PATH = os.path.join(FIGURE_DIR, f"svm_base_cm_{DATASET_NAME}.png")
plt.savefig(CM_SAVE_PATH, dpi=300, bbox_inches='tight')
print(f"📈 CM 이미지 저장 완료: {CM_SAVE_PATH}")
plt.close()

# =======================================================
# 9. ROC Curve & PR Curve 시각화 및 저장
# =======================================================

# --- ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', color='blue', lw=2)
plt.plot([0, 1], [0, 1], 'k--', lw=2) # 대각선 기준선
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve (Baseline SVM - {DATASET_NAME})')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

ROC_SAVE_PATH = os.path.join(FIGURE_DIR, f"svm_base_roc_{DATASET_NAME}.png")
plt.savefig(ROC_SAVE_PATH, dpi=300, bbox_inches='tight')
print(f"📈 ROC Curve 이미지 저장 완료: {ROC_SAVE_PATH}")
plt.close()

# --- Precision-Recall (PR) Curve ---
precisions, recalls, _ = precision_recall_curve(y_test, y_probs)
plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, label=f'PR Curve (AUC = {pr_auc:.4f})', color='green', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve (Baseline SVM - {DATASET_NAME})')
plt.legend(loc='lower left')
plt.grid(True, alpha=0.3)

PR_SAVE_PATH = os.path.join(FIGURE_DIR, f"svm_base_pr_{DATASET_NAME}.png")
plt.savefig(PR_SAVE_PATH, dpi=300, bbox_inches='tight')
print(f"📈 PR Curve 이미지 저장 완료: {PR_SAVE_PATH}")
plt.close()