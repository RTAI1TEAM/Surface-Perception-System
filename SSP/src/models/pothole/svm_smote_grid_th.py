import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    confusion_matrix,
    roc_curve, 
    roc_auc_score, 
    precision_recall_curve, 
    average_precision_score,
    make_scorer,    # 💡 추가: 커스텀 채점관을 만들기 위해 필요
    f1_score        # 💡 추가: F1 점수 계산용
)
from imblearn.over_sampling import SMOTE  # SMOTE 추가
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

# 2. 특징 데이터(X)와 정답 레이블(y) 분리
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']

X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

print(f"SMOTE 적용 전 학습 데이터 클래스 분포 ({DATASET_NAME}): \n{y_train.value_counts()}\n")

# ==========================================
# 3. 데이터 스케일링 & SMOTE 적용
# ==========================================
# SMOTE는 거리(KNN) 기반이므로, 반드시 스케일링을 먼저 진행해야 합니다.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE 적용 (포트홀 데이터를 정상 데이터 수만큼 증식)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print(f"SMOTE 적용 후 학습 데이터 클래스 분포: \n{y_train_resampled.value_counts()}\n")

# ==========================================
# 4. GridSearchCV를 이용한 최적의 SVM 모델 학습
# ==========================================
print("최적의 하이퍼파라미터 탐색 중 (GridSearchCV)... 시간이 다소 소요될 수 있습니다.")

# 💡 핵심 1: 포트홀(Class 1)의 F1 점수만 바라보는 전용 채점관 생성
pothole_f1_scorer = make_scorer(f1_score, pos_label=1)

param_grid = {
    'C': [0.1, 1, 10, 100],               # 오차 허용도
    'gamma': ['scale', 0.01, 0.1, 1],     # 데이터 영향력 거리
    'kernel': ['rbf', 'linear']           # 경계선 모양
}

base_svm = SVC(probability=True, random_state=42)

grid_search = GridSearchCV(
    estimator=base_svm, 
    param_grid=param_grid, 
    cv=5, 
    scoring=pothole_f1_scorer, # 💡 일반 f1 대신 포트홀 전용 채점관 투입!
    n_jobs=-1, 
    verbose=1
)

grid_search.fit(X_train_resampled, y_train_resampled)

print("\n=== GridSearchCV 탐색 완료 ===")
print("가장 성능이 좋은 조합:", grid_search.best_params_)

# 최적 모델 추출
best_svm_model = grid_search.best_estimator_

# ==========================================
# 5. 확률 계산 및 최적 임계값(Threshold) 찾기
# ==========================================
y_probs = best_svm_model.predict_proba(X_test_scaled)[:, 1] # 포트홀(1)일 확률

# 💡 핵심 2: Precision-Recall Curve를 이용해 포트홀 F1-Score가 최대가 되는 지점 찾기
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)

# 모든 Threshold에 대한 F1-Score 계산 (분모 0 방지)
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-9)

# F1-Score가 가장 높은 인덱스 찾기
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_pothole_f1 = f1_scores[best_idx]

print("\n" + "="*50)
print(f"🎯 포트홀(1) F1-Score 극대화를 위한 최적 Threshold: {best_threshold:.4f}")
print(f"🏆 예상되는 포트홀(1) F1-Score 최대치: {best_pothole_f1:.4f}")
print("="*50)

# 최적 임계값을 적용하여 최종 예측 (기본 0.5 사용 안 함)
y_pred_optimal = (y_probs >= best_threshold).astype(int)

# 6. 결과 평가
print(f"\n=== 최종 SVM (SMOTE + GridSearch, Threshold: {best_threshold:.4f}) 평가 결과 ===")
print(classification_report(y_test, y_pred_optimal, target_names=['Normal(0)', 'Pothole(1)']))
print(f"Accuracy: {accuracy_score(y_test, y_pred_optimal):.4f}")

# AUC 점수 계산 및 출력
roc_auc = roc_auc_score(y_test, y_probs)
pr_auc = average_precision_score(y_test, y_probs)
print(f"ROC_AUC Score: {roc_auc:.4f}")
print(f"PR_AUC Score:  {pr_auc:.4f}")

# ==========================================
# 7. 모델 및 스케일러 저장
# ==========================================
MODEL_DIR = os.path.join(BASE_DIR, "../../../models/pothole")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(MODEL_DIR, f"svm_smote_grid_{DATASET_NAME}.pkl")
SCALER_SAVE_PATH = os.path.join(MODEL_DIR, f"svm_scaler_{DATASET_NAME}.pkl")

joblib.dump(best_svm_model, MODEL_SAVE_PATH)
joblib.dump(scaler, SCALER_SAVE_PATH)

print(f"\n🎉 모델 저장 완료: {MODEL_SAVE_PATH}")
print(f"🎉 스케일러 저장 완료: {SCALER_SAVE_PATH}")

# ==========================================
# 8. 혼동 행렬 시각화 및 이미지 저장
# ==========================================
FIGURE_DIR = os.path.join(BASE_DIR, "../../../reports/figures/outdoor_performance/svm")
os.makedirs(FIGURE_DIR, exist_ok=True)

cm = confusion_matrix(y_test, y_pred_optimal)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Pothole'], 
            yticklabels=['Normal', 'Pothole'])
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
# 제목에 최적 Threshold 명시
plt.title(f'SMOTE+Grid SVM CM (Threshold: {best_threshold:.3f})')

CM_SAVE_PATH = os.path.join(FIGURE_DIR, f"svm_smote_grid_cm_{DATASET_NAME}.png")
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
plt.plot([0, 1], [0, 1], 'k--', lw=2) 
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve (SMOTE+Grid SVM - {DATASET_NAME})')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

ROC_SAVE_PATH = os.path.join(FIGURE_DIR, f"svm_smote_grid_roc_{DATASET_NAME}.png")
plt.savefig(ROC_SAVE_PATH, dpi=300, bbox_inches='tight')
print(f"📈 ROC Curve 이미지 저장 완료: {ROC_SAVE_PATH}")
plt.close()

# --- Precision-Recall (PR) Curve ---
precisions, recalls, _ = precision_recall_curve(y_test, y_probs)
plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, label=f'PR Curve (AUC = {pr_auc:.4f})', color='green', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve (SMOTE+Grid SVM - {DATASET_NAME})')
plt.legend(loc='lower left')
plt.grid(True, alpha=0.3)

PR_SAVE_PATH = os.path.join(FIGURE_DIR, f"svm_smote_grid_pr_{DATASET_NAME}.png")
plt.savefig(PR_SAVE_PATH, dpi=300, bbox_inches='tight')
print(f"📈 PR Curve 이미지 저장 완료: {PR_SAVE_PATH}")
plt.close()