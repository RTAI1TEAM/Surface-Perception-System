import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

print(f"✅ 학습 데이터 로드 완료 ({DATASET_NAME})")
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
print("🚀 최적의 하이퍼파라미터 탐색 중 (GridSearchCV)...")

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
print(f"📊 [최종 보고서 - Logistic Regression] Dataset: {DATASET_NAME}")
print("="*60)
print(f"✅ 최적 임계값 (Optimal Threshold): {best_threshold:.4f}")
print(f"✅ 예상 최대 F1-Score (Pothole): {f1_scores[best_idx]:.4f}")
print("-" * 60)
print(classification_report(y_test, y_pred_tuned, target_names=['Normal(0)', 'Pothole(1)']))
print("-" * 60)

print(f"accuracy: {accuracy_score(y_test, y_pred_tuned):.4f}")

roc_auc = roc_auc_score(y_test, y_probs)
pr_auc = average_precision_score(y_test, y_probs)
print(f"ROC_AUC Score: {roc_auc:.4f}")
print(f"PR_AUC Score:  {pr_auc:.4f}")
print("="*60)

# =======================================================
# 5. 모델 저장 및 시각화
# =======================================================
MODEL_DIR = os.path.join(BASE_DIR, "../../../models/pothole")
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(best_logistic_model, os.path.join(MODEL_DIR, f"logistic_class_grid_{DATASET_NAME}.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, f"logistic_scaler_{DATASET_NAME}.pkl"))

FIGURE_DIR = os.path.join(BASE_DIR, "../../../reports/figures/outdoor_performance/logistic")
os.makedirs(FIGURE_DIR, exist_ok=True)

# 5-1. Confusion Matrix
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test, y_pred_tuned)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greys',
            xticklabels=['Normal', 'Pothole'], yticklabels=['Normal', 'Pothole'])
plt.title(f'Logistic CM (Thresh: {best_threshold:.4f})')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig(os.path.join(FIGURE_DIR, f"logistic_class_grid_cm_{DATASET_NAME}.png"), dpi=300, bbox_inches='tight')
plt.close()

# 5-2. Feature Importance (Coefficients)
# 로지스틱 회귀는 변수의 중요도를 '회귀 계수(coef_)'의 절대값 크기로 판단합니다.
coefficients = best_logistic_model.coef_[0]
abs_coefs = np.abs(coefficients)
indices = np.argsort(abs_coefs)[::-1][:10] # 절대값이 큰 상위 10개 추출

plt.figure(figsize=(10, 6))
# 양수(파란색: 포트홀 확률 증가), 음수(빨간색: 정상 확률 증가) 구분
colors = ['blue' if c > 0 else 'red' for c in coefficients[indices]]
sns.barplot(x=abs_coefs[indices], y=X_train.columns[indices], palette=colors)
plt.title(f'Top 10 Feature Coefficients (Logistic - {DATASET_NAME})\nBlue: (+) Pothole, Red: (-) Normal')
plt.xlabel('Absolute Coefficient Value')
plt.savefig(os.path.join(FIGURE_DIR, f"logistic_class_grid_fi_{DATASET_NAME}.png"), dpi=300, bbox_inches='tight')
plt.close()

# 5-3. PR Curve
plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, label=f'AP = {pr_auc:.4f}', color='gray')
plt.plot(recalls[best_idx], precisions[best_idx], 'ro', label=f'Optimal (Thresh={best_threshold:.4f})')
plt.title(f'Logistic Precision-Recall Curve ({DATASET_NAME})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.savefig(os.path.join(FIGURE_DIR, f"logistic_class_grid_pr_{DATASET_NAME}.png"), dpi=300, bbox_inches='tight')
plt.close()

# 5-4. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}', color='black')
plt.plot([0, 1], [0, 1], 'k--')
plt.title(f'Logistic ROC Curve ({DATASET_NAME})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.savefig(os.path.join(FIGURE_DIR, f"logistic_class_grid_roc_{DATASET_NAME}.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"\n📈 로지스틱 모델과 4종의 시각화 이미지가 성공적으로 저장되었습니다.")