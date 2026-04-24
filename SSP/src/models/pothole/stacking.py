import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 스태킹 및 개별 모델 라이브러리
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# =======================================================
# 1. 데이터 로드 및 경로 설정 (v3_s2 데이터 기준)
# =======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../../../data/processed/pothole")
TRAIN_PATH = os.path.join(DATA_DIR, "train_v3_s2.csv")
TEST_PATH = os.path.join(DATA_DIR, "test_v3_s2.csv")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

print(f"학습 데이터 클래스 분포: \n{y_train.value_counts()}")

# XGBoost를 위한 scale_pos_weight 계산
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_weight = neg_count / pos_count
print(f"XGBoost 적용 scale_pos_weight: {scale_weight:.2f}")

# =======================================================
# 2. 스태킹 베이스 모델 & 메타 모델 정의
# =======================================================
print("\nStacking 베이스 모델 설정 중...")

# Base 1: Random Forest (Class Weight 적용)
# SMOTE 대신 내부 클래스 가중치를 사용하여 스태킹의 안정성을 높입니다.
rf_base = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

# Base 2: SVM (Pipeline 적용)
# SVM은 스케일링이 필수이므로 Pipeline으로 묶어서 전달합니다.
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('svc', SVC(probability=True, random_state=42)) # probability=True 필수
])

# Base 3: XGBoost
xgb_base = XGBClassifier(
    scale_pos_weight=scale_weight, 
    eval_metric='logloss', 
    random_state=42,
    use_label_encoder=False
)

estimators = [
    ('rf', rf_base),
    ('svm_scaled', svm_pipeline),
    ('xgb', xgb_base)
]

# Meta Model: Logistic Regression (결정권자)
meta_model = LogisticRegression()

# 스태킹 앙상블 결합
stacking_clf = StackingClassifier(
    estimators=estimators, 
    final_estimator=meta_model,
    cv=5 # 5-Fold 교차검증으로 과적합 방지
)

# =======================================================
# 3. 스태킹 모델 학습
# =======================================================
print("Stacking 모델 학습 중... (시간이 다소 소요될 수 있습니다)")
stacking_clf.fit(X_train, y_train)

# =======================================================
# 4. 모델 저장 (models/pothole/)
# =======================================================
MODEL_DIR = os.path.join(BASE_DIR, "../../../models/pothole")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "stacking_best.pkl")

# 파이프라인이 포함된 스태킹 모델 전체를 통째로 저장합니다.
# 나중에 불러올 때 별도의 스케일러 적용 없이 원본 데이터를 바로 넣으면 됩니다.
joblib.dump(stacking_clf, MODEL_SAVE_PATH)
print(f"\n🎉 모델 저장 완료: {MODEL_SAVE_PATH}")

# =======================================================
# 5. 예측 및 평가 (임계값 0.5 기준)
# =======================================================
y_probs = stacking_clf.predict_proba(X_test)[:, 1]
threshold = 0.15 # 스태킹은 확률값이 보정되므로 기본 0.5부터 시작
y_pred_adjusted = (y_probs >= threshold).astype(int)

print(f"\n--- [최종 완성] Stacking 모델 평가 결과 (Threshold: {threshold}) ---")
print(classification_report(y_test, y_pred_adjusted, target_names=['Normal(0)', 'Pothole(1)']))
print(f"Accuracy: {accuracy_score(y_test, y_pred_adjusted):.4f}")

# =======================================================
# 6. 혼동 행렬 시각화 및 저장
# =======================================================
FIGURE_DIR = os.path.join(BASE_DIR, "../../../reports/figures/outdoor_performance/stacking")
os.makedirs(FIGURE_DIR, exist_ok=True)

plt.figure(figsize=(7, 5))
cm = confusion_matrix(y_test, y_pred_adjusted)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', # 차별화를 위해 보라색 계열 사용
            xticklabels=['Normal(0)', 'Pothole(1)'], 
            yticklabels=['Normal(0)', 'Pothole(1)'])
plt.title(f'Stacking Confusion Matrix (Threshold: {threshold})')
plt.xlabel('Predicted')
plt.ylabel('Actual')

CM_SAVE_PATH = os.path.join(FIGURE_DIR, "stacking_cm.png")
plt.savefig(CM_SAVE_PATH, dpi=300, bbox_inches='tight')
print(f"📈 CM 이미지 저장 완료: {CM_SAVE_PATH}")
plt.close()

# =======================================================
# 7. 메타 모델 가중치 (어떤 모델의 의견을 가장 신뢰했는가?) 시각화
# =======================================================
meta_coefs = stacking_clf.final_estimator_.coef_[0]
base_model_names = [name for name, _ in estimators]

plt.figure(figsize=(8, 5))
sns.barplot(x=base_model_names, y=meta_coefs, palette='Set2')
plt.title('Meta-Model Coefficients (Base Model Trust Factor)')
plt.xlabel('Base Models')
plt.ylabel('Logistic Regression Coefficient')

COEF_SAVE_PATH = os.path.join(FIGURE_DIR, "stacking_meta_weights.png")
plt.savefig(COEF_SAVE_PATH, dpi=300, bbox_inches='tight')
print(f"📈 메타 가중치 이미지 저장 완료: {COEF_SAVE_PATH}")
plt.show()