import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE 
import joblib  # 모델 저장을 위해 추가
import os

# 1. 데이터 로드
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../../../data/processed/pothole")
TRAIN_PATH = os.path.join(DATA_DIR, "train_v2.csv")
TEST_PATH = os.path.join(DATA_DIR, "test_v2.csv")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# 학습/테스트 데이터 분리
X_train_raw = train_df.drop('label', axis=1)
y_train_raw = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# 2. SMOTE 적용 (데이터를 1:1 비율로 증식)
print(f"SMOTE 적용 전 클래스 분포: \n{y_train_raw.value_counts()}")
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train_raw, y_train_raw)
print(f"SMOTE 적용 후 클래스 분포: \n{y_train.value_counts()}")

# 3. 모델 학습
print("\nSMOTE 적용 랜덤포레스트 모델 학습 중...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)

# =======================================================
# 4. 모델 저장 (models/pothole/rf_smote.pkl)
# =======================================================
MODEL_DIR = os.path.join(BASE_DIR, "../../../models/pothole")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "rf_smote.pkl")

joblib.dump(rf_model, MODEL_SAVE_PATH)
print(f"\n🎉 모델 저장 완료: {MODEL_SAVE_PATH}")

# 5. 예측 및 임계값 조정
y_probs = rf_model.predict_proba(X_test)[:, 1]
threshold = 0.3 # 포트홀 탐지 최적화 임계값
y_pred_adjusted = (y_probs >= threshold).astype(int)

# 결과 출력
print("\n--- [전략 A] SMOTE 적용 모델 결과 ---")
print(classification_report(y_test, y_pred_adjusted))
print("accuracy:", accuracy_score(y_test, y_pred_adjusted))

# =======================================================
# 6. 시각화 및 이미지 저장 (Confusion Matrix)
# =======================================================
plt.figure(figsize=(7, 5))
cm = confusion_matrix(y_test, y_pred_adjusted)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=['Normal(0)', 'Pothole(1)'], 
            yticklabels=['Normal(0)', 'Pothole(1)'])
plt.title(f'Confusion Matrix (SMOTE + Threshold: {threshold})')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# 이미지 저장 설정
FIGURE_DIR = os.path.join(BASE_DIR, "../../../reports/figures/outdoor_performance/rf")
os.makedirs(FIGURE_DIR, exist_ok=True)
CM_SAVE_PATH = os.path.join(FIGURE_DIR, "rf_smote_cm.png")

plt.savefig(CM_SAVE_PATH, dpi=300, bbox_inches='tight')
print(f"📈 CM 이미지 저장 완료: {CM_SAVE_PATH}")
plt.show()

# =======================================================
# 7. 변수 중요도 시각화 및 저장 (Feature Importance)
# =======================================================
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1][:10] # 상위 10개

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=X_train.columns[indices], palette='viridis')
plt.title('Top 10 Feature Importances (SMOTE RF)')
plt.xlabel('Relative Importance')
plt.ylabel('Features')

FI_SAVE_PATH = os.path.join(FIGURE_DIR, "rf_smote_fi.png")
plt.savefig(FI_SAVE_PATH, dpi=300, bbox_inches='tight')
print(f"📈 FI 이미지 저장 완료: {FI_SAVE_PATH}")
plt.show()