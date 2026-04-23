import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE # SMOTE 임포트
import os

# 1. 데이터 로드
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../../../data/processed/pothole")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# 학습/테스트 데이터 분리
X_train_raw = train_df.drop('label', axis=1)
y_train_raw = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# 2. SMOTE 적용 (데이터를 1:1 비율로 증식)
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train_raw, y_train_raw)

print(f"SMOTE 적용 전 클래스 분포: \n{y_train_raw.value_counts()}")
print(f"SMOTE 적용 후 클래스 분포: \n{y_train.value_counts()}")

# 3. 모델 학습 (데이터가 이미 1:1이므로 class_weight는 쓰지 않음)
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)

# 4. 예측 및 임계값 조정
y_probs = rf_model.predict_proba(X_test)[:, 1]
threshold = 0.3 # 포트홀 탐지를 위해 낮게 설정
y_pred_adjusted = (y_probs >= threshold).astype(int)

# 5. 결과 출력
print("\n--- [전략 A] SMOTE 적용 모델 결과 ---")
print(classification_report(y_test, y_pred_adjusted))

# 6. 시각화 (Confusion Matrix)
plt.figure(figsize=(7, 5))
cm = confusion_matrix(y_test, y_pred_adjusted)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
plt.title(f'Confusion Matrix (SMOTE + Threshold: {threshold})')
plt.xlabel('Predicted (0: Normal, 1: Pothole)')
plt.ylabel('Actual')
plt.show()