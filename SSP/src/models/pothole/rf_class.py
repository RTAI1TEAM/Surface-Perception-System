import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os

# 1. 데이터 로드
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../../../data/processed/pothole")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# 학습/테스트 데이터 분리
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

print(f"원본 학습 데이터 클래스 분포: \n{y_train.value_counts()}")

# 2. 모델 학습 (SMOTE 없이 원본 데이터 사용, 대신 class_weight 추가)
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced', # 원본 데이터 불균형을 스스로 계산해서 보완
    random_state=42
)
rf_model.fit(X_train, y_train)

# 3. 예측 및 임계값 조정
y_probs = rf_model.predict_proba(X_test)[:, 1]
threshold = 0.3 # 포트홀 탐지를 위해 낮게 설정
y_pred_adjusted = (y_probs >= threshold).astype(int)

# 4. 결과 출력
print("\n--- [전략 B] Class Weight 적용 모델 결과 ---")
print(classification_report(y_test, y_pred_adjusted))

# 5. 시각화 (Confusion Matrix)
plt.figure(figsize=(7, 5))
cm = confusion_matrix(y_test, y_pred_adjusted)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges')
plt.title(f'Confusion Matrix (Class Weight + Threshold: {threshold})')
plt.xlabel('Predicted (0: Normal, 1: Pothole)')
plt.ylabel('Actual')
plt.show()