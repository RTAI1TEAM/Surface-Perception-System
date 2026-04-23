import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib  # 모델 저장을 위해 추가된 모듈
import os

# 1. 데이터 로드
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../../../data/processed/pothole")
TRAIN_PATH = os.path.join(DATA_DIR, "train_v2.csv")
TEST_PATH = os.path.join(DATA_DIR, "test_v2.csv")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# 학습/테스트 데이터 분리
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# 원본 데이터의 심각한 불균형 상태를 확인합니다.
print(f"학습 데이터 클래스 분포 (불균형 상태 그대로): \n{y_train.value_counts()}\n")

# =======================================================
# 2. 기본 모델 학습 (아무런 처리 없음)
# =======================================================
print("기본 랜덤포레스트 모델 학습 중...")

# class_weight='balanced' 옵션도 없고, SMOTE도 적용하지 않은 순정 상태입니다.
baseline_rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
baseline_rf.fit(X_train, y_train)

# =======================================================
# 3. 예측 (기본 임계값 0.5 적용)
# =======================================================
# 특별히 임계값을 낮추지 않고 모델의 기본 판단(0.5 기준)에 맡깁니다.
y_pred = baseline_rf.predict(X_test)

# 4. 결과 출력
print("\n--- [기본 모델 (Baseline)] 결과 ---")
print(classification_report(y_test, y_pred))
print("accuracy:", accuracy_score(y_test, y_pred))

# =======================================================
# 5. 모델 저장 (joblib 사용)
# =======================================================
MODEL_DIR = os.path.join(BASE_DIR, "../../../models/pothole")
os.makedirs(MODEL_DIR, exist_ok=True) # 폴더가 없으면 자동 생성
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "rf_base.pkl")

joblib.dump(baseline_rf, MODEL_SAVE_PATH)
print(f"\n🎉 베이스라인 모델이 성공적으로 저장되었습니다:\n👉 {MODEL_SAVE_PATH}")


# =======================================================
# 6. 시각화 및 이미지 저장 (Confusion Matrix)
# =======================================================
plt.figure(figsize=(7, 5))
cm = confusion_matrix(y_test, y_pred)
# 기본 모델의 경각심을 주기 위해 붉은색 톤(Reds)으로 설정했습니다.
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Normal(0)', 'Pothole(1)'], 
            yticklabels=['Normal(0)', 'Pothole(1)'])
plt.title('Confusion Matrix (Baseline - No Treatment)')
plt.xlabel('Predicted (0: Normal, 1: Pothole)')
plt.ylabel('Actual')

# --- 이미지 저장 설정 ---
FIGURE_DIR = os.path.join(BASE_DIR, "../../../reports/figures/outdoor_performance/rf")
os.makedirs(FIGURE_DIR, exist_ok=True) # 폴더가 없으면 자동 생성
CM_SAVE_PATH = os.path.join(FIGURE_DIR, "rf_base_cm.png")

# 반드시 plt.show() 전에 저장해야 합니다!
plt.savefig(CM_SAVE_PATH, dpi=300, bbox_inches='tight')
print(f"📈 혼동 행렬 이미지가 성공적으로 저장되었습니다:\n👉 {CM_SAVE_PATH}\n")
# 화면에 혼동 행렬 띄우기
plt.show()

# =======================================================
# 7. 변수 중요도(Feature Importance) 시각화 및 저장
# =======================================================
importances = baseline_rf.feature_importances_
indices = np.argsort(importances)[::-1][:10] # 상위 10개 변수 추출

plt.figure(figsize=(10, 6))
# 미래 버전 호환성을 위해 x, y 인자를 명시적으로 지정
sns.barplot(x=importances[indices], y=X_train.columns[indices], palette='viridis')
plt.title('Top 10 Feature Importances (Baseline RF)')
plt.xlabel('Relative Importance')
plt.ylabel('Sensor Features')

# 요청하신 파일명인 'rf_base_fi.png'로 저장합니다.
FI_SAVE_PATH = os.path.join(FIGURE_DIR, "rf_base_fi.png")
plt.savefig(FI_SAVE_PATH, dpi=300, bbox_inches='tight')
print(f"📈 변수 중요도 이미지가 성공적으로 저장되었습니다:\n👉 {FI_SAVE_PATH}\n")

# 화면에 변수 중요도 그래프 띄우기
plt.show()