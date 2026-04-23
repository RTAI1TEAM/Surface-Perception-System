import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib  # 모델 저장을 위해 추가
import os

# 1. 데이터 로드
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../../../data/processed/pothole")
TRAIN_PATH = os.path.join(DATA_DIR, "train_v2.csv")
TEST_PATH = os.path.join(DATA_DIR, "test_v2.csv")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# 2. 특징 데이터(X)와 정답 레이블(y) 분리
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']

X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# 3. 데이터 스케일링 (SVM 필수 과정)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 4. 기본 SVM 모델 생성 및 학습 (Baseline)
# ==========================================
print("기본 SVM 모델 학습 중 (Baseline)...")
baseline_svm_model = SVC(kernel='rbf', C=1.0, random_state=42)
baseline_svm_model.fit(X_train_scaled, y_train)

# ==========================================
# 5. 모델 저장 (models/pothole/svm_base.pkl)
# ==========================================
MODEL_DIR = os.path.join(BASE_DIR, "../../../models/pothole")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "svm_base.pkl")

# 모델과 함께 스케일러도 저장하는 것이 좋습니다 (나중에 새로운 데이터 예측 시 필요)
joblib.dump(baseline_svm_model, MODEL_SAVE_PATH)
# 스케일러도 저장 (선택 사항이지만 권장)
joblib.dump(scaler, os.path.join(MODEL_DIR, "svm_scaler.pkl"))

print(f"🎉 모델 저장 완료: {MODEL_SAVE_PATH}")

# 6. 테스트 데이터로 예측
y_pred = baseline_svm_model.predict(X_test_scaled)

# 7. 결과 평가
print("\n=== 베이스라인 SVM 최종 평가 결과 ===")
print(f"Accuracy (정확도): {accuracy_score(y_test, y_pred):.4f}")
print("\n=== 분류 상세 리포트 ===")
print(classification_report(y_test, y_pred, target_names=['Normal(0)', 'Pothole(1)']))

# ==========================================
# 8. 혼동 행렬 시각화 및 이미지 저장 (reports/figures/svm_base_cm.png)
# ==========================================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Pothole'], 
            yticklabels=['Normal', 'Pothole'])
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title('Baseline SVM Confusion Matrix')

# 이미지 저장 설정
FIGURE_DIR = os.path.join(BASE_DIR, "../../../reports/figures/outdoor_performance/svm")
os.makedirs(FIGURE_DIR, exist_ok=True)
CM_SAVE_PATH = os.path.join(FIGURE_DIR, "svm_base_cm.png")

plt.savefig(CM_SAVE_PATH, dpi=300, bbox_inches='tight')
print(f"📈 CM 이미지 저장 완료: {CM_SAVE_PATH}")

plt.show()