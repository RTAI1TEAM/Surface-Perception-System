import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib  # 모델 및 스케일러 저장을 위해 추가
import os

# 1. 데이터 로드 및 경로 설정
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
# 4. GridSearchCV를 이용한 최적의 SVM 모델 생성
# ==========================================
print("최적의 하이퍼파라미터 탐색 중 (GridSearchCV)... 시간이 조금 걸릴 수 있습니다.")

param_grid = {
    'C': [0.1, 1, 10, 100],               # 오차 허용도
    'gamma': ['scale', 0.01, 0.1, 1],     # 데이터 영향력 거리
    'kernel': ['rbf', 'linear']           # 경계선 모양
}

base_svm = SVC(random_state=42, probability=True) # 확률 예측을 위해 probability=True 권장

grid_search = GridSearchCV(
    estimator=base_svm, 
    param_grid=param_grid, 
    cv=5, 
    scoring='accuracy', 
    n_jobs=-1, 
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print("\n=== GridSearchCV 탐색 완료 ===")
print("가장 성능이 좋은 조합:", grid_search.best_params_)
print(f"학습 데이터 교차 검증 정확도: {grid_search.best_score_:.4f}\n")

# 최적 모델 추출
best_svm_model = grid_search.best_estimator_

# ==========================================
# 5. 모델 및 스케일러 저장 (models/pothole/)
# ==========================================
MODEL_DIR = os.path.join(BASE_DIR, "../../../models/pothole")
os.makedirs(MODEL_DIR, exist_ok=True)

# 모델 저장
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "svm_grid.pkl")
joblib.dump(best_svm_model, MODEL_SAVE_PATH)

# 스케일러 저장 (SVM은 새로운 데이터를 예측할 때도 동일한 스케일러가 반드시 필요함)
SCALER_SAVE_PATH = os.path.join(MODEL_DIR, "svm_grid_scaler.pkl")
joblib.dump(scaler, SCALER_SAVE_PATH)

print(f"🎉 모델 저장 완료: {MODEL_SAVE_PATH}")
print(f"🎉 스케일러 저장 완료: {SCALER_SAVE_PATH}")

# 6. 테스트 데이터로 예측
y_pred = best_svm_model.predict(X_test_scaled)

# 7. 결과 평가
print("\n=== 최적화된 SVM 최종 평가 결과 ===")
print(f"Accuracy (정확도): {accuracy_score(y_test, y_pred):.4f}")
print("\n=== 분류 상세 리포트 ===")
print(classification_report(y_test, y_pred, target_names=['Normal(0)', 'Pothole(1)']))

# ==========================================
# 8. 혼동 행렬 시각화 및 이미지 저장 (reports/figures/)
# ==========================================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Pothole'], 
            yticklabels=['Normal', 'Pothole'])
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title('Optimized SVM Confusion Matrix')

# 이미지 저장 설정
FIGURE_DIR = os.path.join(BASE_DIR, "../../../reports/figures/outdoor_performance/svm")
os.makedirs(FIGURE_DIR, exist_ok=True)
FIGURE_SAVE_PATH = os.path.join(FIGURE_DIR, "svm_grid_cm.png")

# 반드시 plt.show() 전에 저장
plt.savefig(FIGURE_SAVE_PATH, dpi=300, bbox_inches='tight')
print(f"📈 CM 이미지 저장 완료: {FIGURE_SAVE_PATH}\n")

plt.show()