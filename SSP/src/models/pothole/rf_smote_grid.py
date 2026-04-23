import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib  # 모델 저장을 위해 추가
import os

# 1. 데이터 로드 및 경로 설정
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

print(f"원본 학습 데이터 클래스 분포: \n{y_train_raw.value_counts()}")

# =======================================================
# 2. 파이프라인 및 GridSearchCV 설정
# =======================================================
pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('rf', RandomForestClassifier(random_state=42))
])

param_grid = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [None, 10, 20],
    'rf__min_samples_split': [2, 5, 10]
}

print("\n최적의 하이퍼파라미터를 탐색 중입니다 (GridSearchCV)...")
grid_search = GridSearchCV(
    estimator=pipeline, 
    param_grid=param_grid, 
    cv=5, 
    scoring='f1', 
    n_jobs=-1, 
    verbose=1
)

grid_search.fit(X_train_raw, y_train_raw)
best_model = grid_search.best_estimator_

print(f"\n최적 파라미터 조합: {grid_search.best_params_}")
print(f"최고 교차 검증 F1 점수: {grid_search.best_score_:.4f}")

# =======================================================
# 3. 모델 저장 (models/pothole/rf_smote_grid.pkl)
# =======================================================
MODEL_DIR = os.path.join(BASE_DIR, "../../../models/pothole")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "rf_smote_grid.pkl")

joblib.dump(best_model, MODEL_SAVE_PATH)
print(f"\n🎉 모델 저장 완료: {MODEL_SAVE_PATH}")

# =======================================================
# 4. 테스트 데이터 예측 및 임계값 조정
# =======================================================
y_probs = best_model.predict_proba(X_test)[:, 1]
threshold = 0.3 
y_pred_adjusted = (y_probs >= threshold).astype(int)

print(f"\n--- [최종 완성] SMOTE + GridSearchCV (Threshold: {threshold}) 결과 ---")
print(classification_report(y_test, y_pred_adjusted))
print("accuracy:", accuracy_score(y_test, y_pred_adjusted))

# =======================================================
# 5. 혼동 행렬 시각화 및 저장 (reports/figures/rf_smote_grid_cm.png)
# =======================================================
plt.figure(figsize=(7, 5))
cm = confusion_matrix(y_test, y_pred_adjusted)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=['Normal(0)', 'Pothole(1)'], 
            yticklabels=['Normal(0)', 'Pothole(1)'])
plt.title(f'Confusion Matrix (Optimized SMOTE + Threshold: {threshold})')
plt.xlabel('Predicted')
plt.ylabel('Actual')

FIGURE_DIR = os.path.join(BASE_DIR, "../../../reports/figures/outdoor_performance/rf")
os.makedirs(FIGURE_DIR, exist_ok=True)
CM_SAVE_PATH = os.path.join(FIGURE_DIR, "rf_smote_grid_cm.png")

plt.savefig(CM_SAVE_PATH, dpi=300, bbox_inches='tight')
print(f"📈 CM 이미지 저장 완료: {CM_SAVE_PATH}")
plt.show()

# =======================================================
# 6. 변수 중요도 시각화 및 저장 (reports/figures/rf_smote_grid_fi.png)
# =======================================================
best_rf = best_model.named_steps['rf']
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1][:10]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=X_train_raw.columns[indices], palette='viridis')
plt.title('Top 10 Feature Importances (Optimized SMOTE + RF)')
plt.xlabel('Relative Importance')
plt.ylabel('Features')

FI_SAVE_PATH = os.path.join(FIGURE_DIR, "rf_smote_grid_fi.png")
plt.savefig(FI_SAVE_PATH, dpi=300, bbox_inches='tight')
print(f"📈 FI 이미지 저장 완료: {FI_SAVE_PATH}")
plt.show()