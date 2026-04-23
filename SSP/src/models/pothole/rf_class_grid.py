import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

# 1. 데이터 로드 및 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../../../data/processed/pothole")
TRAIN_PATH = os.path.join(DATA_DIR, "train_v2.csv")
TEST_PATH = os.path.join(DATA_DIR, "test_v2.csv")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

print(f"학습 데이터 클래스 분포: \n{y_train.value_counts()}\n")

# ==========================================
# 2. GridSearchCV (Class Weight: Balanced)
# ==========================================
print("최적의 하이퍼파라미터 탐색 중 (GridSearchCV)...")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# 불균형 해소를 위해 class_weight='balanced' 설정
rf_base = RandomForestClassifier(class_weight='balanced', random_state=42)

grid_search = GridSearchCV(
    estimator=rf_base, 
    param_grid=param_grid, 
    cv=5, 
    scoring='f1', 
    n_jobs=-1, 
    verbose=1
)

grid_search.fit(X_train, y_train)
best_rf_model = grid_search.best_estimator_

print(f"\n최적 파라미터: {grid_search.best_params_}")
print(f"최고 F1 점수: {grid_search.best_score_:.4f}")

# ==========================================
# 3. 모델 저장 (models/pothole/)
# ==========================================
MODEL_DIR = os.path.join(BASE_DIR, "../../../models/pothole")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "rf_class_grid.pkl")

joblib.dump(best_rf_model, MODEL_SAVE_PATH)
print(f"🎉 모델 저장 완료: {MODEL_SAVE_PATH}")

# ==========================================
# 4. 예측 및 임계값(Threshold) 조정
# ==========================================
y_probs = best_rf_model.predict_proba(X_test)[:, 1]
threshold = 0.3 
y_pred_adjusted = (y_probs >= threshold).astype(int)

print(f"\n--- [결과] Class Weight + GridSearch (Threshold: {threshold}) ---")
print(classification_report(y_test, y_pred_adjusted))

# ==========================================
# 5. 혼동 행렬 시각화 및 저장 (reports/figures/)
# ==========================================
plt.figure(figsize=(7, 5))
cm = confusion_matrix(y_test, y_pred_adjusted)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Normal(0)', 'Pothole(1)'], 
            yticklabels=['Normal(0)', 'Pothole(1)'])
plt.title(f'Confusion Matrix (CW + Grid + Threshold: {threshold})')
plt.xlabel('Predicted')
plt.ylabel('Actual')

FIGURE_DIR = os.path.join(BASE_DIR, "../../../reports/figures/outdoor_performance/rf")
os.makedirs(FIGURE_DIR, exist_ok=True)
CM_SAVE_PATH = os.path.join(FIGURE_DIR, "rf_class_grid_cm.png")

plt.savefig(CM_SAVE_PATH, dpi=300, bbox_inches='tight')
print(f"📈 CM 이미지 저장 완료: {CM_SAVE_PATH}")
plt.show()

# ==========================================
# 6. 변수 중요도 시각화 및 저장 (reports/figures/)
# ==========================================
importances = best_rf_model.feature_importances_
indices = np.argsort(importances)[::-1][:10]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=X_train.columns[indices], palette='viridis')
plt.title('Top 10 Feature Importances (CW + Grid RF)')
plt.xlabel('Relative Importance')
plt.ylabel('Features')

FI_SAVE_PATH = os.path.join(FIGURE_DIR, "rf_class_grid_fi.png")
plt.savefig(FI_SAVE_PATH, dpi=300, bbox_inches='tight')
print(f"📈 FI 이미지 저장 완료: {FI_SAVE_PATH}")
plt.show()