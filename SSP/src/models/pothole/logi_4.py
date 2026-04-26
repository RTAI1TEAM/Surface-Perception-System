import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV # 추가
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score, average_precision_score

# 1. 경로 및 데이터 로드
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(current_dir, "../../../data/processed/pothole"))

train_df = pd.read_csv(os.path.join(data_dir, 'train_v3_s2.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'test_v3_s2.csv'))

X_train_raw = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test_raw = test_df.drop('label', axis=1)
y_test = test_df['label']

# 2. 스케일링 (GridSearch 전 필수)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# --- 3. [LR4 실험의 핵심: GridSearchCV 설정] ---
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100], # 규제 강도 (작을수록 강한 규제)
    'penalty': ['l2'],                  # 로지스틱 기본 페널티
    'solver': ['lbfgs']                 # 최적화 알고리즘
}

# 기본 모델 설정 (실험 표 조건에 따라 Class Weight는 'balanced' 권장)
base_lr = LogisticRegression(random_state=42, max_iter=2000)

# GridSearchCV 실행 (5-Fold 교차 검증, 평가 지표는 f1-score 기준)
grid_search = GridSearchCV(
    base_lr, 
    param_grid, 
    cv=5, 
    scoring='f1', 
    n_jobs=-1
)

print("🚀 최적의 파라미터를 찾는 중... (Grid Search 진행 중)")
grid_search.fit(X_train_scaled, y_train)

# 4. 최적 모델 추출
best_model = grid_search.best_estimator_
print(f"✅ 최적 파라미터: {grid_search.best_params_}")

# 5. 예측 및 지표 계산
y_probs = best_model.predict_proba(X_test_scaled)[:, 1]
threshold = 0.5
y_pred = (y_probs >= threshold).astype(int)

# 6. 결과 출력
print(f"\n{'='*20} [LR4 GridSearchCV 결과] {'='*20}")
print(f"조건: Scaling(O), ClassWeight(O), SMOTE(X), CV(O), Threshold(0.5)")
print("-" * 60)
print(classification_report(y_test, y_pred, target_names=['Normal(0)', 'Pothole(1)']))
print("-" * 60)
print(f"ROC-AUC:               {roc_auc_score(y_test, y_probs):.4f}")
print(f"PR-AUC (Avg Precision): {average_precision_score(y_test, y_probs):.4f}")
print(f"F1-Score (Pothole):    {f1_score(y_test, y_pred):.4f}")