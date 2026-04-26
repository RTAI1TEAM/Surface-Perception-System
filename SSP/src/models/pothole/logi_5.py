import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE # SMOTE 임포트

# 1. 경로 및 데이터 로드
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(current_dir, "../../../data/processed/pothole"))

train_df = pd.read_csv(os.path.join(data_dir, 'train_v3_s2.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'test_v3_s2.csv'))

X_train_raw = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test_raw = test_df.drop('label', axis=1)
y_test = test_df['label']

# 2. 스케일링 (SMOTE 전 수행 권장)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# --- 3. [LR5 실험의 핵심 1: SMOTE 적용] ---
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print(f"✅ SMOTE 완료: {len(y_train)} -> {len(y_train_resampled)} samples")

# --- 4. [LR5 실험의 핵심 2: GridSearchCV 설정] ---
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['lbfgs']
}

# 실험 조건: SMOTE를 썼으므로 Class Weight는 사용하지 않음 (None)
base_lr = LogisticRegression(random_state=42, max_iter=2000)

grid_search = GridSearchCV(
    base_lr, 
    param_grid, 
    cv=5, 
    scoring='f1', 
    n_jobs=-1
)

print("🚀 SMOTE 데이터로 최적 파라미터 찾는 중...")
grid_search.fit(X_train_resampled, y_train_resampled)

# 5. 최적 모델 추출
best_model = grid_search.best_estimator_
print(f"✅ 최적 파라미터: {grid_search.best_params_}")

# 6. 예측 및 지표 계산
y_probs = best_model.predict_proba(X_test_scaled)[:, 1]
threshold = 0.55
y_pred = (y_probs >= threshold).astype(int)

# 7. 결과 출력
print(f"\n{'='*20} [LR5 SMOTE + GridSearch 결과] {'='*20}")
print(f"조건: Scaling(O), ClassWeight(X), SMOTE(O), CV(O), Threshold(0.5)")
print("-" * 60)
print(classification_report(y_test, y_pred, target_names=['Normal(0)', 'Pothole(1)']))
print("-" * 60)
print(f"ROC-AUC:               {roc_auc_score(y_test, y_probs):.4f}")
print(f"PR-AUC (Avg Precision): {average_precision_score(y_test, y_probs):.4f}")
print(f"F1-Score (Pothole):    {f1_score(y_test, y_pred):.4f}")