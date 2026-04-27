import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score, average_precision_score
# --- SMOTE 임포트 ---
from imblearn.over_sampling import SMOTE 

# 1. 경로 및 데이터 로드 (v3_s2)
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(current_dir, "../../../data/processed/pothole"))

train_df = pd.read_csv(os.path.join(data_dir, 'train_v3_s2.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'test_v3_s2.csv'))

X_train_raw = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test_raw = test_df.drop('label', axis=1)
y_test = test_df['label']

# 2. 스케일링 (SMOTE 전후 상관없지만, 보통 먼저 진행)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# --- 3. [LR3 실험의 핵심: SMOTE 적용] ---
# 훈련 데이터만 오버샘플링합니다. (테스트 데이터는 건드리면 안 됨!)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print(f"✅ SMOTE 적용 전 학습 데이터 수: {len(y_train)}")
print(f"✅ SMOTE 적용 후 학습 데이터 수: {len(y_train_resampled)}")

# 4. 모델 학습 (실험 조건: Class Weight X)
lr3_model = LogisticRegression(random_state=42, max_iter=2000)
lr3_model.fit(X_train_resampled, y_train_resampled)

# 5. 예측 및 지표 계산
y_probs = lr3_model.predict_proba(X_test_scaled)[:, 1]
threshold = 0.5
y_pred = (y_probs >= threshold).astype(int)

# 6. 결과 출력
print(f"\n{'='*20} [LR3 SMOTE 결과] {'='*20}")
print(f"조건: Scaling(O), ClassWeight(X), SMOTE(O), Threshold(0.5)")
print("-" * 60)
print(classification_report(y_test, y_pred, target_names=['Normal(0)', 'Pothole(1)']))
print("-" * 60)
print(f"ROC-AUC:               {roc_auc_score(y_test, y_probs):.4f}")
print(f"PR-AUC (Avg Precision): {average_precision_score(y_test, y_probs):.4f}")
print(f"F1-Score (Pothole):    {f1_score(y_test, y_pred):.4f}")