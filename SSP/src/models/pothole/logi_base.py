import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score, average_precision_score

# --- 1. 경로 설정 및 데이터 로드 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(current_dir, "../../../data/processed/pothole"))

train_path = os.path.join(data_dir, 'train_v3_s2.csv')
test_path = os.path.join(data_dir, 'test_v3_s2.csv')

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

X_train_raw = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test_raw = test_df.drop('label', axis=1)
y_test = test_df['label']

# --- 2. 베이스 모델 학습 (스케일링 없이 생 데이터 투입) ---
# 가중치(class_weight)도 없고 스케일링도 없는 가장 기본적인 상태입니다.
base_model = LogisticRegression(random_state=42, max_iter=2000) # 수렴을 위해 iter만 조금 높였습니다.
base_model.fit(X_train_raw, y_train)

# --- 3. Feature Importance 상위 10개 자동 선택 ---
importances = np.abs(base_model.coef_[0])
importance_df = pd.DataFrame({'Feature': X_train_raw.columns, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
top_features = importance_df.head(10)['Feature'].tolist()
print(f"✅ 선택된 중요 컬럼 (10개): {top_features}")

# --- 4. 선택된 컬럼으로 재구성 및 재학습 (Raw Data) ---
X_train_selected = X_train_raw[top_features]
X_test_selected = X_test_raw[top_features]

final_model = LogisticRegression(random_state=42, max_iter=2000)
final_model.fit(X_train_selected, y_train)

# --- 5. 예측 및 지표 계산 ---
y_probs = final_model.predict_proba(X_test_selected)[:, 1]
threshold = 0.5
y_pred_final = (y_probs >= threshold).astype(int)

roc_auc = roc_auc_score(y_test, y_probs)
pr_auc = average_precision_score(y_test, y_probs)

# --- 6. 결과 출력 ---
print(f"\n--- [LR1 Raw Baseline] (No Scaling, No Weight, Threshold {threshold}) ---")
print("-" * 60)
print(classification_report(y_test, y_pred_final, target_names=['Normal(0)', 'Pothole(1)']))
print("-" * 60)
print(f"전체 정확도 (Accuracy): {accuracy_score(y_test, y_pred_final):.4f}")
print(f"F1-Score (Pothole):   {f1_score(y_test, y_pred_final):.4f}")
print(f"ROC-AUC:              {roc_auc:.4f}")
print(f"PR-AUC (Avg Precision): {pr_auc:.4f}")
print("-" * 60)