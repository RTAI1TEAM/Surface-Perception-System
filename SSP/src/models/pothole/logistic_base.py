import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


# 1. Raw 데이터 로드
train_df = pd.read_csv('train_v2.csv')
test_df = pd.read_csv('test_v2.csv')

X_train_raw = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test_raw = test_df.drop('label', axis=1)
y_test = test_df['label']

# 2. 베이스 모델 학습 (중요도 파악용)
scaler_base = StandardScaler()
X_train_scaled = scaler_base.fit_transform(X_train_raw)
base_model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
base_model.fit(X_train_scaled, y_train)

# 3. Feature Importance 상위 10개 자동 선택
importances = np.abs(base_model.coef_[0])
importance_df = pd.DataFrame({'Feature': X_train_raw.columns, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

top_features = importance_df.head(10)['Feature'].tolist()
print(f"선택된 중요 컬럼 (10개): {top_features}")

# 4. 선택된 6개 컬럼으로 데이터 재구성 및 재학습
X_train_selected = X_train_raw[top_features]
X_test_selected = X_test_raw[top_features]

scaler_final = StandardScaler()
X_train_sel_scaled = scaler_final.fit_transform(X_train_selected)
X_test_sel_scaled = scaler_final.transform(X_test_selected)

final_model = LogisticRegression(class_weight='balanced', random_state=42)
final_model.fit(X_train_sel_scaled, y_train)

# 5. 임계값(Threshold) 조정 적용
# 단순 predict() 대신 확률(predict_proba)을 구합니다.
y_probs = final_model.predict_proba(X_test_sel_scaled)[:, 1]

# 정밀도를 높이기 위해 임계값을 0.6로 설정 
threshold = 0.6
y_pred_final = (y_probs >= threshold).astype(int)

# 6. 최종 지표 출력 (요청하신 리포트 형식 포함)
print(f"\n--- 최종 모델 (상위 10개 컬럼 + 임계값 {threshold}) 지표 ---")
print("-" * 60)
# 상세 리포트 출력
print(classification_report(y_test, y_pred_final, target_names=['Normal(0)', 'Pothole(1)']))
print("-" * 60)

print(f"전체 정확도 (Accuracy): {accuracy_score(y_test, y_pred_final):.4f}")
print(f"F1-Score:             {f1_score(y_test, y_pred_final):.4f}")

# 7. 혼돈 행렬 시각화
cm = confusion_matrix(y_test, y_pred_final)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal(0)', 'Pothole(1)'], 
            yticklabels=['Normal(0)', 'Pothole(1)'])
plt.title(f'Optimized Confusion Matrix (Threshold: {threshold})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


#저장
import os
import joblib

# 1. 현재 파일(logistic_base.py)의 절대 경로
current_file_path = os.path.abspath(__file__) 
# 2. 현재 파일이 위치한 폴더 (src/models/pothole)
current_dir = os.path.dirname(current_file_path)

# 3. 프로젝트 루트의 models/pothole 폴더로 이동 
target_dir = os.path.abspath(os.path.join(current_dir, "../../models/pothole"))

# 만약 폴더가 없으면 생성
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 4. 최종 저장 경로 설정
model_path = os.path.join(target_dir, "logistic_base.pkl")

# 5. 저장 실행
joblib.dump(base_model, model_path)
print(f"모델이 다음 위치로 이동 저장되었습니다:\n{model_path}")

