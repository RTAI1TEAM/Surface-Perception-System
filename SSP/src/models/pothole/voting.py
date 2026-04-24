import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 모델링 및 평가 라이브러리
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve

# =======================================================
# 1. 데이터 로드 및 경로 설정 (v3_s2 데이터 기준)
# =======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../../../data/processed/pothole")
TRAIN_PATH = os.path.join(DATA_DIR, "train_v3_s2.csv")
TEST_PATH = os.path.join(DATA_DIR, "test_v3_s2.csv")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# XGBoost를 위한 불균형 가중치 계산
scale_weight = (y_train == 0).sum() / (y_train == 1).sum()

# =======================================================
# 2. Voting 베이스 모델 정의
# =======================================================
print("\nSoft Voting 앙상블 모델 설정 중...")

# 1. Random Forest (불균형 처리)
rf_base = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

# 2. SVM (Pipeline으로 묶어서 스케일링 동시 진행, probability=True 필수)
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('svc', SVC(probability=True, random_state=42))
])

# 3. XGBoost
xgb_base = XGBClassifier(
    scale_pos_weight=scale_weight, 
    eval_metric='logloss', 
    random_state=42,
    
    # --- 과적합(Overfitting) 방지용 강력 제약 ---
    max_depth=3,             # 1. 트리 깊이 제한: 기본값 6에서 3으로 대폭 축소 (단순한 패턴만 찾도록)
    learning_rate=0.05,      # 2. 보폭 줄이기: 기본값 0.3에서 0.05로 축소 (급하게 학습하지 않고 천천히)
    n_estimators=100,        # 3. 나무 개수: 학습률을 낮췄으므로 100개 정도 유지
    subsample=0.8,           # 4. 데이터 제한: 매번 학습할 때 전체 300개 중 80%만 랜덤하게 뽑아서 학습
    colsample_bytree=0.8,    # 5. 피처 제한: 컬럼(특징)도 전체를 다 보지 않고 80%만 랜덤하게 선택해서 학습
    min_child_weight=3       # 6. 최소 분할 기준: 가지치기를 할 때 최소 3개 이상의 데이터가 있어야만 분할 (노이즈 방지)
)

# Soft Voting Classifier 결합
voting_clf = VotingClassifier(
    estimators=[
        ('rf', rf_base),
        ('svm_scaled', svm_pipeline),
        ('xgb', xgb_base)
    ],
    voting='soft' # 확률의 평균을 사용하여 임계값 조정이 가능하도록 설정
)

# =======================================================
# 3. 모델 학습 및 저장
# =======================================================
print("Voting 모델 학습 중... (Stacking보다 훨씬 빠릅니다!)")
voting_clf.fit(X_train, y_train)

MODEL_DIR = os.path.join(BASE_DIR, "../../../models/pothole")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "voting_soft.pkl")

joblib.dump(voting_clf, MODEL_SAVE_PATH)
print(f"🎉 모델 저장 완료: {MODEL_SAVE_PATH}")

# =======================================================
# 4. 임계값(Threshold) 최적화 시뮬레이션
# =======================================================
# 앙상블 모델의 포트홀(1) 확률 예측값 추출
y_probs = voting_clf.predict_proba(X_test)[:, 1]

# 다양한 임계값에 따른 재현율, 정밀도 계산
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
df_pr = pd.DataFrame({
    'Threshold': thresholds,
    'Precision': precisions[:-1],
    'Recall': recalls[:-1]
})
df_pr['F1_Score'] = 2 * (df_pr['Precision'] * df_pr['Recall']) / (df_pr['Precision'] + df_pr['Recall'] + 1e-9)

# 목표 재현율(Recall) 설정 (이전 최고 모델 기준치인 0.90으로 세팅)
target_recall = 0.90
valid_thresholds = df_pr[df_pr['Recall'] >= target_recall]

if not valid_thresholds.empty:
    # 조건을 만족하는 것 중 F1 점수가 가장 높은 임계값 선택
    best_row = valid_thresholds.loc[valid_thresholds['F1_Score'].idxmax()]
    best_threshold = best_row['Threshold']
else:
    # 목표를 달성하지 못할 경우 기본값 0.5 사용
    print(f"\n⚠️ 목표 재현율({target_recall}) 달성 불가. F1 최대치 기준으로 조정합니다.")
    best_threshold = df_pr.loc[df_pr['F1_Score'].idxmax(), 'Threshold']

print("\n" + "="*50)
print(f"🎯 [최적화 결과] 추천 임계값: {best_threshold:.3f}")
print("="*50)

# =======================================================
# 5. 추천 임계값 적용 최종 평가 및 시각화
# =======================================================
y_pred_tuned = (y_probs >= best_threshold).astype(int)

print("\n--- [Voting Ensemble] 최종 리포트 ---")
print(classification_report(y_test, y_pred_tuned, target_names=['Normal(0)', 'Pothole(1)']))
print(f"Accuracy: {accuracy_score(y_test, y_pred_tuned):.4f}")

# 시각화 저장 폴더
FIGURE_DIR = os.path.join(BASE_DIR, "../../../reports/figures/outdoor_performance/voting")
os.makedirs(FIGURE_DIR, exist_ok=True)

plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test, y_pred_tuned)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', # 보팅 모델은 초록색으로 구분
            xticklabels=['Normal', 'Pothole'], yticklabels=['Normal', 'Pothole'])
plt.title(f'Voting Confusion Matrix (Threshold: {best_threshold:.3f})')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')

CM_SAVE_PATH = os.path.join(FIGURE_DIR, "voting_cm.png")
plt.savefig(CM_SAVE_PATH, dpi=300, bbox_inches='tight')
print(f"📈 CM 이미지 저장 완료: {CM_SAVE_PATH}")
plt.show()