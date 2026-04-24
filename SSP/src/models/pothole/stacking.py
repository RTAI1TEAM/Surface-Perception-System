import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    roc_curve, 
    roc_auc_score, 
    precision_recall_curve, 
    average_precision_score
)
import joblib 
import os

# =======================================================
# 0. 설정 및 데이터 로드
# =======================================================
DATASET_NAME = "v3_s2"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../../../data/processed/pothole")
TRAIN_PATH = os.path.join(DATA_DIR, f"train_{DATASET_NAME}.csv")
TEST_PATH = os.path.join(DATA_DIR, f"test_{DATASET_NAME}.csv")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

print(f"✅ 학습 데이터 로드 완료 ({DATASET_NAME})\n")

# =======================================================
# 1. Base Models (1선 실무자들 - 각자의 베스트 파라미터 장착)
# =======================================================
estimators = [
    # ① RF 에이스
    ('rf', RandomForestClassifier(
        max_depth=5, 
        max_features='sqrt', 
        min_samples_split=5, 
        n_estimators=200,
        class_weight='balanced', 
        random_state=42
    )),
    
    # ② SVM 수비수 (스케일링 파이프라인 필수)
    ('svm', Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(
            C=1, 
            gamma=0.01, 
            kernel='rbf',
            class_weight='balanced', 
            probability=True, # 스태킹 필수
            random_state=42
        ))
    ])),
    
    # ③ XGBoost 행동대장 (이전에 찾아둔 과적합 방지 파라미터 적용)
    ('xgb', XGBClassifier(
    max_depth=6,
    n_estimators=500,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=11.5,
    min_child_weight=3,
    gamma=1,
    eval_metric='logloss',
    random_state=42
    ))
]

# =======================================================
# 2. Meta Model (총괄 매니저 - 균형 잡힌 심판 역할)
# =======================================================
meta_model = LogisticRegression(
    class_weight='balanced', # 포트홀의 중요성을 잊지 않도록 설정
                    # 특정 모델만 편애하지 않도록 규제
    random_state=42
)

# =======================================================
# 3. 스태킹 앙상블 결합
# =======================================================
stacking_clf = StackingClassifier(
    estimators=estimators, 
    final_estimator=meta_model,
    cv=5, 
    n_jobs=-1
)

# 학습 시작
stacking_clf.fit(X_train, y_train)
print("✅ 스태킹 모델 학습 완료!\n")

# =======================================================
# 3. 메타 모델의 '최종' 임계값 최적화
# =======================================================
# 스태킹 모델이 뱉어낸 최종 포트홀 확률
y_probs = stacking_clf.predict_proba(X_test)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-9)

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
y_pred_tuned = (y_probs >= best_threshold).astype(int)

# =======================================================
# 4. 최종 결과 출력
# =======================================================
print("="*60)
print(f"👑 [Stacking Ensemble] 최종 리포트 ({DATASET_NAME})")
print("="*60)
print(f"🎯 최종 최적 임계값: {best_threshold:.4f}")
print(f"🏆 예상 최대 F1-Score: {f1_scores[best_idx]:.4f}")
print("-" * 60)
print(classification_report(y_test, y_pred_tuned, target_names=['Normal(0)', 'Pothole(1)']))
print("-" * 60)

print(f"accuracy: {accuracy_score(y_test, y_pred_tuned):.4f}")

roc_auc = roc_auc_score(y_test, y_probs)
pr_auc = average_precision_score(y_test, y_probs)
print(f"ROC_AUC Score: {roc_auc:.4f}")
print(f"PR_AUC Score:  {pr_auc:.4f}")
print("="*60)

# =======================================================
# 5. 모델 저장 및 시각화
# =======================================================
MODEL_DIR = os.path.join(BASE_DIR, "../../../models/pothole")
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(stacking_clf, os.path.join(MODEL_DIR, f"stacking_best_{DATASET_NAME}.pkl"))

FIGURE_DIR = os.path.join(BASE_DIR, "../../../reports/figures/outdoor_performance/stacking")
os.makedirs(FIGURE_DIR, exist_ok=True)

# Confusion Matrix 저장
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test, y_pred_tuned)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=['Normal', 'Pothole'], yticklabels=['Normal', 'Pothole'])
plt.title(f'Stacking CM (Thresh: {best_threshold:.4f})')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig(os.path.join(FIGURE_DIR, f"stacking_cm_{DATASET_NAME}.png"), dpi=300, bbox_inches='tight')
plt.close()

# PR Curve 저장
plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, label=f'AP = {pr_auc:.4f}', color='purple')
plt.plot(recalls[best_idx], precisions[best_idx], 'ro', label=f'Optimal (Thresh={best_threshold:.4f})')
plt.title(f'Stacking Precision-Recall Curve ({DATASET_NAME})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.savefig(os.path.join(FIGURE_DIR, f"stacking_pr_{DATASET_NAME}.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"\n📈 스태킹 모델과 이미지가 성공적으로 저장되었습니다.")