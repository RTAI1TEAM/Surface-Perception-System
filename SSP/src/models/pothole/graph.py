import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import (
    roc_curve, roc_auc_score, 
    precision_recall_curve, average_precision_score,
    f1_score, recall_score, confusion_matrix
)
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

print(f"✅ 학습 데이터 로드 완료 ({DATASET_NAME})")

# =======================================================
# 1. 확정된 4개 베스트 모델 정의 (파라미터 적용)
# =======================================================
models = {
    'XGBoost': xgb.XGBClassifier(
        max_depth=6, n_estimators=500, learning_rate=0.05, 
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=11.5, 
        min_child_weight=3, gamma=1, eval_metric='logloss', random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        max_depth=5, max_features='sqrt', min_samples_split=5, 
        n_estimators=200, class_weight='balanced', random_state=42
    ),
    # SVM과 Logistic은 스케일링 필수이므로 Pipeline 사용
    'SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(
            C=1, gamma=0.01, kernel='rbf', 
            class_weight='balanced', probability=True, random_state=42
        ))
    ]),
    'Logistic': Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(
            C=0.01, max_iter=1000, solver='lbfgs', 
            class_weight='balanced', random_state=42
        ))
    ])
}

# 모델별 확정된 최적 임계값 딕셔너리
optimal_thresholds = {
    'XGBoost': 0.5,
    'Random Forest': 0.4528,
    'SVM': 0.2083,
    'Logistic': 0.5152
}

# 시각화 색상 테마
colors = {'XGBoost': '#2ca02c', 'Random Forest': '#ff7f0e', 'SVM': '#1f77b4', 'Logistic': '#7f7f7f'}

# =======================================================
# 2. 모델 학습 및 평가 지표 계산
# =======================================================
results = {}

print("🚀 4개 베스트 모델 학습 및 평가 진행 중...")
for name, model in models.items():
    # 학습
    model.fit(X_train, y_train)
    
    # 확률 예측
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # 확정된 임계값 적용
    thresh = optimal_thresholds[name]
    y_pred = (y_probs >= thresh).astype(int)
    
    # AUC 곡선 데이터
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = roc_auc_score(y_test, y_probs)
    
    precisions, recalls, _ = precision_recall_curve(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)
    
    # 임계값 기준 핵심 지표
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    results[name] = {
        'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc,
        'precisions': precisions, 'recalls': recalls, 'pr_auc': pr_auc,
        'thresh': thresh, 'f1': f1, 'recall': recall, 'cm': cm
    }
    print(f"✔️ {name} 완료 (Threshold: {thresh})")

# =======================================================
# 3. 보고서용 시각화 및 저장
# =======================================================
# 저장 폴더 생성
FIGURE_DIR = os.path.join(BASE_DIR, "../../../reports/figures/final_comparison")
os.makedirs(FIGURE_DIR, exist_ok=True)

# 폰트 및 스타일 기본 설정
plt.style.use('default')

# -------------------------------------------------------
# Graph 1: 다중 PR Curve (Precision-Recall)
# -------------------------------------------------------
plt.figure(figsize=(9, 6))
for name in models.keys():
    res = results[name]
    plt.plot(res['recalls'], res['precisions'], 
             label=f"{name} (PR-AUC: {res['pr_auc']:.4f})", color=colors[name], lw=2)
    # 최적 임계값 지점 점찍기
    best_p_idx = np.abs(res['recalls'] - res['recall']).argmin()
    plt.plot(res['recall'], res['precisions'][best_p_idx], marker='o', markersize=8, color=colors[name], 
             markeredgecolor='white', markeredgewidth=1.5)

plt.title('Precision-Recall Curve Comparison', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Recall (Pothole Detection Rate)', fontsize=12)
plt.ylabel('Precision (True Pothole Ratio)', fontsize=12)
plt.legend(loc='lower left', frameon=True, shadow=True)
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig(os.path.join(FIGURE_DIR, "01_final_PR_Curve.png"), dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------
# Graph 2: 다중 ROC Curve
# -------------------------------------------------------
plt.figure(figsize=(9, 6))
for name in models.keys():
    res = results[name]
    plt.plot(res['fpr'], res['tpr'], 
             label=f"{name} (ROC-AUC: {res['roc_auc']:.4f})", color=colors[name], lw=2)
plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7)

plt.title('ROC Curve Comparison', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.legend(loc='lower right', frameon=True, shadow=True)
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig(os.path.join(FIGURE_DIR, "02_final_ROC_Curve.png"), dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------
# Graph 3: 핵심 지표 비교 바 차트 (Grouped Bar Chart)
# -------------------------------------------------------
metrics_df = pd.DataFrame({
    'Model': list(models.keys()),
    'PR-AUC': [results[m]['pr_auc'] for m in models.keys()],
    'Recall (at Thresh)': [results[m]['recall'] for m in models.keys()],
    'F1-Score (at Thresh)': [results[m]['f1'] for m in models.keys()]
}).set_index('Model')

ax = metrics_df.plot(kind='bar', figsize=(10, 6), colormap='Set2', width=0.7, edgecolor='white')
plt.title('Key Performance Metrics by Model', fontsize=14, fontweight='bold', pad=15)
plt.ylabel('Score', fontsize=12)
plt.xlabel('')
plt.xticks(rotation=0, fontsize=11)
plt.legend(loc='lower right', frameon=True)
plt.ylim(0, 1.15)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# 막대 위에 수치 표시
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f"{p.get_height():.3f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=9, xytext=(0, 4), 
                    textcoords='offset points')

plt.savefig(os.path.join(FIGURE_DIR, "03_final_Metrics_Bar.png"), dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------
# Graph 4: 2x2 혼동 행렬 그리드 (Confusion Matrix)
# -------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for i, name in enumerate(models.keys()):
    cm = results[name]['cm']
    thresh = results[name]['thresh']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['Normal(0)', 'Pothole(1)'], 
                yticklabels=['Normal(0)', 'Pothole(1)'],
                annot_kws={"size": 14, "weight": "bold"})
    
    axes[i].set_title(f"{name}\n(Threshold: {thresh})", fontweight='bold', fontsize=12)
    axes[i].set_ylabel('Actual Label', fontsize=10)
    axes[i].set_xlabel('Predicted Label', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, "04_final_Confusion_Matrices.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"\n🎉 보고서용 그래프 4종 생성이 완료되었습니다.")
print(f"👉 저장 경로: {os.path.abspath(FIGURE_DIR)}")