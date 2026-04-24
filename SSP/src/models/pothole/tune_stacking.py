import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, accuracy_score, precision_recall_curve, confusion_matrix

# =======================================================
# 1. 경로 설정 및 데이터/모델 로드
# =======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../../../data/processed/pothole")
MODEL_DIR = os.path.join(BASE_DIR, "../../../models/pothole")
FIGURE_DIR = os.path.join(BASE_DIR, "../../../reports/figures/outdoor_performance/stacking")

TEST_PATH = os.path.join(DATA_DIR, "test_v3_s2.csv")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "stacking_best.pkl")

# 테스트 데이터 로드
test_df = pd.read_csv(TEST_PATH)
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# 스태킹 모델 로드
print("저장된 스태킹 모델을 불러옵니다...")
stacking_clf = joblib.load(MODEL_SAVE_PATH)

# =======================================================
# 2. 확률 예측 및 임계값별 성능 시뮬레이션
# =======================================================
# 모델이 평가한 '포트홀일 확률'
y_probs = stacking_clf.predict_proba(X_test)[:, 1]

# Precision-Recall Curve 데이터 추출
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)

# 시각화를 위한 데이터프레임 생성 (Threshold는 길이가 1 짧으므로 맞춤)
df_pr = pd.DataFrame({
    'Threshold': thresholds,
    'Precision': precisions[:-1],
    'Recall': recalls[:-1]
})
# F1-Score 계산 (분모가 0이 되는 것 방지)
df_pr['F1_Score'] = 2 * (df_pr['Precision'] * df_pr['Recall']) / (df_pr['Precision'] + df_pr['Recall'] + 1e-9)

# =======================================================
# 3. 최적의 Threshold 시각화
# =======================================================
plt.figure(figsize=(10, 6))
plt.plot(df_pr['Threshold'], df_pr['Precision'], label='Precision', color='blue', linewidth=2)
plt.plot(df_pr['Threshold'], df_pr['Recall'], label='Recall', color='orange', linewidth=2)
plt.plot(df_pr['Threshold'], df_pr['F1_Score'], label='F1 Score', color='green', linestyle='--', linewidth=2)

# 재현율 0.85 목표선 (가상의 기준선, 필요시 변경)
plt.axhline(y=0.85, color='red', linestyle=':', label='Target Recall (0.85)')

plt.title('Performance Metrics vs. Threshold (Stacking Model)')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.legend(loc='lower left')
plt.grid(True, alpha=0.3)

TUNE_SAVE_PATH = os.path.join(FIGURE_DIR, "stacking_threshold_tuning.png")
plt.savefig(TUNE_SAVE_PATH, dpi=300, bbox_inches='tight')
print(f"📈 튜닝 그래프 저장 완료: {TUNE_SAVE_PATH}")
plt.show()

# =======================================================
# 4. 추천 Threshold 분석 (목표 Recall = 0.85 이상)
# =======================================================
target_recall = 0.9

# Recall이 목표치 이상인 구간 중 F1 Score가 가장 높은 지점 찾기
valid_thresholds = df_pr[df_pr['Recall'] >= target_recall]

if not valid_thresholds.empty:
    best_row = valid_thresholds.loc[valid_thresholds['F1_Score'].idxmax()]
    best_threshold = best_row['Threshold']
    
    print("\n" + "="*50)
    print(f"🎯 목표 재현율({target_recall}) 달성을 위한 추천 임계값: {best_threshold:.3f}")
    print("="*50)
    
    # 추천 임계값으로 최종 평가
    y_pred_tuned = (y_probs >= best_threshold).astype(int)
    print(classification_report(y_test, y_pred_tuned, target_names=['Normal(0)', 'Pothole(1)']))
    
    # 추천 임계값의 혼동 행렬 시각화
    plt.figure(figsize=(6, 4))
    cm = confusion_matrix(y_test, y_pred_tuned)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=['Normal', 'Pothole'], yticklabels=['Normal', 'Pothole'])
    plt.title(f'Tuned Stacking Confusion Matrix (Threshold: {best_threshold:.2f})')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    
    TUNED_CM_PATH = os.path.join(FIGURE_DIR, f"stacking_tuned_cm_{best_threshold:.2f}.png")
    plt.savefig(TUNED_CM_PATH, dpi=300, bbox_inches='tight')
    plt.show()

else:
    print(f"\n⚠️ 경고: 현재 스태킹 모델로는 임계값을 아무리 낮춰도 재현율 {target_recall}을 달성할 수 없습니다.")