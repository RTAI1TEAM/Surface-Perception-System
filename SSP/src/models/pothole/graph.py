import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 1. 제시해주신 하드코딩 데이터 설정
# [Accuracy, Precision, Recall, F1-score] 순서
data = {
    'Logistic Regression': [0.8447, 0.67, 0.34, 0.4545],
    'SVM': [0.87, 0.68, 0.55, 0.61],
    'Random Forest': [0.88, 0.69, 0.64, 0.66],
    'XGBoost': [0.8641, 0.60, 0.83, 0.70]
}
metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

# 2. 지정된 저장 경로 설정
save_dir = '../../../reports/figures/outdoor_performance'
os.makedirs(save_dir, exist_ok=True)

# 3. 시각화 (모델이 4개이므로 2x2 서브플롯 구성)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model-Specific Performance (Base Models)', fontsize=22, fontweight='bold')
axes = axes.flatten()

print("🚀 4종 베이스 모델 성능 시각화 생성 중...")

for i, (model_name, scores) in enumerate(data.items()):
    # 데이터프레임 생성
    m_df = pd.DataFrame({
        'Metric': metrics_list,
        'Score': scores
    })
    
    # 2x2 서브플롯에 각각 바 차트 그리기 (기존 스타일 적용)
    sns.barplot(x='Metric', y='Score', data=m_df, ax=axes[i], palette='coolwarm')
    
    # 스타일 세팅
    axes[i].set_title(f"Model: {model_name}", fontsize=14, fontweight='bold')
    axes[i].set_ylim(0, 1.1)
    axes[i].grid(axis='y', linestyle='--', alpha=0.5)
    axes[i].set_ylabel("Score", fontsize=11)
    axes[i].set_xlabel("", fontsize=11) # X축 라벨은 지표명으로 충분하므로 생략
    
    # 막대 위에 수치 표시 (겹치지 않게 va='bottom'으로 미세조정)
    for p in axes[i].patches:
        height = p.get_height()
        if height > 0:
            axes[i].annotate(f"{height:.3f}", (p.get_x() + p.get_width() / 2., height),
                             ha='center', va='bottom', xytext=(0, 5), textcoords='offset points', fontsize=10)

# 전체 레이아웃 자동 조정
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# 4. 이미지 저장
save_path = os.path.join(save_dir, "base_model_metrics.png")
plt.savefig(save_path, dpi=300)
print(f"\n📊 시각화 파일 저장 완료: {save_path}")

plt.close() # 메모리 관리를 위해 창 닫기