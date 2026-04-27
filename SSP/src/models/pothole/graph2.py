import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 1. 제시해주신 하드코딩 데이터
data = {
    'Logistic Regression': {
        'Base': [0.635, 0.340, 0.455],
        'Imbalance': [0.644, 0.810, 0.614],
        'Final': [0.619, 0.860, 0.650]
    },
    'SVM': {
        'Base': [0.524, 0.550, 0.610],
        'Imbalance': [0.496, 0.780, 0.660],
        'Final': [0.517, 0.900, 0.670]
    },
    'Random Forest': {
        'Base': [0.651, 0.810, 0.730],
        'Imbalance': [0.569, 0.860, 0.680],
        'Final': [0.678, 0.930, 0.700]
    },
    'XGBoost': {
        'Base': [0.610, 0.828, 0.696],
        'Imbalance': [0.601, 0.741, 0.672],
        'Final': [0.639, 0.741, 0.667]
    }
}

# 2. 지정된 저장 경로 설정
save_dir = '../../../reports/figures/outdoor_performance'
os.makedirs(save_dir, exist_ok=True)

# 3. 그래프 생성 및 저장 루프
stages = ["Base", "Imbalance", "Final"]
metrics = ["PR-AUC", "Recall", "F1-Score"]

print("🚀 알고리즘별 개별 리포트 생성 중...")

for algo, stage_data in data.items():
    # 데이터를 Melt하기 좋은 형태(List of Dicts)로 변환
    algo_results = []
    for stage in stages:
        scores = stage_data[stage]
        algo_results.append({
            "Stage": stage,
            "PR-AUC": scores[0],
            "Recall": scores[1],
            "F1-Score": scores[2]
        })

    # 데이터 재구성 (Melt) - 기존 스타일 유지
    df_algo = pd.DataFrame(algo_results).melt(id_vars="Stage", var_name="Metric", value_name="Score")

    # --- 시각화 시작 ---
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid", font='Malgun Gothic')
    
    # 지표별 막대 그래프 (viridis 팔레트 적용)
    ax = sns.barplot(data=df_algo, x="Stage", y="Score", hue="Metric", palette="viridis")
    
    # 막대 위에 수치 표시
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f'{h:.3f}', (p.get_x() + p.get_width() / 2., h), 
                        ha='center', va='bottom', xytext=(0, 5), 
                        textcoords='offset points', fontsize=11, fontweight='bold')

    plt.title(f"모델 성능 발전사: {algo}", fontsize=20, fontweight='bold', pad=20)
    plt.ylim(0, 1.15)
    plt.ylabel("Performance Score", fontsize=12)
    plt.xlabel("개선 단계", fontsize=12)
    plt.legend(title="평가 지표", bbox_to_anchor=(1, 1))
    
    # 파일 저장 (파일명에 공백이 있으면 _로 치환하여 안전하게 저장)
    save_name = f"report_performance_{algo.replace(' ', '_')}.png"
    save_path = os.path.join(save_dir, save_name)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ {algo} 리포트 저장 완료: {save_path}")
    
    plt.close() # 메모리 관리를 위해 창 닫기

print("🎉 모든 시각화 작업이 완료되었습니다!")