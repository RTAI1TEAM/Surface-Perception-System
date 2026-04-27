import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. 한글 폰트 및 스타일 설정 (그림 7과 완벽 일치)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font='Malgun Gothic')

# 경로 설정
CSV_PATH = r'c:\Users\USER\dev\rotem\ml_project\SSP\reports\figures\indoor_performance\indoor_inference_benchmark.csv'
OUTPUT_PATH = r'c:\Users\USER\dev\rotem\ml_project\SSP\reports\figures\indoor_performance\final_best_selection_v2.png'

# 2. 데이터 로드
df = pd.read_csv(CSV_PATH)

# 모델명 간소화
model_map = {
    'XGBoost (Base)': 'XGBoost (Base)',
    'XGBoost (HyperTuned+Final)': 'XGBoost (Tuned)',
    'RandomForest (SMOTE+Final)': 'Random Forest',
    'DecisionTree (ClassWeight+Final)': 'Decision Tree'
}
df['Algorithm'] = df['Model'].map(model_map)

# 3. 데이터 프레임 변환 (Metric 이름 변경하여 그림 7과 통일)
results = []
for idx, row in df.iterrows():
    results.append({'Algorithm': row['Algorithm'], 'Metric': 'Accuracy', 'Score': row['Accuracy']})
    results.append({'Algorithm': row['Algorithm'], 'Metric': 'Recall', 'Score': row['Recall_Macro']})
    results.append({'Algorithm': row['Algorithm'], 'Metric': 'F1-Score', 'Score': row['F1_Macro']})

df_res = pd.DataFrame(results)

# 4. 시각화 (그림 7 스타일 완벽 재현)
plt.figure(figsize=(12, 7))

ax = sns.barplot(data=df_res, x='Algorithm', y='Score', hue='Metric', palette='viridis')

# 수치 표시 (fontweight='bold' 및 위치 조정 일치)
for p in ax.patches:
    h = p.get_height()
    if h > 0:
        ax.annotate(f'{h:.3f}', (p.get_x() + p.get_width() / 2., h), 
                    ha='center', va='bottom', xytext=(0, 9), 
                    textcoords='offset points', fontsize=10, fontweight='bold')

plt.title('Final Champion Selection: Indoor Models Comparison', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Performance Score')
plt.xlabel('Algorithm')
plt.ylim(0, 1.15)  # 상단 라벨 여백 일치
plt.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()

# 이미지 저장
plt.savefig(OUTPUT_PATH, dpi=300)
print(f"Success: {OUTPUT_PATH}")
