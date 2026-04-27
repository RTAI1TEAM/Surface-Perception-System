import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def select_and_save_final_best():
    # 한글 폰트 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 제시해주신 데이터 (Final 값만 사용)
    data = {
        'Logistic Regression': {'Final': [0.619, 0.860, 0.650]},
        'SVM': {'Final': [0.517, 0.900, 0.670]},
        'Random Forest': {'Final': [0.678, 0.930, 0.700]},
        'XGBoost': {'Final': [0.639, 0.741, 0.667]}
    }
    
    # 2. 데이터 프레임 변환 (List of Dicts)
    results = []
    for algo, stage_data in data.items():
        scores = stage_data['Final']
        # [PR-AUC, Recall, F1-Score] 순서로 저장되어 있음
        results.append({'Algorithm': algo, 'Metric': 'PR-AUC', 'Score': scores[0]})
        results.append({'Algorithm': algo, 'Metric': 'Recall', 'Score': scores[1]})
        results.append({'Algorithm': algo, 'Metric': 'F1-Score', 'Score': scores[2]})

    df_res = pd.DataFrame(results)

    # 3. 지정된 저장 경로 설정
    REPORT_DIR = '../../../reports/figures/outdoor_performance'
    os.makedirs(REPORT_DIR, exist_ok=True)

    # 4. 시각화 (요청하신 스타일 완벽 적용)
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid", font='Malgun Gothic')
    
    ax = sns.barplot(data=df_res, x='Algorithm', y='Score', hue='Metric', palette='viridis')
    
    # 수치 표시
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            # 텍스트가 막대 선과 겹치지 않게 va='bottom'으로 미세조정 하였습니다.
            ax.annotate(f'{h:.3f}', (p.get_x() + p.get_width() / 2., h), 
                        ha='center', va='bottom', xytext=(0, 9), 
                        textcoords='offset points', fontsize=10, fontweight='bold')

    plt.title('Final Champion Selection: Best Models Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Performance Score')
    plt.xlabel('Algorithm')
    plt.ylim(0, 1.15)  # 상단 라벨이 잘리지 않도록 여백 추가
    plt.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # 이미지 저장
    save_path = os.path.join(REPORT_DIR, "final_best_selection.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ 최종 통합 리포트 저장 완료: {save_path}")
    
    plt.close() # 메모리 관리를 위해 창 닫기

if __name__ == "__main__":
    select_and_save_final_best()