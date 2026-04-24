import os
import sys
import joblib
import numpy as np
import matplotlib.pyplot as plt

# 1. 절대 경로 설정 (Root 설정)
base_path = r"C:\work\mlproject\Surface-Perception-System\SSP\src"

if base_path not in sys.path:
    sys.path.append(base_path)

# 데이터 로더 임포트
from data.routes_indoor import load_processed_data

def visualize_importance():
    # 2. 데이터 로드 (피처 이름을 가져오기 위함)
    # 데이터 경로를 직접 전달 (필요 시)
    data_file_path = os.path.abspath(os.path.join(base_path, "../data/processed/indoor/indoor_train_features.csv"))
    X_train, _, _, _ = load_processed_data(file_path=data_file_path)
    
    # 3. 모델 폴더 경로 설정
    model_dir = os.path.abspath(os.path.join(base_path, "../models/indoor/"))
    
    if not os.path.exists(model_dir):
        print(f"❌ 모델 폴더를 찾을 수 없습니다: {model_dir}")
        return

    # 4. 트리 기반 모델(.pkl) 필터링 및 로드
    # 파일명에 'Forest'나 'XGB'가 포함된 모델만 가져옵니다.
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl') and ('Forest' in f or 'XGB' in f)]
    
    if not model_files:
        print("❌ 시각화할 트리 기반 모델(RF, XGB)이 폴더에 없습니다.")
        return

    models = {f: joblib.load(os.path.join(model_dir, f)) for f in model_files}
    
    # 5. 시각화 (모델 개수에 따라 서브플롯 생성)
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 8))
    
    # 모델이 하나일 경우 axes가 배열이 아니므로 리스트로 변환
    if n_models == 1:
        axes = [axes]

    for i, (fname, model) in enumerate(models.items()):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            # 중요도 상위 15개 피처 추출
            indices = np.argsort(importances)[-15:]
            
            axes[i].barh(range(len(indices)), importances[indices], color='#1ABC9C', align='center')
            axes[i].set_yticks(range(len(indices)))
            axes[i].set_yticklabels([X_train.columns[idx] for idx in indices])
            axes[i].set_title(f'Feature Importance: {fname}', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Relative Importance')
        else:
            axes[i].text(0.5, 0.5, f"{fname}\n(No Importance attribute)", ha='center')

    plt.suptitle("Sensor Feature Importance Analysis", fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 이미지 저장
    output_img = os.path.join(base_path, "feature_importance.png")
    plt.savefig(output_img, bbox_inches='tight')
    print(f"📊 피처 중요도 시각화 완료: {output_img}")
    plt.show()

if __name__ == "__main__": 
    visualize_importance()