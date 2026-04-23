import os, sys, joblib, numpy as np, matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.routes_indoor import load_processed_data

def visualize_importance():
    X_train, _, _, _ = load_processed_data()
    model_dir = "../../models/indoor/"
    
    # 트리 기반 모델(RF, XGB)만 필터링
    models = {f: joblib.load(os.path.join(model_dir, f)) for f in os.listdir(model_dir) if 'Forest' in f or 'XGB' in f}
    
    fig, axes = plt.subplots(1, len(models), figsize=(16, 8))
    for i, (fname, model) in enumerate(models.items()):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-15:]
        axes[i].barh(range(len(indices)), importances[indices], color='#1ABC9C')
        axes[i].set_yticks(range(len(indices)))
        axes[i].set_yticklabels([X_train.columns[idx] for idx in indices])
        axes[i].set_title(f'Importance: {fname}')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__": 
    visualize_importance()