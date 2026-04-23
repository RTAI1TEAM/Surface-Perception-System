import os, sys, joblib, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import cross_val_score


base_path = r"C:\work\mlproject\Surface-Perception-System\SSP\src"

if base_path not in sys.path:
    sys.path.append(base_path)

from data.routes_indoor import load_processed_data

def visualize_cv():
    X_train, X_test, y_train, y_test = load_processed_data()
    model_dir = "../../models/indoor/"
    
    cv_means, cv_stds, names = [], [], []
    for f in os.listdir(model_dir):
        if f.endswith('.pkl'):
            name = f.replace('.pkl', '')
            model = joblib.load(os.path.join(model_dir, f))
            scores = cross_val_score(model, X_train, y_train, cv=5)
            cv_means.append(scores.mean())
            cv_stds.append(scores.std())
            names.append(name)

    plt.figure(figsize=(12, 6))
    plt.errorbar(names, cv_means, yerr=cv_stds, fmt='o', linestyle='-', color='r', ecolor='lightgray', capsize=5)
    plt.title('5-Fold Cross-Validation Accuracy (Mean & Std Dev)')
    plt.ylabel('Accuracy')
    plt.grid(axis='y', alpha=0.3)
    plt.show()

if __name__ == "__main__": visualize_cv()