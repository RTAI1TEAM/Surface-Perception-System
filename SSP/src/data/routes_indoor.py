import pandas as pd
from sklearn.model_selection import train_test_split

def load_processed_data(file_path='../../../data/processed/indoor/indoor_train_features.csv'):
    df = pd.read_csv(file_path)
    target_col = 'surface_encoded'
    
    y = df[target_col]
    # series_id, group_id, surface 등 불필요 컬럼 제거 [cite: 108]
    X = df.drop(columns=['series_id', 'group_id', 'surface', target_col])
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)