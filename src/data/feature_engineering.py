import pandas as pd
import numpy as np
from boruta import BorutaPy
from utils.project_functions import load_data
from sklearn.ensemble import RandomForestClassifier

np.int = int
np.float = float
np.bool = bool

def create_binary_classification(data):
    """
    Create binary classification for price movement.
    """
    price = pd.DataFrame()
    price['today'] = data['price-ohlc-usd-close']
    price['next day'] = price['today'].shift(-1)
    target = (price['next day'] > price['today']).astype(int)
    return target

def select_features(data, target):
    """
    Select relevant features using Boruta feature selection method.
    """
    X = data.drop('timestamp', axis=1)
    X_np = X.values.copy()
    y_np = target.values.copy()

    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=1)
    feat_selector.fit(X_np, y_np)

    features_bool = feat_selector.support_
    feature_ranking = feat_selector.ranking_
    features = np.array(X.columns)
    features_selected = features[features_bool]
    features_selected_tentative = features[feature_ranking <= feat_selector.max_iter]
    
    return features_selected, features_selected_tentative

def save_selected_features(df, features_selected, output_path):
    """
    Save selected features.
    """
    try:
        features_selected = np.insert(features_selected, 0, 'timestamp')
        df[features_selected].to_csv(output_path, index=False)
        print(f"Selected features saved to {output_path}")
    except Exception as e:
        print(f"Error saving selected features: {e}")