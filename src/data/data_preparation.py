import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

def create_sequences(data, timesteps):
    """
    Function creates sequences for time series data
    """
    X = []
    for i in range(len(data) - timesteps + 1):
        X.append(data[i:i + timesteps])
    return np.array(X)

def prepare_data(df, timesteps):
    """
    Function prepares data
    """
    X = df.drop('timestamp', axis=1)
    price = pd.DataFrame()
    price['today'] = df['price-ohlc-usd-close']
    price['next day'] = price['today'].shift(-1)
    y = (price['next day'] > price['today']).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    time_train, time_test = train_test_split(df['timestamp'], test_size=0.2, shuffle=False)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_reshaped = create_sequences(X_train_scaled, timesteps)
    X_test_reshaped = create_sequences(X_test_scaled, timesteps)
    y_train = y_train[timesteps - 1:]
    y_test = y_test[timesteps - 1:]
    time_test = time_test[timesteps:]
    
    return X_train_reshaped, X_test_reshaped, y_train, y_test, time_test, price, scaler

def save_preprocessed_data(filename, X_train, X_test, y_train, y_test, time_test, price, scaler):
    """
    Function saves preprocessed data
    """
    with open(filename, 'wb') as f:
        pickle.dump((X_train, X_test, y_train, y_test, time_test, price, scaler), f)

def load_preprocessed_data(filename):
    """
    Function loads preprocessed data
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)
