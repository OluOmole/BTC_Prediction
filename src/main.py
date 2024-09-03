import os
from utils.project_functions import load_data, reset_random_seeds
from data.data_cleaning import preprocess_data
from data.data_preparation import prepare_data, save_preprocessed_data, load_preprocessed_data
from data.feature_engineering import create_binary_classification, select_features, save_selected_features
from models.cnn_lstm_model import build_cnn_lstm
from models.lstnet_model import build_lstnet
from models.tcn_model import build_tcn
from models.train import train_model
from models.predict import predict

def main():
    # Data cleaning and preprocessing
    raw_data_path = 'data/raw/bitcoin_data.csv'
    cleaned_data_path = 'data/processed/cleaned_data.csv'
    
    print(f'Loading data from {raw_data_path}...')
    data = load_data(raw_data_path)
    if data is not None:
        print('Data loaded. Starting preprocessing...')
        cleaned_data = preprocess_data(data)
        if cleaned_data is not None:
            os.makedirs(os.path.dirname(cleaned_data_path), exist_ok=True)
            cleaned_data.to_csv(cleaned_data_path, index=False)
            print(f'Cleaned data saved to {cleaned_data_path}')
        else:
            print('Data cleaning failed.')
            return
    else:
        print('Data loading failed.')
        return

    # Feature engineering
    boruta_data_path = 'data/processed/Boruta_data.csv'
    
    df = load_data(cleaned_data_path)
    if df is not None:
        y = create_binary_classification(df)
        features_selected, features_selected_tentative = select_features(df, y)
        print('Features selected:', features_selected)
        save_selected_features(df, features_selected, boruta_data_path)
    else:
        print('Error loading preprocessed data.')
        return
    
    # Data preparation
    selected_features_path = 'data/processed/Boruta_data.csv'
    df = load_data(selected_features_path)

    timesteps = 5
    X_train, X_test, y_train, y_test, time_test, price, scaler = prepare_data(df, timesteps)
    save_preprocessed_data('data/processed/split_data.pkl', X_train, X_test, y_train, y_test, time_test, price, scaler)
    
    # Model training
    input_shape = (timesteps, X_train.shape[2])
   
    reset_random_seeds()
    cnn_lstm_model = build_cnn_lstm(input_shape)
    train_model(cnn_lstm_model, X_train, y_train, 'cnn_lstm')
    
    """
    reset_random_seeds()
    lstnet_model = build_lstnet(input_shape)
    train_model(lstnet_model, X_train, y_train, 'lstnet')
    
    reset_random_seeds()
    tcn_model = build_tcn(input_shape)
    train_model(tcn_model, X_train, y_train, 'tcn')
    """
    
    # Prediction
    timesteps = 5
    model_path = 'model_save/cnn_lstm.keras'
    data_path = 'data/processed/split_data.pkl'
    
    predict(model_path, data_path, timesteps)

if __name__ == "__main__":
    main()
