import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from models.cnn_lstm_model import build_cnn_lstm
from models.lstnet_model import build_lstnet
from models.tcn_model import build_tcn
from utils.project_functions import load_data, reset_random_seeds
from data.data_preparation import prepare_data, save_preprocessed_data

def train_model(model, X_train, y_train, model_name):
    early_stopping = EarlyStopping(monitor='val_loss', patience=100)
    model.fit(X_train, y_train, epochs=1000, batch_size=50, validation_split=0.1, callbacks=[early_stopping])
    
    # Ensure the save directory exists
    save_dir = 'model_save'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save the model architecture and weights
    model.save(os.path.join(save_dir, f'{model_name}.keras'))