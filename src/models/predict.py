import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from utils.project_functions import load_data, f1_score
from data.data_preparation import load_preprocessed_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score as skl_f1_score, roc_auc_score, matthews_corrcoef


def predict(model_path, data_path, timesteps):
    # Load the preprocessed data
    X_train, X_test, y_train, y_test, time_test, price, scaler = load_preprocessed_data(data_path)
    
    # Load the model without compiling it
    model = load_model(model_path, compile=False)
    
    # Compile the model with custom objects
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy', 'precision', 'recall', f1_score])
    
    # Make predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    # Ensure y_test and y_pred are binary arrays
    y_test = y_test.astype(int)
    y_pred = y_pred.astype(int)
        
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1-score:", skl_f1_score(y_test, y_pred))
    print("AUC-ROC Score:", roc_auc_score(y_test, y_pred))
    print("MCC:", matthews_corrcoef(y_test, y_pred))
