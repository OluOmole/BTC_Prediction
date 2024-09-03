import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, BatchNormalization, AveragePooling1D # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras import metrics # type: ignore
from utils.project_functions import reset_random_seeds, f1_score

# Resetting the seeds for reproducibility
reset_random_seeds(seed=1)

# Build the CNN-LSTM model
def build_cnn_lstm(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))    
    model.add(BatchNormalization())
    model.add(AveragePooling1D(pool_size=1))
    model.add(Dropout(0.5))
    model.add(LSTM(units=128, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(LSTM(units=80))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=[
        metrics.BinaryAccuracy(), metrics.Precision(), metrics.Recall(), f1_score])
    return model
