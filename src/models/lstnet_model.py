import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, TimeDistributed, GlobalAveragePooling1D, Add, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras import metrics # type: ignore
from utils.project_functions import reset_random_seeds, f1_score

# Resetting the seeds for reproducibility
reset_random_seeds(seed=1)

# Build the LSTNet model
def build_lstnet(input_shape):
    input_layer = Input(shape=input_shape)

    # CNN component
    cnn = Conv1D(filters=128, kernel_size=3, activation='relu')(input_layer)
    cnn = Conv1D(filters=64, kernel_size=3, activation='relu')(cnn)
    cnn = TimeDistributed(Dense(64, activation='relu'))(cnn)
    cnn = TimeDistributed(Dense(32, activation='relu'))(cnn)
    cnn = TimeDistributed(Dense(1))(cnn)
    cnn = GlobalAveragePooling1D()(cnn)

    # LSTM component
    lstm = LSTM(units=256, return_sequences=True)(input_layer)
    lstm = LSTM(units=128, return_sequences=True)(lstm)
    lstm = TimeDistributed(Dense(64, activation='relu'))(lstm)
    lstm = TimeDistributed(Dense(32, activation='relu'))(lstm)
    lstm = TimeDistributed(Dense(1))(lstm)
    lstm = GlobalAveragePooling1D()(lstm)

    # Autoregression component
    ar = Dense(64, activation='linear')(input_layer)
    ar = Dense(32, activation='linear')(ar)
    ar = Dense(1, activation='linear')(ar)
    ar = GlobalAveragePooling1D()(ar)

    # Merge all components
    merged = Add()([cnn, lstm, ar])
    merged = Dense(64, activation='relu')(merged)
    merged = Dropout(0.5)(merged)
    merged = Dense(32, activation='relu')(merged)
    merged = Dropout(0.5)(merged)
    merged = Dense(1, activation='sigmoid')(merged)

    # Create the model
    model = Model(inputs=input_layer, outputs=merged)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=[
        metrics.BinaryAccuracy(), metrics.Precision(), metrics.Recall(), f1_score])
    return model