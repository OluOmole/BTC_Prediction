import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras import metrics # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tcn import TCN
from utils.project_functions import reset_random_seeds, f1_score

# Resetting the seeds for reproducibility
reset_random_seeds(seed=1)

# Build the TCN model
def build_tcn(input_shape):
    model = Sequential()
    
    # First TCN layer
    model.add(TCN(input_shape=input_shape, 
                  nb_filters=64, 
                  kernel_size=3, 
                  dilations=[1, 2, 4, 8], 
                  activation='relu',
                  return_sequences=True)) 
    model.add(Dropout(0.5))
    
    # Second TCN layer
    model.add(TCN(nb_filters=64, 
                  kernel_size=3, 
                  dilations=[1, 2, 4, 8, 16], 
                  activation='relu',
                  return_sequences=False))
    model.add(Dropout(0.5))
    
    # Dense output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=[
        metrics.BinaryAccuracy(), metrics.Precision(), metrics.Recall(), f1_score])
    
    return model