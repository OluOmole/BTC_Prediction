import pandas as pd
import os
from utils.project_functions import load_data

def preprocess_data(data):
    """
    Handles missing values, ensures consistency in data types. Return preprocessed data as dataframe.
    """
    try:
        data.ffill(inplace=True)
        data.bfill(inplace=True)
        print("Data preprocessing completed successfully")
        return data
    except Exception as e:
        print(f"Error in preprocessing data: {e}")
        return None