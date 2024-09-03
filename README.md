# Bitcoin Price Prediction

## Overview

This project aims to predict Bitcoin prices using various machine learning models, including CNN-LSTM, LSTNet, and TCN. The project involves data cleaning, feature engineering, data preparation, model training, and prediction steps. The project structure is organized to maintain modularity and readability.

## Project Structure

```
bitcoin_price_prediction_with_onchain_data/
├── data/
│   ├── raw/
│   │   ├── bitcoin_data.csv
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_cleaning.py
│   │   ├── data_preparation.py
│   │   └── feature_engineering.py
│   ├── evaluation/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn_lstm_model.py
│   │   ├── lstnet_model.py
│   │   ├── predict.py
│   │   ├── tcn_model.py
│   │   └── train.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── project_functions.py
│   └── main.py
├── .gitignore
├── README.md
├── requirements.txt
└── setup.py
```

## Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.4+
- scikit-learn
- pandas
- numpy
- boruta
- pickle

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Stevenomole/Bitcoin_price_prediction_with_onchain_data.git
   cd Bitcoin_price_prediction_with_onchain_data
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

You can run the entire pipeline using the main script:
```python
python src/main.py
```

## Code Explanation

### Data Cleaning (`data/data_cleaning.py`)
This script loads the raw Bitcoin data, handles missing values, and ensures consistency in data types. The cleaned data is saved to the `processed` directory.

### Feature Engineering (`data/feature_engineering.py`)
This script creates binary classification labels for the price movement and selects relevant features using the Boruta feature selection method.

### Data Preparation (`data/data_preparation.py`)
This script prepares the data for model training by creating sequences for time series data, splitting the data into training and testing sets, and scaling the features.

### Models

- **CNN-LSTM Model (`models/cnn_lstm_model.py`)**: Combines Convolutional Neural Networks and Long Short-Term Memory networks for time series prediction.
- **LSTNet Model (`models/lstnet_model.py`)**: Utilizes LSTM networks and autoregression for time series prediction.
- **TCN Model (`models/tcn_model.py`)**: Uses Temporal Convolutional Networks for time series prediction.

### Training (`models/train.py`)
This script trains the models using the prepared data and saves the trained models to the `model_save` directory.

### Prediction (`models/predict.py`)
This script loads the trained models and makes predictions on the test data, evaluating the model performance.

### Utility Functions (`utils/project_functions.py`)
This script contains helper functions for data loading, resetting random seeds for reproducibility, and calculating the F1 score.
