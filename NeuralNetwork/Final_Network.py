import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
# import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

import matplotlib
matplotlib.use('Agg')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.train_split import split_train_validation  # NOQA
from data.data_prep import prepare_features  # NOQA
from data.data_prep import merge_datasets  # NOQA
from data.data_prep import handle_missing_values  # NOQA
from data.Final_wetter_imputation import analyze_weather_code_distribution, print_missing_analysis, impute_weather_data  # NOQA


def create_product_features(df):
    df_with_features = df.copy()

    weather_features = ['Temperatur', 'Bewoelkung']
    weather_codes = [col for col in df.columns if col.startswith('weather_')]
    wind_features = [col for col in df.columns if col.startswith('wind_')]

    weekday_dummies = [col for col in df.columns if col.startswith('weekday_')]
    month_dummies = [col for col in df.columns if col.startswith('month_')]
    season_dummies = [col for col in df.columns if col.startswith('season_')]

    time_features = weekday_dummies + month_dummies + season_dummies
    event_features = ['is_holiday', 'is_school_holiday', 'is_last_day_of_month', 'is_december_weekend', 'is_june_weekend',
                      'KielerWoche', 'is_nye', 'is_christmas_eve', 'is_weekend_holiday', 'is_pre_holiday']

    all_features = []
    """ df_with_features['Datum'] = pd.to_datetime(
        df_with_features['Datum'], errors='coerce')

    df_with_features['is_seasonal_bread'] = ((df_with_features['Warengruppe'] == 6) &
                                             df_with_features['Datum'].dt.month.isin([10, 11, 12, 1])).astype(int)

    df_with_features['seasonal_bread_intensity'] = 0.0
    seasonal_bread = df_with_features['is_seasonal_bread'] == 1

    # Assign intensity values based on the sales pattern
    df_with_features.loc[seasonal_bread & (
        # October
        df_with_features['Datum'].dt.month == 10), 'seasonal_bread_intensity'] = 0.5
    df_with_features.loc[seasonal_bread & (
        # November (peak)
        df_with_features['Datum'].dt.month == 11), 'seasonal_bread_intensity'] = 1.0
    df_with_features.loc[seasonal_bread & (
        # December (peak)
        df_with_features['Datum'].dt.month == 12), 'seasonal_bread_intensity'] = 1.2
    df_with_features.loc[seasonal_bread & (
        # January
        df_with_features['Datum'].dt.month == 1), 'seasonal_bread_intensity'] = 0.4

    all_features.extend(['is_seasonal_bread', 'seasonal_bread_intensity']) """

    # Temperature polynomials
    temp_poly = PolynomialFeatures(degree=3, include_bias=False)
    temp_features = temp_poly.fit_transform(df_with_features[['Temperatur']])
    feature_names = ['Temp', 'Temp2', 'Temp3']

    for i, name in enumerate(feature_names):
        df_with_features[name] = temp_features[:, i]
        all_features.append(name)

    # Weather interactions
    df_with_features['Temp_Cloud'] = df_with_features['Temperatur'] * \
        df_with_features['Bewoelkung']
    all_features.append('Temp_Cloud')

    all_features.extend(weekday_dummies)
    all_features.extend(month_dummies)
    all_features.extend(season_dummies)
    all_features.extend(weather_codes)
    all_features.extend(wind_features)

    # Product-specific features
    for product_id in range(1, 7):
        product_col = f'is_product_{product_id}'
        df_with_features[product_col] = (
            df_with_features['Warengruppe'] == product_id).astype(int)
        all_features.append(product_col)

        # Weather effects
        for weather in weather_features:
            col_name = f'{weather}_product_{product_id}'
            df_with_features[col_name] = df_with_features[product_col] * \
                df_with_features[weather]
            all_features.append(col_name)

        # Weather code effects
        for weather_code in weather_codes:
            col_name = f'{weather_code}_product_{product_id}'
            df_with_features[col_name] = df_with_features[product_col] * \
                df_with_features[weather_code]
            all_features.append(col_name)

        # Wind effects
        for wind in wind_features:
            col_name = f'{wind}_product_{product_id}'
            df_with_features[col_name] = df_with_features[product_col] * \
                df_with_features[wind]
            all_features.append(col_name)

        # Time effects
        for time in time_features:
            col_name = f'{time}_product_{product_id}'
            df_with_features[col_name] = df_with_features[product_col] * \
                df_with_features[time]
            all_features.append(col_name)

        # Event effects
        for event in event_features:
            col_name = f'{event}_product_{product_id}'
            df_with_features[col_name] = df_with_features[product_col] * \
                df_with_features[event]
            all_features.append(col_name)

        col_name = f'Temp2_product_{product_id}'
        df_with_features[col_name] = df_with_features[product_col] * \
            df_with_features['Temp2']
        all_features.append(col_name)

    return df_with_features, all_features

# ===== MODIFIED: New function for submission prediction =====


def prepare_and_predict_submission_nn(train_df, test_df):
    """Neural network model for predicting test data submission, trained on full dataset"""
    print("\nPreparing features for training and test data...")

    # Create features for training data
    train_with_features, feature_columns = create_product_features(train_df)

    # Scale features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Prepare training data
    X_train = train_with_features[feature_columns]
    y_train = train_df['Umsatz']

    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(
        y_train.values.reshape(-1, 1)).ravel()

    print("\nTraining neural network on full dataset...")
    # Create and train model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(len(feature_columns),),
              kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.40),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.4),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.000557),
                  loss='mse',
                  metrics=['mae'])

    # Train on full dataset
    model.fit(X_train_scaled, y_train_scaled,
              epochs=50,
              batch_size=32,
              verbose=1)

    print("\nMaking predictions on test data...")
    # Prepare test data
    test_with_features, _ = create_product_features(test_df)
    X_test = test_with_features[feature_columns]
    X_test_scaled = scaler_X.transform(X_test)

    # Make predictions
    y_pred_scaled = model.predict(X_test_scaled)
    predictions = scaler_y.inverse_transform(y_pred_scaled)

    return predictions

# ===== MODIFIED: New main function for submission =====


def main():
    print("Loading and preparing training data...")
    # Load training data
    df_merged = merge_datasets()
    df_imputed = impute_weather_data(df_merged)
    df_featured = prepare_features(df_imputed)
    df_cleaned = handle_missing_values(df_featured)

    print("\nLoading test data...")
    # Load test data
    test_df = pd.read_csv("data/test_final.csv")

    # Make predictions
    predictions = prepare_and_predict_submission_nn(df_cleaned, test_df)

    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'Umsatz': predictions.flatten()
    })

    # Save predictions
    submission_df.to_csv("data/submission_nn.csv", index=False)
    print("\nPredictions saved to submission_nn.csv")


if __name__ == "__main__":
    main()
