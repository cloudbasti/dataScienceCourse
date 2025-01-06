import matplotlib
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

matplotlib.use('Agg')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.train_split import split_train_validation  # NOQA
from data.data_prep import prepare_features  # NOQA
from data.data_prep import merge_datasets  # NOQA
from data.data_prep import handle_missing_values  # NOQA
from data.Final_wetter_imputation import analyze_weather_code_distribution, print_missing_analysis, impute_weather_data  # NOQA


def create_callbacks():
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=50,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=50,
        min_lr=0.00001,
        verbose=1
    )

    return [early_stopping, reduce_lr]


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

    df_with_features['Datum'] = pd.to_datetime(
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

    all_features.extend(['is_seasonal_bread', 'seasonal_bread_intensity'])

    """ product6_features = ['p6_is_season', 'p6_season_intensity']
    mask_p6 = df_with_features['Warengruppe'] == 6
    if mask_p6.any():
        all_features.extend(product6_features)
 """
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

    """ if 6 in df_with_features['Warengruppe'].unique():
        for feature in product6_features:
            # Weather effects
            for weather in weather_features:
                col_name = f'{weather}_{feature}'
                df_with_features[col_name] = df_with_features[feature] * \
                    df_with_features[weather]
                all_features.append(col_name)  # Moved inside the loop

            # Weather code effects
            for weather_code in weather_codes:
                col_name = f'{weather_code}_{feature}'
                df_with_features[col_name] = df_with_features[feature] * \
                    df_with_features[weather_code]
                all_features.append(col_name)  # Moved inside the loop

            # Time effects
            for time in time_features:
                col_name = f'{time}_{feature}'
                df_with_features[col_name] = df_with_features[feature] * \
                    df_with_features[time]
                all_features.append(col_name)  # Moved inside the loop """

    return df_with_features, all_features


def plot_history(history, product_metrics):
    # Create figure with 2 rows, first row for loss and errors, second row for MAPE
    fig = plt.figure(figsize=(24, 8))

    # First row plots
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(history.history['loss'], label='Training')
    ax1.plot(history.history['val_loss'], label='Validation')
    ax1.set_title('Loss Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(history.history['mae'], label='Training')
    ax2.plot(history.history['val_mae'], label='Validation')
    ax2.set_title('Mean Absolute Error Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()

    # Second row plots - MAPE
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(
        history.history['mean_absolute_percentage_error'], label='Training')
    ax3.plot(
        history.history['val_mean_absolute_percentage_error'], label='Validation')
    ax3.set_title('Mean Absolute Percentage Error Over Time')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('MAPE (%)')
    ax3.legend()

    # Product-specific MAPE plot
    ax4 = plt.subplot(2, 2, 4)
    products = list(product_metrics.keys())
    mape_values = [metrics['MAPE'] for metrics in product_metrics.values()]

    ax4.bar(products, mape_values)
    ax4.set_title('MAPE by Product Category')
    ax4.set_ylabel('MAPE (%)')
    ax4.set_xlabel('Product Category')
    for i, v in enumerate(mape_values):
        ax4.text(i, v, f'{v:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()


def prepare_and_predict_umsatz_nn(df):
    """
    Neural network model for predicting turnover with product interactions and polynomial features
    """
    df_with_features, feature_columns = create_product_features(df)

    X_train, X_test, y_train, y_test = split_train_validation(
        df_with_features, feature_columns)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(
        y_train.values.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

    model = Sequential([
        Dense(128, activation='relu', input_shape=(len(feature_columns),),
              kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.30),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.17),
        Dense(1)
    ])

    # model.compile(optimizer=Adam(learning_rate=0.000665),
    model.compile(optimizer=Adam(learning_rate=0.000665),
                  loss='mse',
                  metrics=['mae', tf.keras.metrics.MeanAbsolutePercentageError()])

    callbacks = create_callbacks()

    history = model.fit(X_train_scaled, y_train_scaled,
                        epochs=50,
                        batch_size=32,
                        validation_data=(X_test_scaled, y_test_scaled),
                        callbacks=callbacks,
                        verbose=2)

    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_unscaled = y_test.values.reshape(-1, 1)

    r2 = r2_score(y_test_unscaled, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred))

    # Calculate overall MAPE
    overall_mape = np.mean(
        np.abs((y_test_unscaled - y_pred) / y_test_unscaled)) * 100

    product_metrics = {}
    for product_id in range(1, 7):
        mask = X_test[f'is_product_{product_id}'] == 1
        if mask.any():
            product_r2 = r2_score(y_test_unscaled[mask], y_pred[mask])
            product_rmse = np.sqrt(mean_squared_error(
                y_test_unscaled[mask], y_pred[mask]))
            product_mape = np.mean(
                np.abs((y_test_unscaled[mask] - y_pred[mask]) / y_test_unscaled[mask])) * 100
            product_metrics[f"Product {product_id}"] = {
                'R2': product_r2,
                'RMSE': product_rmse,
                'MAPE': product_mape}

    return model, r2, rmse, product_metrics, scaler_X, scaler_y, history, feature_columns, overall_mape


def main():
    # Load and merge data
    df_merged = merge_datasets()
    df_imputed = impute_weather_data(df_merged)
    df_featured = prepare_features(df_imputed)
    df_cleaned = handle_missing_values(df_featured)

    print("Product 6 features present:", [
          col for col in df_cleaned.columns if 'p6_' in col])
    # Train model
    model, r2, rmse, product_metrics, scaler_X, scaler_y, history, feature_columns, overall_mape = prepare_and_predict_umsatz_nn(
        df_cleaned)

    # Print results
    print(f"\nNeural Network Overall Performance:")
    print(f"R-squared score: {r2:.3f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"Overall MAPE: {overall_mape:.2f}%")

    print("\nNeural Network Performance by Product Category:")
    for category, metrics in product_metrics.items():
        print(f"\n{category}:")
        print(f"R-squared: {metrics['R2']:.3f}")
        print(f"RMSE: {metrics['RMSE']:.2f}")
        print(f"MAPE: {metrics['MAPE']:.2f}%")

    # Plot training history
    plot_history(history, product_metrics)


if __name__ == "__main__":
    main()
