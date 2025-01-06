
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_prep import handle_missing_values  # NOQA
from data.data_prep import merge_datasets  # NOQA
from data.data_prep import prepare_features  # NOQA
from data.train_split import split_train_validation  # NOQA
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

    temp_poly = PolynomialFeatures(degree=3, include_bias=False)
    temp_features = temp_poly.fit_transform(df_with_features[['Temperatur']])
    feature_names = ['Temp', 'Temp2', 'Temp3']

    for i, name in enumerate(feature_names):
        df_with_features[name] = temp_features[:, i]
        all_features.append(name)

    df_with_features['Temp_Cloud'] = df_with_features['Temperatur'] * \
        df_with_features['Bewoelkung']
    all_features.append('Temp_Cloud')

    all_features.extend(weekday_dummies)
    all_features.extend(month_dummies)
    all_features.extend(season_dummies)
    all_features.extend(weather_codes)
    all_features.extend(wind_features)

    for product_id in range(1, 7):
        product_col = f'is_product_{product_id}'
        df_with_features[product_col] = (
            df_with_features['Warengruppe'] == product_id).astype(int)
        all_features.append(product_col)

        for weather in weather_features:
            col_name = f'{weather}_product_{product_id}'
            df_with_features[col_name] = df_with_features[product_col] * \
                df_with_features[weather]
            all_features.append(col_name)

        for weather_code in weather_codes:
            col_name = f'{weather_code}_product_{product_id}'
            df_with_features[col_name] = df_with_features[product_col] * \
                df_with_features[weather_code]
            all_features.append(col_name)

        for wind in wind_features:
            col_name = f'{wind}_product_{product_id}'
            df_with_features[col_name] = df_with_features[product_col] * \
                df_with_features[wind]
            all_features.append(col_name)

        for time in time_features:
            col_name = f'{time}_product_{product_id}'
            df_with_features[col_name] = df_with_features[product_col] * \
                df_with_features[time]
            all_features.append(col_name)

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


def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['loss'], label='Training')
    ax1.plot(history.history['val_loss'], label='Validation')
    ax1.set_title('Loss Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(history.history['mae'], label='Training')
    ax2.plot(history.history['val_mae'], label='Validation')
    ax2.set_title('Mean Absolute Error Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()


def prepare_and_predict_umsatz_nn(df, learning_rate=0.001):
    """
    Neural network model for predicting turnover with product interactions
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
        Dense(128, activation='relu', input_shape=(
            len(feature_columns),), kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.42),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.12),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='mse',
                  metrics=['mae'])

    callbacks = create_callbacks()
    history = model.fit(X_train_scaled, y_train_scaled,
                        epochs=50,
                        batch_size=32,
                        validation_data=(X_test_scaled, y_test_scaled),
                        callbacks=callbacks,
                        verbose=1)

    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_unscaled = y_test.values.reshape(-1, 1)

    r2 = r2_score(y_test_unscaled, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred))

    product_metrics = {}
    for product_id in range(1, 7):
        mask = X_test[f'is_product_{product_id}'] == 1
        if mask.any():
            product_r2 = r2_score(y_test_unscaled[mask], y_pred[mask])
            product_rmse = np.sqrt(mean_squared_error(
                y_test_unscaled[mask], y_pred[mask]))
            product_metrics[f"Product {product_id}"] = {
                'R2': product_r2, 'RMSE': product_rmse}

    tf.keras.backend.clear_session()
    return model, r2, rmse, product_metrics, history


def plot_learning_rate_results(results_df):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    ax1.semilogx(results_df['learning_rate'], results_df['r2_score'])
    ax1.set_xlabel('Learning Rate')
    ax1.set_ylabel('R² Score')
    ax1.set_title('R² Score vs Learning Rate')
    ax1.grid(True)

    ax2.semilogx(results_df['learning_rate'], results_df['rmse'])
    ax2.set_xlabel('Learning Rate')
    ax2.set_ylabel('RMSE')
    ax2.set_title('RMSE vs Learning Rate')
    ax2.grid(True)

    ax3.semilogx(results_df['learning_rate'], results_df['min_val_loss'])
    ax3.set_xlabel('Learning Rate')
    ax3.set_ylabel('Minimum Validation Loss')
    ax3.set_title('Min Validation Loss vs Learning Rate')
    ax3.grid(True)

    product_columns = [
        col for col in results_df.columns if 'Product_' in col and '_r2' in col]
    for col in product_columns:
        ax4.semilogx(results_df['learning_rate'], results_df[col],
                     label=col.replace('_r2', ''))
    ax4.set_xlabel('Learning Rate')
    ax4.set_ylabel('R² Score')
    ax4.set_title('Product-specific R² Scores vs Learning Rate')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig('learning_rate_analysis.png')
    plt.close()


def main():
    df_merged = merge_datasets()
    df_imputed = impute_weather_data(df_merged)
    df_featured = prepare_features(df_imputed)
    df_cleaned = handle_missing_values(df_featured)

    min_lr = 0.000500
    max_lr = 0.000600
    num_steps = 10
    learning_rates = np.logspace(np.log10(min_lr), np.log10(max_lr), num_steps)

    results = []
    for lr in learning_rates:
        print(f"\nTraining with learning rate: {lr:.6f}")
        model, r2, rmse, product_metrics, history = prepare_and_predict_umsatz_nn(
            df_cleaned, lr)

        result = {
            'learning_rate': lr,
            'r2_score': r2,
            'rmse': rmse,
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'epochs_trained': len(history.history['loss']),
            'min_val_loss': min(history.history['val_loss'])
        }

        for product, metrics in product_metrics.items():
            result[f"{product}_r2"] = metrics['R2']
            result[f"{product}_rmse"] = metrics['RMSE']

        results.append(result)

        # Save intermediate results after each iteration
        pd.DataFrame(results).to_csv(
            'learning_rate_analysis_intermediate.csv', index=False)

    results_df = pd.DataFrame(results)
    results_df.to_csv('learning_rate_analysis_final.csv', index=False)

    plot_learning_rate_results(results_df)

    best_lr_idx = results_df['r2_score'].idxmax()
    best_lr = results_df.loc[best_lr_idx, 'learning_rate']
    print(f"\nBest learning rate found: {best_lr:.6f}")
    print(f"Best R² score: {results_df.loc[best_lr_idx, 'r2_score']:.3f}")


if __name__ == "__main__":
    main()
