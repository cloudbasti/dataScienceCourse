import pandas as pd
import numpy as np


def add_time_series_features(df):
    df_with_lags = df.copy()
    df_with_lags = df_with_lags.sort_values(['Datum', 'Warengruppe'])

    for product_group in df_with_lags['Warengruppe'].unique():
        # Create mask for this product
        mask = df_with_lags['Warengruppe'] == product_group

        # Get only this product's data
        product_data = df_with_lags[mask].copy()

        # Create lags only for this product's data
        for lag in range(1, 8):
            col_name = f'turnover_lag_{lag}_days_group_{product_group}'
            product_data[col_name] = product_data['Umsatz'].shift(lag)

        # Previous week same day
        product_data[f'same_weekday_lag_group_{
            product_group}'] = product_data['Umsatz'].shift(7)

        # Special handling for Product 6 only
        if product_group == 6:
            # Add previous year same day (seasonal pattern)
            product_data['last_year_lag_p6'] = product_data['Umsatz'].shift(
                365)

            # Add season indicator (Oct-Jan)
            product_data['is_season_p6'] = product_data['Datum'].dt.month.isin(
                [10, 11, 12, 1]).astype(int)

            # Add these to lag columns for this product
            lag_columns = [col for col in product_data.columns if 'lag' in col or col in [
                'last_year_lag_p6', 'is_season_p6']]
        else:
            lag_columns = [col for col in product_data.columns if 'lag' in col]

        # Fill NaN values with median for this product
        for col in lag_columns:
            product_data[col] = product_data[col].fillna(
                product_data['Umsatz'].median())

        # Update only this product's rows in the original dataframe
        df_with_lags.loc[mask, lag_columns] = product_data[lag_columns]

    # Any remaining NaN values should be 0 as they're for different products
    lag_columns = [col for col in df_with_lags.columns if 'lag' in col or col in [
        'last_year_lag_p6', 'is_season_p6']]
    df_with_lags[lag_columns] = df_with_lags[lag_columns].fillna(0)

    return df_with_lags


def prepare_features(df):
    df_prepared = df.copy()

    # Convert date
    df_prepared['Datum'] = pd.to_datetime(df_prepared['Datum'])

    df_prepared = add_time_series_features(df_prepared)

    # Create weather categories based on weather codes (Wettercode)
    df_prepared['weather_clear'] = df_prepared['Wettercode'].between(
        0, 9, inclusive='both').astype(int)
    df_prepared['weather_no_precip'] = df_prepared['Wettercode'].between(
        10, 19, inclusive='both').astype(int)
    df_prepared['weather_past_weather'] = df_prepared['Wettercode'].between(
        20, 29, inclusive='both').astype(int)
    df_prepared['weather_dust_sand'] = df_prepared['Wettercode'].between(
        30, 39, inclusive='both').astype(int)
    df_prepared['weather_fog'] = df_prepared['Wettercode'].between(
        40, 49, inclusive='both').astype(int)
    df_prepared['weather_drizzle'] = df_prepared['Wettercode'].between(
        50, 59, inclusive='both').astype(int)
    df_prepared['weather_rain'] = df_prepared['Wettercode'].between(
        60, 69, inclusive='both').astype(int)
    df_prepared['weather_snow'] = df_prepared['Wettercode'].between(
        70, 79, inclusive='both').astype(int)
    df_prepared['weather_shower'] = df_prepared['Wettercode'].between(
        80, 90, inclusive='both').astype(int)
    df_prepared['weather_thunderstorm'] = df_prepared['Wettercode'].between(
        91, 99, inclusive='both').astype(int)

    # Wind speed categories
    df_prepared['wind_calm'] = (
        df_prepared['Windgeschwindigkeit'] < 5).astype(int)
    df_prepared['wind_moderate'] = ((df_prepared['Windgeschwindigkeit'] >= 5) &
                                   (df_prepared['Windgeschwindigkeit'] < 15)).astype(int)
    df_prepared['wind_strong'] = (
        df_prepared['Windgeschwindigkeit'] >= 15).astype(int)

    # New Year's Eve (highest turnover days)
    df_prepared['is_nye'] = (df_prepared['Datum'].dt.month == 12) & (
        df_prepared['Datum'].dt.day == 31)
    df_prepared['is_nye'] = df_prepared['is_nye'].fillna(0)

    # Christmas Eve
    df_prepared['is_christmas_eve'] = (df_prepared['Datum'].dt.month == 12) & (
        df_prepared['Datum'].dt.day == 24)
    df_prepared['is_christmas_eve'] = df_prepared['is_christmas_eve'].fillna(0)

    # Last day of month
    df_prepared['is_last_day_of_month'] = df_prepared['Datum'].dt.is_month_end
    df_prepared['is_last_day_of_month'] = df_prepared['is_last_day_of_month'].fillna(
        0)

    # Create weekday name and dummies
    df_prepared['Wochentag'] = df_prepared['Datum'].dt.day_name()
    weekday_dummies = pd.get_dummies(
        df_prepared['Wochentag'], prefix='weekday')
    df_prepared = pd.concat([df_prepared, weekday_dummies], axis=1)

    # Add is_weekend feature and dummies
    df_prepared['is_weekend'] = df_prepared['Datum'].dt.dayofweek.isin(
        [5, 6]).map({True: 'Weekend', False: 'Weekday'})
    weekend_dummies = pd.get_dummies(df_prepared['is_weekend'], prefix='is')
    df_prepared = pd.concat([df_prepared, weekend_dummies], axis=1)

    # Create month dummy variables
    df_prepared['month'] = df_prepared['Datum'].dt.month
    month_dummies = pd.get_dummies(df_prepared['month'], prefix='month')
    df_prepared = pd.concat([df_prepared, month_dummies], axis=1)

    # Create season feature and dummies
    df_prepared['season'] = df_prepared['month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    })
    season_dummies = pd.get_dummies(df_prepared['season'], prefix='season')
    df_prepared = pd.concat([df_prepared, season_dummies], axis=1)

    # Add cyclical encoding for month
    df_prepared['month_sin'] = np.sin(2 * np.pi * df_prepared['month']/12)
    df_prepared['month_cos'] = np.cos(2 * np.pi * df_prepared['month']/12)

    # Pre-holiday indicator (day before holidays)
    df_prepared['is_pre_holiday'] = df_prepared['Datum'].shift(
        -1).isin(df_prepared[df_prepared['is_holiday'] == 1]['Datum'])
    df_prepared['is_pre_holiday'] = df_prepared['is_pre_holiday'].fillna(
        0)

    # Weekend + Holiday combination
    df_prepared['is_weekend_holiday'] = (
        df_prepared['is_weekend'] == 'Weekend') & (df_prepared['is_holiday'] == 1)
    df_prepared['is_weekend_holiday'] = df_prepared['is_weekend_holiday'].fillna(
        0)

    # Summer weekend indicator
    df_prepared['is_summer_weekend'] = ((df_prepared['season'] == 'Summer') &
                                        (df_prepared['is_weekend'] == 'Weekend')).astype(int)
    df_prepared['is_summer_weekend'] = df_prepared['is_summer_weekend'].fillna(
        0)

    df_prepared['is_june_weekend'] = ((df_prepared['Datum'].dt.month == 6) &
                                      (df_prepared['is_weekend'] == 'Weekend')).astype(int)
    df_prepared['is_june_weekend'] = df_prepared['is_june_weekend'].fillna(0)

    df_prepared['is_december_weekend'] = ((df_prepared['Datum'].dt.month == 12) &
                                          (df_prepared['is_weekend'] == 'Weekend')).astype(int)
    df_prepared['is_december_weekend'] = df_prepared['is_december_weekend'].fillna(
        0)

    # High season indicators (specific summer months)
    df_prepared['is_peak_summer'] = df_prepared['Datum'].dt.month.isin([
        7, 8]).astype(int)
    df_prepared['is_peak_summer'] = df_prepared['is_peak_summer'].fillna(
        0)
    df_prepared['is_peak_summer_weekend'] = (df_prepared['is_peak_summer'] &
                                             (df_prepared['is_weekend'] == 'Weekend')).astype(int)
    df_prepared['is_peak_summer_weekend'] = df_prepared['is_peak_summer_weekend'].fillna(
        0)

    return df_prepared


def handle_missing_values(df):
    # Simply return the dataframe as is, since we're using imputation instead
    # of removing rows with missing values
    return df.copy()


def merge_datasets():
    # Load all datasets
    weather = pd.read_csv("data/WeatherImputation/wetter_imputed.csv")
    turnover = pd.read_csv("data/OriginalData/umsatzdaten_gekuerzt.csv")
    kiwo = pd.read_csv("data/OriginalData/kiwo.csv")
    school_holidays = pd.read_csv("data/HolidayData/school_holidays.csv")
    public_holidays = pd.read_csv("data/HolidayData/bank_holidays.csv")

    # Convert dates to datetime in all dataframes
    weather['Datum'] = pd.to_datetime(weather['Datum'])
    turnover['Datum'] = pd.to_datetime(turnover['Datum'])
    kiwo['Datum'] = pd.to_datetime(kiwo['Datum'])
    school_holidays['Datum'] = pd.to_datetime(school_holidays['Datum'])
    public_holidays['Datum'] = pd.to_datetime(public_holidays['Datum'])

    # Merge weather and turnover first
    df = pd.merge(turnover, weather, on='Datum', how='left')

    # Merge with Kieler Woche data
    df = pd.merge(df, kiwo, on='Datum', how='left')

    df = pd.merge(df, school_holidays, on='Datum', how='left')
    df = pd.merge(df, public_holidays, on='Datum', how='left')

    # Fill NaN values in KielerWoche column with 0
    df['KielerWoche'] = df['KielerWoche'].fillna(0)
    df['is_school_holiday'] = df['is_school_holiday'].fillna(0)
    df['is_holiday'] = df['is_holiday'].fillna(0)

    df.to_csv("data/TrainingPreparation/merged_data.csv", index=False)

    return df