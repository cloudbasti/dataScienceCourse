import pandas as pd
# import numpy as np


def merge_test_datasets(test_df):
    # Load all auxiliary datasets
    weather = pd.read_csv("data/wetter.csv")
    kiwo = pd.read_csv("data/kiwo.csv")
    school_holidays = pd.read_csv("data/school_holidays.csv")
    public_holidays = pd.read_csv("data/bank_holidays.csv")

    # Convert dates to datetime in all dataframes
    test_df['Datum'] = pd.to_datetime(test_df['Datum'])
    weather['Datum'] = pd.to_datetime(weather['Datum'])
    kiwo['Datum'] = pd.to_datetime(kiwo['Datum'])
    school_holidays['Datum'] = pd.to_datetime(school_holidays['Datum'])
    public_holidays['Datum'] = pd.to_datetime(public_holidays['Datum'])

    # Start with test_df and merge all other datasets
    df = test_df.copy()

    # Merge weather data
    df = pd.merge(df, weather, on='Datum', how='left')

    # Merge with Kieler Woche data
    df = pd.merge(df, kiwo, on='Datum', how='left')

    # Merge holiday data
    df = pd.merge(df, school_holidays, on='Datum', how='left')
    df = pd.merge(df, public_holidays, on='Datum', how='left')

    # Fill NaN values in categorical columns
    df['KielerWoche'] = df['KielerWoche'].fillna(0)
    df['is_school_holiday'] = df['is_school_holiday'].fillna(0)
    df['is_holiday'] = df['is_holiday'].fillna(0)

    return df


def prepare_features(df):
    df_prepared = df.copy()

    # Convert date
    df_prepared['Datum'] = pd.to_datetime(df_prepared['Datum'])

    # Clean weather codes
    df_prepared['Wettercode'] = pd.to_numeric(
        df_prepared['Wettercode'], errors='coerce')
    mask = (df_prepared['Wettercode'] >= 0) & (df_prepared['Wettercode'] <= 99)
    print(f"Invalid weather codes removed: {(~mask).sum()}")

    # for the weather also the most frequent ones might need to get their own features.

    # Create weather categories based on weather codes (Wettercode)
    df_prepared['weather_clear'] = df_prepared['Wettercode'].between(
        0, 9, inclusive='both').astype(int)
    df_prepared['weather_no_precip'] = df_prepared['Wettercode'].between(
        10, 19, inclusive='both').astype(int)
    df_prepared['weather_past_weather'] = df_prepared['Wettercode'].between(
        20, 29, inclusive='both').astype(int)  # Added this category
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

    # Print distribution of weather categories
    print("\nWeather category distribution:")
    for col in df_prepared.filter(like='weather_').columns:
        count = df_prepared[col].sum()
        pct = (count / len(df_prepared)) * 100
        print(f"{col}: {count} occurrences ({pct:.2f}%)")

    # for the wind the analysis many data is in moderate wind, so this might also be split to
    # a finer accuracy
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

    # Pre-holiday indicator (day before holidays)
    # df_prepared['is_pre_holiday'] = df_prepared['Datum'].shift(-1).isin(df_prepared[df_prepared['is_holiday'] == 1]['Datum'])

    # Weekend + Holiday combination
    # df_prepared['is_weekend_holiday'] = (df_prepared['is_weekend'] == 'Weekend') & (df_prepared['is_holiday'] == 1)

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
