import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def add_time_series_features(df):
    """
    Add time series features (lagged values) for each product group's turnover.
    """
    df_with_lags = df.copy()
    df_with_lags = df_with_lags.sort_values(['Datum', 'Warengruppe'])

    for product_group in df_with_lags['Warengruppe'].unique():
        mask = df_with_lags['Warengruppe'] == product_group
        for lag in range(1, 8):
            col_name = f'turnover_lag_{lag}_days_group_{product_group}'
            df_with_lags.loc[mask, col_name] = df_with_lags.loc[mask, 'Umsatz'].shift(
                lag)

    lag_columns = [
        col for col in df_with_lags.columns if 'turnover_lag' in col]
    for col in lag_columns:
        product_group = int(col.split('_')[-1])
        mask = df_with_lags['Warengruppe'] == product_group
        mean_value = df_with_lags.loc[mask, col].mean()
        df_with_lags.loc[mask, col] = df_with_lags.loc[mask,
                                                       col].fillna(mean_value)

    return df_with_lags


def prepare_features(df):
    df_prepared = df.copy()

    # Convert date
    df_prepared['Datum'] = pd.to_datetime(df_prepared['Datum'])

    df_prepared = add_time_series_features(df_prepared)

    # Clean weather codes
    df_prepared['Wettercode'] = pd.to_numeric(
        df_prepared['Wettercode'], errors='coerce')
    mask = (df_prepared['Wettercode'] >= 0) & (df_prepared['Wettercode'] <= 99)
    print(f"Invalid weather codes removed: {(~mask).sum()}")

    # Create weather categories
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

    # Print distribution of weather categories
    print("\nWeather category distribution:")
    for col in df_prepared.filter(like='weather_').columns:
        count = df_prepared[col].sum()
        pct = (count / len(df_prepared)) * 100
        print(f"{col}: {count} occurrences ({pct:.2f}%)")

    df_prepared['wind_calm'] = (
        df_prepared['Windgeschwindigkeit'] < 5).astype(int)
    df_prepared['wind_moderate'] = ((df_prepared['Windgeschwindigkeit'] >= 5) &
                                    (df_prepared['Windgeschwindigkeit'] < 15)).astype(int)
    df_prepared['wind_strong'] = (
        df_prepared['Windgeschwindigkeit'] >= 15).astype(int)

    df_prepared['is_nye'] = ((df_prepared['Datum'].dt.month == 12) &
                             (df_prepared['Datum'].dt.day == 31))
    df_prepared['is_nye'] = df_prepared['is_nye'].fillna(0)

    df_prepared['Wochentag'] = df_prepared['Datum'].dt.day_name()
    weekday_dummies = pd.get_dummies(
        df_prepared['Wochentag'], prefix='weekday')
    df_prepared = pd.concat([df_prepared, weekday_dummies], axis=1)

    df_prepared['is_weekend'] = df_prepared['Datum'].dt.dayofweek.isin(
        [5, 6]).map({True: 'Weekend', False: 'Weekday'})
    weekend_dummies = pd.get_dummies(df_prepared['is_weekend'], prefix='is')
    df_prepared = pd.concat([df_prepared, weekend_dummies], axis=1)

    df_prepared['month'] = df_prepared['Datum'].dt.month
    month_dummies = pd.get_dummies(df_prepared['month'], prefix='month')
    df_prepared = pd.concat([df_prepared, month_dummies], axis=1)

    seasons = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    }
    df_prepared['season'] = df_prepared['month'].map(seasons)
    season_dummies = pd.get_dummies(df_prepared['season'], prefix='season')
    df_prepared = pd.concat([df_prepared, season_dummies], axis=1)

    df_prepared['month_sin'] = np.sin(2 * np.pi * df_prepared['month']/12)
    df_prepared['month_cos'] = np.cos(2 * np.pi * df_prepared['month']/12)

    df_prepared['is_pre_holiday'] = df_prepared['Datum'].shift(
        -1).isin(df_prepared[df_prepared['is_holiday'] == 1]['Datum'])
    df_prepared['is_pre_holiday'] = df_prepared['is_pre_holiday'].fillna(0)

    df_prepared['is_weekend_holiday'] = ((df_prepared['is_weekend'] == 'Weekend') &
                                         (df_prepared['is_holiday'] == 1))
    df_prepared['is_weekend_holiday'] = df_prepared['is_weekend_holiday'].fillna(
        0)

    df_prepared['is_summer_weekend'] = ((df_prepared['season'] == 'Summer') &
                                        (df_prepared['is_weekend'] == 'Weekend')).astype(int)
    df_prepared['is_summer_weekend'] = df_prepared['is_summer_weekend'].fillna(
        0)

    df_prepared['is_peak_summer'] = df_prepared['Datum'].dt.month.isin([
                                                                       7, 8]).astype(int)
    df_prepared['is_peak_summer'] = df_prepared['is_peak_summer'].fillna(0)

    df_prepared['is_peak_summer_weekend'] = (df_prepared['is_peak_summer'] &
                                             (df_prepared['is_weekend'] == 'Weekend')).astype(int)
    df_prepared['is_peak_summer_weekend'] = df_prepared['is_peak_summer_weekend'].fillna(
        0)

    return df_prepared


def handle_missing_values(df):
    """
    Remove rows with missing values in key columns and save removed rows to a CSV file.
    """
    df_cleaned = df.copy()
    columns_to_check = ['Temperatur', 'Bewoelkung', 'Umsatz', 'Warengruppe']

    original_len = len(df_cleaned)
    missing_mask = df_cleaned[columns_to_check].isnull().any(axis=1)
    deleted_rows = df_cleaned[missing_mask]
    deleted_rows.to_csv("data/deleted.csv", index=False)
    df_cleaned = df_cleaned.dropna(subset=columns_to_check)

    rows_removed = original_len - len(df_cleaned)
    print(f"Removed {rows_removed} rows with missing values")
    print(f"Retained {len(df_cleaned)} rows")

    missing_counts = df[columns_to_check].isnull().sum()
    print("\nMissing values by column:")
    for col, count in missing_counts.items():
        if count > 0:
            print(f"{col}: {count} missing values")

    return df_cleaned


def merge_datasets():
    weather = pd.read_csv("data/wetter_imputed.csv")
    turnover = pd.read_csv("data/umsatzdaten_gekuerzt.csv")
    kiwo = pd.read_csv("data/kiwo.csv")
    school_holidays = pd.read_csv("data/school_holidays.csv")
    public_holidays = pd.read_csv("data/bank_holidays.csv")

    weather['Datum'] = pd.to_datetime(weather['Datum'])
    turnover['Datum'] = pd.to_datetime(turnover['Datum'])
    kiwo['Datum'] = pd.to_datetime(kiwo['Datum'])
    school_holidays['Datum'] = pd.to_datetime(school_holidays['Datum'])
    public_holidays['Datum'] = pd.to_datetime(public_holidays['Datum'])

    start_date = turnover['Datum'].min()
    end_date = turnover['Datum'].max()
    print(f"Turnover data ranges from {start_date} to {end_date}")

    df = pd.merge(turnover, weather, on='Datum', how='left')
    df = pd.merge(df, kiwo, on='Datum', how='left')
    df = pd.merge(df, school_holidays, on='Datum', how='left')
    df = pd.merge(df, public_holidays, on='Datum', how='left')

    df['KielerWoche'] = df['KielerWoche'].fillna(0)
    df['is_school_holiday'] = df['is_school_holiday'].fillna(0)
    df['is_holiday'] = df['is_holiday'].fillna(0)

    df.to_csv("data/merged_data.csv", index=False)
    return df


def analyze_weather_codes(df):
    print("Weather code distribution:")
    print(df['Wettercode'].value_counts().sort_index())
    print("\nMissing values:", df['Wettercode'].isnull().sum())
    print("\nUnique codes:", len(df['Wettercode'].unique()))


def analyze_wind_data(df):
    print("\nWind Speed Analysis:")
    print("\nBasic Statistics:")
    print(df['Windgeschwindigkeit'].describe())

    print("\nValue Distribution:")
    wind_dist = df['Windgeschwindigkeit'].value_counts().sort_index()
    print(wind_dist)

    print("\nDistribution by Categories:")
    calm = (df['Windgeschwindigkeit'] < 5).sum()
    moderate = ((df['Windgeschwindigkeit'] >= 5) &
                (df['Windgeschwindigkeit'] < 15)).sum()
    strong = (df['Windgeschwindigkeit'] >= 15).sum()

    total = len(df)
    print(f"Calm Wind (<5): {calm} occurrences ({(calm/total*100):.2f}%)")
    print(
        f"Moderate Wind (5-15): {moderate} occurrences ({(moderate/total*100):.2f}%)")
    print(f"Strong Wind (>=15): {
          strong} occurrences ({(strong/total*100):.2f}%)")

    print("\nMissing values:", df['Windgeschwindigkeit'].isnull().sum())


def main():
    print("Merging datasets...")
    df = merge_datasets()

    print("\nHandling missing values...")
    df_cleaned = handle_missing_values(df)

    print("\nAnalyzing weather codes...")
    analyze_weather_codes(df_cleaned)

    print("\nAnalyzing wind data...")
    analyze_wind_data(df_cleaned)

    print("\nPreparing features...")
    df_features = prepare_features(df_cleaned)

    print("\nSaving processed dataset...")
    df_features.to_csv("data/processed_data.csv", index=False)
    print("Processing completed!")


if __name__ == "__main__":
    main()
