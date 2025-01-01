import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def prepare_features(df):
    df_prepared = df.copy()
    
    # Convert date
    df_prepared['Datum'] = pd.to_datetime(df_prepared['Datum'])
    
     # Clean weather codes
    df_prepared['Wettercode'] = pd.to_numeric(df_prepared['Wettercode'], errors='coerce')
    mask = (df_prepared['Wettercode'] >= 0) & (df_prepared['Wettercode'] <= 99)
    print(f"Invalid weather codes removed: {(~mask).sum()}")
    
     # Create weather categories based on weather codes (Bewoelkung)
    df_prepared['weather_clear'] = df_prepared['Bewoelkung'].isin(range(0, 10)).astype(int)
    df_prepared['weather_no_precip'] = df_prepared['Bewoelkung'].isin(range(10, 20)).astype(int)
    df_prepared['weather_dust_sand'] = df_prepared['Bewoelkung'].isin(range(30, 40)).astype(int)
    df_prepared['weather_fog'] = df_prepared['Bewoelkung'].isin(range(40, 50)).astype(int)
    df_prepared['weather_drizzle'] = df_prepared['Bewoelkung'].isin(range(50, 60)).astype(int)
    df_prepared['weather_rain'] = df_prepared['Bewoelkung'].isin(range(60, 70)).astype(int)
    df_prepared['weather_snow'] = df_prepared['Bewoelkung'].isin(range(70, 80)).astype(int)
    df_prepared['weather_shower'] = df_prepared['Bewoelkung'].isin(range(80, 91)).astype(int)
    df_prepared['weather_thunderstorm'] = df_prepared['Bewoelkung'].isin(range(91, 100)).astype(int)
    
     # Print distribution of weather categories
    print("\nWeather category distribution:")
    for col in df_prepared.filter(like='weather_').columns:
        count = df_prepared[col].sum()
        pct = (count / len(df_prepared)) * 100
        print(f"{col}: {count} occurrences ({pct:.2f}%)")
    
    # Create weekday name and dummies
    df_prepared['Wochentag'] = df_prepared['Datum'].dt.day_name()
    weekday_dummies = pd.get_dummies(df_prepared['Wochentag'], prefix='weekday')
    df_prepared = pd.concat([df_prepared, weekday_dummies], axis=1)
    
    # Add is_weekend feature and dummies
    df_prepared['is_weekend'] = df_prepared['Datum'].dt.dayofweek.isin([5, 6]).map({True: 'Weekend', False: 'Weekday'})
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
    
    return df_prepared

def handle_missing_values(df):
    """
    Remove rows with missing values in key columns and save removed rows to a CSV file.
    
    Parameters:
    df (pandas.DataFrame): Input dataframe
    
    Returns:
    pandas.DataFrame: Dataframe with rows containing missing values removed
    """
    df_cleaned = df.copy()
    
    # Define columns to check for missing values
    columns_to_check = ['Temperatur', 'Bewoelkung', 'Umsatz', 'Warengruppe']
    
    # Store original length
    original_len = len(df_cleaned)
    
    # Identify rows with missing values
    missing_mask = df_cleaned[columns_to_check].isnull().any(axis=1)
    deleted_rows = df_cleaned[missing_mask]
    
    # Save deleted rows to CSV
    deleted_rows.to_csv("data/deleted.csv", index=False)
    
    # Drop rows with missing values in specified columns
    df_cleaned = df_cleaned.dropna(subset=columns_to_check)
    
    # Print information about removed rows
    rows_removed = original_len - len(df_cleaned)
    print(f"Removed {rows_removed} rows with missing values")
    print(f"Retained {len(df_cleaned)} rows")
    
    # Print which columns had missing values
    missing_counts = df[columns_to_check].isnull().sum()
    print("\nMissing values by column:")
    for col, count in missing_counts.items():
        if count > 0:
            print(f"{col}: {count} missing values")
    
    return df_cleaned


def merge_datasets():
    # Load all datasets
    weather = pd.read_csv("data/wetter.csv")
    turnover = pd.read_csv("data/umsatzdaten_gekuerzt.csv")
    kiwo = pd.read_csv("data/kiwo.csv")
    school_holidays = pd.read_csv("data/school_holidays.csv")
    public_holidays = pd.read_csv("data/bank_holidays.csv")

    
    # Convert dates to datetime in all dataframes
    weather['Datum'] = pd.to_datetime(weather['Datum'])
    turnover['Datum'] = pd.to_datetime(turnover['Datum'])
    kiwo['Datum'] = pd.to_datetime(kiwo['Datum'])
    school_holidays['Datum'] =  pd.to_datetime(school_holidays['Datum'])
    public_holidays['Datum'] =  pd.to_datetime(public_holidays['Datum'])
    
    # Find start and end dates from turnover data
    start_date = turnover['Datum'].min()
    end_date = turnover['Datum'].max()
    
    print(f"Turnover data ranges from {start_date} to {end_date}")
    
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
    
    #print(df.head())
    
    # Filter data to only include dates within the turnover date range
    df = df[(df['Datum'] >= start_date) & (df['Datum'] <= end_date)]
    
    df.to_csv("data/merged_data.csv", index=False)
    
    # here are all rows removed from the final merged dataframe for which no turnover data
    # was available from the very beginning. Its not necessary to handle missing values for this 
    # later anyways if the data didnt exist. 
    
    return df

