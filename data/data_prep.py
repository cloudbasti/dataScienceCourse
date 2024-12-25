import pandas as pd
from sklearn.preprocessing import LabelEncoder

def prepare_features(df):
    
    # Create a copy to avoid modifying the original
    df_prepared = df.copy()
    
    # Convert date and add weekday
    df_prepared['Datum'] = pd.to_datetime(df_prepared['Datum'])
    df_prepared['Wochentag'] = df_prepared['Datum'].dt.day_name()
    
    # Convert weekdays to numerical values
    weekday_encoder = LabelEncoder()
    df_prepared['Wochentag_encoded'] = weekday_encoder.fit_transform(df_prepared['Wochentag'])
    
    return df_prepared, weekday_encoder

def handle_missing_values(df):
    df_cleaned = df.copy()
    
    # Define columns to check for missing values
    columns_to_check = ['Temperatur', 'Bewoelkung', 'Umsatz', 'Warengruppe']
    
    # Store original length
    original_len = len(df_cleaned)
    
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


def merge_datasets(weather_path, turnover_path, kiwo_path):
    
    # Load all datasets
    weather = pd.read_csv(weather_path)
    turnover = pd.read_csv(turnover_path)
    kiwo = pd.read_csv(kiwo_path)
    
    # Convert dates to datetime in all dataframes
    weather['Datum'] = pd.to_datetime(weather['Datum'])
    turnover['Datum'] = pd.to_datetime(turnover['Datum'])
    kiwo['Datum'] = pd.to_datetime(kiwo['Datum'])
    
    # Merge weather and turnover first
    df = pd.merge(weather, turnover, on='Datum', how='outer')
    
    # Merge with Kieler Woche data
    df = pd.merge(df, kiwo, on='Datum', how='outer')
    
    # Fill NaN values in KielerWoche column with 0
    df['KielerWoche'] = df['KielerWoche'].fillna(0)
    
    return df

