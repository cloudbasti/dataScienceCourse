import pandas as pd
from sklearn.preprocessing import LabelEncoder

def prepare_features(df):
    """
    Prepare features from the raw dataframe.
    Returns processed dataframe with new features.
    """
    # Create a copy to avoid modifying the original
    df_prepared = df.copy()
    
    # Convert date and add weekday
    df_prepared['Datum'] = pd.to_datetime(df_prepared['Datum'])
    df_prepared['Wochentag'] = df_prepared['Datum'].dt.day_name()
    
    # Convert weekdays to numerical values
    label_encoder = LabelEncoder()
    df_prepared['Wochentag_encoded'] = label_encoder.fit_transform(df_prepared['Wochentag'])
    
    # Drop rows with missing values
    columns_to_check = ['Temperatur', 'Bewoelkung', 'Umsatz']
    df_cleaned = df_prepared.dropna(subset=columns_to_check)
    
    # Print the number of rows removed
    rows_removed = len(df_prepared) - len(df_cleaned)
    print(f"Removed {rows_removed} rows with missing values")
    
    return df_cleaned, label_encoder

