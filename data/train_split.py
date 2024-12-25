def split_train_validation(df_cleaned, feature_columns):
    """
    Split data into training and validation sets based on date ranges.
    Returns features and target variables for both sets.
    """
    # Split data based on date
    train_mask = (df_cleaned['Datum'] >= '2013-07-01') & (df_cleaned['Datum'] <= '2017-07-31')
    test_mask = (df_cleaned['Datum'] >= '2017-08-01') & (df_cleaned['Datum'] <= '2018-07-31')
    
    # Prepare the features and target
    X = df_cleaned[feature_columns]
    y = df_cleaned['Umsatz']
    
    # Split the data using the masks
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    print(f"Training set size: {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test