import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
import joblib
import matplotlib.pyplot as plt

import re

from prophet import Prophet
#import cmdstanpy
#cmdstanpy.install_cmdstan()
#cmdstanpy.install_cmdstan(compiler=True) 


# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path, parse_dates=True, index_col=0)

def scale_non_date_columns(df):

    
    # Filter out date columns and non-numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    non_numeric_cols = df.select_dtypes(exclude=['float64', 'int64']).columns

    if numeric_cols.empty:
        return df, None

    # Initialize the scaler
    scaler = StandardScaler()
    
    # Scale numeric columns
    df_scaled_numeric = scaler.fit_transform(df[numeric_cols])
    df_scaled_numeric = pd.DataFrame(df_scaled_numeric, columns=numeric_cols, index=df.index)
    print("ZZZZZZZZZZZZZZZZZZZZZZZZZ")
    df_scaled = df[non_numeric_cols].copy()
    
    df_scaled = pd.concat([df_scaled, df_scaled_numeric], axis=1)
    print(df_scaled.info())

    return df_scaled, scaler

# Transformer to rename datetime column
class RenameDatetimeColumn(BaseEstimator, TransformerMixin):
    def detect_datetime_column(self, column_name):
        patterns = [
            r'\bdate\b',          # contains 'date'
            r'\bdatetime\b',      # contains 'datetime'
            r'\btimestamp\b',     # contains 'timestamp'
            r'\byear\b',          # contains 'year'
            r'\bmonth\b',         # contains 'month'
            r'\bday\b'            # contains 'day'
        ]
        return any(re.search(pattern, column_name.lower()) for pattern in patterns)

    def convert_to_datetime(self, series, expected_format):
        # Try expected format first
        series_converted = pd.to_datetime(series, errors='coerce', format=expected_format)
        if series_converted.isna().all():
            # If expected format fails, try without format
            series_converted = pd.to_datetime(series, errors='coerce')
            if series_converted.isna().all():
                # If both fail, try 'dd/mm/yyyy' format
                series_converted = pd.to_datetime(series, errors='coerce', format='%d/%m/%Y')
        return series_converted

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Check if the index contains datetime information
        if X.index.name and self.detect_datetime_column(X.index.name):
            try:
                X.index = self.convert_to_datetime(X.index, expected_format='%Y-%m-%d')
                X.index.names = ['Date']
                X=X.reset_index()
                #print("X info : ")
                #print(X.info())
                print(f"Renamed index to 'Date' and converted to datetime format.")
            except ValueError:
                print(f"Failed to convert index to datetime.")
        else:
            # Check each column for datetime patterns and attempt conversion
            for col in X.columns:
                if self.detect_datetime_column(col):
                    try:
                        X[col] = self.convert_to_datetime(X[col], expected_format='%Y-%m-%d')
                        X.rename(columns={col: 'Date'}, inplace=True)
                        print(f"Renamed '{col}' column to 'Date' and converted to datetime format.")
                        return X  # Return immediately after renaming first datetime column
                    except ValueError:
                        print(f"Failed to convert column '{col}' to datetime.")

            # Convert object columns to datetime
            for col in X.select_dtypes(include=['object']):
                try:
                    X[col] = self.convert_to_datetime(X[col], expected_format='%Y-%m-%d')
                except ValueError:
                    pass

            # Identify datetime columns after conversion
            date_columns = X.select_dtypes(include=['datetime64']).columns
            if not date_columns.empty:
                original_name = date_columns[0]
                X.rename(columns={original_name: 'Date'}, inplace=True)
                print(f"Renamed '{original_name}' column to 'Date' and converted to datetime format.")
            else:
                print("No datetime columns found.")

        return X


# Transformer to impute missing values
class ImputeMissingValues(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in X.select_dtypes(include=['object']):
            try:
                X[col] = pd.to_datetime(X[col], errors='coerce')
            except ValueError:
                pass

        X.fillna(X.mean(), inplace=True)
        return X


# Preprocessing pipeline
def preprocess_data_pipeline(file_path, Y_variable, scale):
    # Load data
    df = load_data(file_path)

    # Instantiate transformers
    rename_datetime = RenameDatetimeColumn()
    impute_missing = ImputeMissingValues()

    # Pipeline definition
    pipeline = Pipeline([
        ('rename_datetime', rename_datetime),
        ('impute_missing', impute_missing)
    ])

    # Fit-transform the pipeline
    df_preprocessed = pipeline.fit_transform(df)
    #print("//////////////////////////")
    #print(df_preprocessed.info())
    # Ensure target variable exists
    if Y_variable not in df_preprocessed.columns:
        raise ValueError(f"Target variable '{Y_variable}' not found in the dataset.")
    feature_names = [col for col in df_preprocessed.columns if col != 'Date' and not df_preprocessed[col].isna().any()]
    df_preprocessed = df_preprocessed[['Date'] + feature_names]

    #print("Done")
    #print(df_preprocessed.info())
    # Separate X and Y
    X = df_preprocessed.drop(columns=[Y_variable])
    y = df_preprocessed[Y_variable]
    X=X.rename(columns={'Date': 'ds'})

    # Split data
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if(scale):
      # Scale X data
      X_scaled, scaler = scale_non_date_columns(X)
      if scaler is None:
          print("Not scaling")
          X_scaled = X
          
      else:
          X_scaled, scaler_X = scale_non_date_columns(X)
          #print("SCCAAALLLLEDDD")
          #print(X_scaled.info())
      joblib.dump(scaler, 'scalerr.joblib')

    else:
      X_scaled = X
    y_scaled = y

    X_scaled=X_scaled.rename(columns={'ds': 'Date'})
    # Split data
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y_scaled, test_size=0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    #feature_names = list(X.columns)

    #print("Features : ", feature_names)
    ##if 'Date' in feature_names:
     #   feature_names.remove('Date')

    if X_train_scaled.index.name == 'Date':
        train_df_scaled = X_train_scaled.copy()
        train_df_scaled[Y_variable] = y_train_scaled
        test_df_scaled = X_test_scaled.copy()
        test_df_scaled[Y_variable] = y_test_scaled
        print("date")
    else:
        train_df_scaled = X_train_scaled.copy()
        train_df_scaled['Date'] = X_train_scaled['Date']
        train_df_scaled[Y_variable] = y_train_scaled
        test_df_scaled = X_test_scaled.copy()
        test_df_scaled['Date'] = X_test_scaled['Date']
        test_df_scaled[Y_variable] = y_test_scaled
        print("date2")

    # Rename 'Date' and Y_variable columns for Prophet
    train_df_scaled = train_df_scaled.rename(columns={'Date': 'ds', Y_variable: 'y'})
    test_df_scaled = test_df_scaled.rename(columns={'Date': 'ds', Y_variable: 'y'})

    X_test_scaled_renamed=X_test_scaled.rename(columns={'Date': 'ds'})

    model = Prophet()

    # Add regressors
    feature_names = [col for col in X.columns if col != 'ds' and not X[col].isna().any()]
    for feature in feature_names:
        #print(feature, "added")
        model.add_regressor(feature)

    #print("c bon")
    model.fit(train_df_scaled)
    #print("c bon")
    joblib.dump(model, 'prophet_model.pkl')

    print(X_test_scaled_renamed.info())

    # Predicting
    future = model.make_future_dataframe(periods=len(X_test_scaled_renamed))
    future = future.merge(X_test_scaled_renamed.reset_index(drop=True), on='ds', how='left')

    #forecast = model.predict(future)
    forecast = model.predict(X_test_scaled_renamed)
    y_pred = forecast['yhat'].iloc[-len(X_test):].values

    # Imputing nan values in y_pred
    if np.isnan(y_pred).any():
        y_pred = pd.Series(y_pred).fillna(y_pred.mean()).values
    mape = mean_absolute_percentage_error(y_test_scaled, y_pred)

    model.plot(forecast)
    plt.title('Forecasting')
    plt.xlabel('Date')
    plt.ylabel(Y_variable)
    plt.show()

    #print("Mape : ", mape, "%")

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, mape, model, forecast, feature_names, df_preprocessed
