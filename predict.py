import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

def predict_features_with_prophet(df_preprocessed, specific_dates, features, Y_variable):
    print("IN IITTT")
    # Assurez-vous que 'Date' est bien la colonne de dates
    combined_forecast = pd.DataFrame({'ds': specific_dates})
    df_preprocessed = df_preprocessed.drop(columns=[Y_variable])
    #df_preprocessed = df_preprocessed.fillna(0)

    for col in features:
        print('Processing column:', col)

        # Préparer le dataframe pour Prophet
        df_prophet = df_preprocessed[['Date', col]].rename(columns={'Date': 'ds', col: 'y'})

        # Initialiser et entraîner le modèle Prophet
        model = Prophet()
        model.fit(df_prophet)

        future = pd.DataFrame({'ds': specific_dates})

        forecast = model.predict(future)

        combined_forecast[col] = forecast['yhat']

    return combined_forecast

def predict(model, scaler_path, dates_to_predict, df_preprocessed, feature_names, Y_variable):
  import joblib
  import pandas as pd

  
#  X_test = pd.DataFrame({'ds': pd.to_datetime(dates_to_predict)})
  
  X_test = predict_features_with_prophet(df_preprocessed, dates_to_predict, feature_names, Y_variable)
  print(X_test.info())
  print("DONE It")

  if(scaler_path==None):
    print(scaler_path)
    print("in")
    print(X_test)
    predictions = model.predict(X_test)
  else :
        scaler = joblib.load(scaler_path)
        ds_column = X_test[['ds']]

        # Drop the 'ds' column and scale the remaining columns
        X_test_numeric = X_test.drop(columns=['ds'])
        X_test_scaled_numeric = scaler.transform(X_test_numeric)

        # Create a dataframe from the scaled numeric data
        X_test_scaled_numeric_df = pd.DataFrame(X_test_scaled_numeric, columns=X_test_numeric.columns, index=X_test.index)

        # Combine the 'ds' column with the scaled numeric data
        X_test_scaled = pd.concat([ds_column, X_test_scaled_numeric_df], axis=1)
        print('scale done')
        predictions = model.predict(X_test_scaled)
        print(predictions)
        #if 'yhat' in predictions.columns and isinstance(predictions['yhat'], pd.Series):
        #  predictions['yhat'] = scaler.inverse_transform(predictions['yhat'].to_numpy().reshape(-1, 1)).flatten()
        #else:
        #  print("Warning: 'yhat' not found or not a Series in predictions. Inverse transform skipped.")

  result_df = pd.DataFrame({'Date': pd.to_datetime(dates_to_predict), 'yhat': predictions['yhat']})
  print("res")
  print(result_df)
  return result_df
