﻿# AutoML_API_Timeseries

## Overview

This project is an AutoML solution for time series prediction using various machine learning techniques. The project is designed to preprocess time series data, train models, and predict future values for specific features.

## Project Structure

The project is organized into the following files:

- **`pipeline.py`**: Contains the data preprocessing pipeline, including scaling, feature selection, and model training.
- **`predict.py`**: Handles the prediction process using the trained models.
- **`scheduler.py`**: Manages asynchronous task scheduling for preprocessing and prediction.
- **`main.py`**: The main FastAPI application that serves as the interface for users to upload data, schedule tasks, and view results.

## Requirements

- Python 3.7+
- FastAPI
- Prophet
- pandas
- scikit-learn
- apscheduler
- matplotlib
- joblib

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/zeineb23/AutoML_API_Timeseries.git
   cd yourproject
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Endpoints 

- GET /: Displays the form to upload a CSV file and start the preprocessing task.
- POST /: Handles the form submission and schedules a preprocessing task.
- GET /status/{task_id}: Returns the status of a scheduled task.
- POST /predict: Handles prediction requests based on a date range.
- GET /results/{task_id}: Displays the results of a completed task.

## File Descriptions

### `pipeline.py`

This file contains the function `preprocess_data_pipeline`, which is responsible for:

- **Reading and Cleaning Data**: Ingests the input CSV file, handles missing values, and prepares the data for further processing.
- **Scaling and Transforming Data**: Applies scaling techniques to normalize the data and ensure it is in the right format for model training.
- **Splitting Data**: Divides the dataset into training and testing subsets to evaluate model performance.
- **Model Training**: Trains a machine learning model on the preprocessed data, preparing it for future predictions.

### `predict.py`

This file contains the function `predict`, which:

- **Loading Trained Model and Scaler**: Retrieves the pre-trained model and scaler used during the preprocessing stage.
- **Predicting Future Values**: Uses the trained model to predict future values for specific features over a defined date range.
- **Scaling and Transforming Predictions**: Ensures the prediction results are scaled back to their original values, if necessary, for accurate interpretation.

### `scheduler.py`

This file manages task scheduling using `AsyncIOScheduler` from the `apscheduler` library. It includes:

- **Global Task Store**: Maintains a global dictionary to track the status of all scheduled tasks, whether waiting, in progress, completed, or errored.
- **Task Scheduling and Processing**: Provides functions to schedule tasks asynchronously, ensuring that data preprocessing and predictions occur in the background without blocking the main application.

### `main.py`

This is the main FastAPI application file, which includes:

- **API Endpoints**: Defines the routes for handling form submissions, task scheduling, status checks, and displaying results to the user.
- **Task Management**: Integrates with `scheduler.py` to schedule and monitor background tasks related to data preprocessing and prediction.
- **HTML Templates**: Utilizes Jinja2 templates to render dynamic content in the user interface, such as displaying prediction results and task statuses.

