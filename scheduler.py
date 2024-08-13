import asyncio
import os
import base64
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import uuid
import datetime
from pipeline import preprocess_data_pipeline  # Adjust this import according to your module structure

scheduler = AsyncIOScheduler()

tasks = {}

class TaskStatus:
    WAITING = "waiting"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    ERROR = "error"

# Global store for prediction data
global_store = {
    "model": None,
    "scaler_path": None,
    "df_preprocessed": None,
    "feature_names": None,
    "Y_variable": None
}

async def process_task(task_id, name, file, f, scale):
    tasks[task_id]['status'] = TaskStatus.IN_PROGRESS
    print("PROCESSING TASK : ",task_id)
    print(f"GET /status/{task_id} - Checking status")
    print(f"STATUS OF TASK {task_id}: {tasks[task_id]}")
    #print(tasks[task_id])
    try:
        os.makedirs("temp", exist_ok=True)
        s = scale.lower() == "yes"

        # Convert bytes to a file-like object
        file_like = BytesIO(f)
        
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, mape, model, forecast, feature_names, df_preprocessed = preprocess_data_pipeline(file_like, name, s)

        global_store["scaler_path"] = 'scalerr.joblib' if s else None
        global_store["model"] = model
        global_store["df_preprocessed"] = df_preprocessed
        global_store["feature_names"] = feature_names
        global_store["Y_variable"] = name

        if X_train_scaled.index.name == 'Date':
            train_df_scaled = pd.DataFrame({'Date': X_train_scaled.index, name: y_train_scaled})
        else:
            train_df_scaled = pd.DataFrame({'Date': X_train_scaled['Date'], name: y_train_scaled})

        buffer = StringIO()
        train_df_scaled.info(buf=buffer)
        info_str = buffer.getvalue()

        fig, ax = plt.subplots()
        model.plot(forecast, ax=ax)
        ax.set_title('Forecasting')
        ax.set_xlabel('Date')
        ax.set_ylabel(name)

        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.read()).decode('utf-8')

        describe_html = train_df_scaled.describe().to_html()

        context = {
            "name": name,
            "filename": file.filename,
            "content_type": file.content_type,
            "mape": mape,
            "growth": model.growth,
            "components": model.component_modes,
            "info_str": info_str,
            "describe_html": describe_html,
            "scale": s,
            "plot": plot_base64
        }

        tasks[task_id]['status'] = TaskStatus.DONE
        tasks[task_id]['result'] = context
    except Exception as e:
        tasks[task_id]['status'] = TaskStatus.ERROR
        tasks[task_id]['error'] = str(e)

# Function to schedule tasks
def schedule_task(task_id, name, file, f, scale):
    run_date = datetime.datetime.now() + datetime.timedelta(seconds=1)  # Schedule to run after 1 second
    scheduler.add_job(process_task, 'date', run_date=run_date, args=[task_id, name, file, f, scale])

scheduler.start()
