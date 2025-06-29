import sys
import os
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import pickle

# Добавим путь к папке проекта, чтобы Airflow находил etl-модули
sys.path.append(os.path.expanduser('~/Desktop/breast-cancer-ml-pipeline'))

from etl import load_data, preprocess, train_model, evaluate, save_results

# Пути
DATA_PATH = os.path.expanduser('~/Desktop/breast-cancer-ml-pipeline/etl/data.csv')
MODEL_PATH = os.path.expanduser('~/Desktop/breast-cancer-ml-pipeline/results/model.pkl')
METRICS_PATH = os.path.expanduser('~/Desktop/breast-cancer-ml-pipeline/results/metrics.json')

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 6, 29),
    'retries': 1,
}

dag = DAG(
    dag_id='pipeline_dag',
    default_args=default_args,
    schedule='0 8 * * *',
    catchup=False,
    description='ML pipeline for breast cancer classification'
)

# === ЗАДАЧИ ===

def task_load_data():
    df = load_data.load_data(DATA_PATH)
    return df.to_json()  # ← теперь возвращаем json-строку, сохранится в XCom

def task_preprocess(**kwargs):
    import pandas as pd
    ti = kwargs['ti']
    raw_json = ti.xcom_pull(task_ids='load_data')  # ← вытягиваем return_value
    df = pd.read_json(raw_json)
    X, y, scaler = preprocess.preprocess(df)
    ti.xcom_push(key='X', value=X.tolist())
    ti.xcom_push(key='y', value=y.tolist())

def task_train_model(**kwargs):
    from sklearn.model_selection import train_test_split
    import numpy as np
    ti = kwargs['ti']
    X = np.array(ti.xcom_pull(task_ids='preprocess', key='X'))
    y = np.array(ti.xcom_pull(task_ids='preprocess', key='y'))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = train_model.train_model(X_train, y_train)
    import pickle
    # Сохраняем модель в файл
    with open(MODEL_PATH, 'wb') as f:
    	pickle.dump(model, f)
    ti.xcom_push(key='X_test', value=X_test.tolist())
    ti.xcom_push(key='y_test', value=y_test.tolist())

def task_evaluate(**kwargs):
    import numpy as np, pickle
    # Загружаем модель из файла
    with open(MODEL_PATH, 'rb') as f:
    	model = pickle.load(f)
    ti = kwargs['ti']
    # model = pickle.loads(ti.xcom_pull(task_ids='train_model', key='model'))
    X_test = np.array(ti.xcom_pull(task_ids='train_model', key='X_test'))
    y_test = np.array(ti.xcom_pull(task_ids='train_model', key='y_test'))
    metrics = evaluate.evaluate_model(model, X_test, y_test)
    ti.xcom_push(key='metrics', value=metrics)

def task_save_results(**kwargs):
    ti = kwargs['ti']
    metrics = ti.xcom_pull(task_ids='evaluate', key='metrics')
    save_results.save_metrics(metrics, METRICS_PATH)

# === ОПЕРАТОРЫ ===

load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=task_load_data,
    dag=dag,
)

preprocess_task = PythonOperator(
    task_id='preprocess',
    python_callable=task_preprocess,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=task_train_model,
    dag=dag,
)

evaluate_task = PythonOperator(
    task_id='evaluate',
    python_callable=task_evaluate,
    dag=dag,
)

save_results_task = PythonOperator(
    task_id='save_results',
    python_callable=task_save_results,
    dag=dag,
)

# === СВЯЗИ ===

load_data_task >> preprocess_task >> train_model_task >> evaluate_task >> save_results_task

