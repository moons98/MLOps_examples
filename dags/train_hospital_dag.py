from airflow.decorators import dag, task
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from pendulum import datetime

from module import train


@dag(
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    doc_md=__doc__,
    default_args={"owner": "Astro", "retries": 3},
    tags=["example"],
)
def train_hospital_pipeline():
    start_task = EmptyOperator(task_id="start_task")
    train_task = PythonOperator(
        task_id="train_task",
        python_callable=train.train_fn,
    )
    end_task = EmptyOperator(task_id="end_task")

    start_task >> train_task >> end_task


train_hospital_pipeline()
