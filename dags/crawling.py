# auto_crawling

from datetime import timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

with DAG(
    dag_id="auto_crawling",
    description="Auto Crawling",
    start_date=days_ago(2),
    schedule_interval="0 2 * * *",
    tags=["auto_crawling"],
) as dag:

    t1 = BashOperator(
        task_id="auto_crawling_3",
        bash_command="bash /opt/ml/final-project-level3-nlp-04/bash/crawling_3.sh  >> /opt/ml/final-project-level3-nlp-04/bash/log/crawling_3.log 2>&1",
        owner="pseeej",
        retries=3,
        retry_delay=timedelta(minutes=5),
    )



    t1