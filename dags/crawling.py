# auto_crawling

from datetime import timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

with DAG(
    dag_id="auto_crawling",
    description="Auto Crawling",
    start_date=days_ago(0),
    schedule_interval="0 2 * * *",
    tags=["auto_crawling"],
) as dag:

    t1 = BashOperator(
        task_id="auto_crawling_1",
        bash_command="bash /opt/ml/final-project-level3-nlp-04/bash/crawling_1.sh  >> /opt/ml/final-project-level3-nlp-04/bash/log/crawling_1.log 2>&1",
        owner="pseeej",
        retries=3,
        retry_delay=timedelta(minutes=5),
    )

    t2 = BashOperator(
        task_id="auto_crawling_2",
        bash_command="bash /opt/ml/final-project-level3-nlp-04/bash/crawling_2.sh  >> /opt/ml/final-project-level3-nlp-04/bash/log/crawling_2.log 2>&1",
        owner="pseeej",
        retries=3,
        retry_delay=timedelta(minutes=5),
    )

    t3 = BashOperator(
        task_id="auto_crawling_3",
        bash_command="bash /opt/ml/final-project-level3-nlp-04/bash/crawling_3.sh  >> /opt/ml/final-project-level3-nlp-04/bash/log/crawling_3.log 2>&1",
        owner="pseeej",
        retries=3,
        retry_delay=timedelta(minutes=5),
    )

    t4 = BashOperator(
        task_id="auto_crawling_4",
        bash_command="bash /opt/ml/final-project-level3-nlp-04/bash/crawling_4.sh  >> /opt/ml/final-project-level3-nlp-04/bash/log/crawling_4.log 2>&1",
        owner="pseeej",
        retries=3,
        retry_delay=timedelta(minutes=5),
    )

    t5 = BashOperator(
        task_id="auto_crawling_5",
        bash_command="bash /opt/ml/final-project-level3-nlp-04/bash/crawling_5.sh  >> /opt/ml/final-project-level3-nlp-04/bash/log/crawling_5.log 2>&1",
        owner="pseeej",
        retries=3,
        retry_delay=timedelta(minutes=5),
    )

    t6 = BashOperator(
        task_id="auto_crawling_6",
        bash_command="bash /opt/ml/final-project-level3-nlp-04/bash/crawling_6.sh  >> /opt/ml/final-project-level3-nlp-04/bash/log/crawling_6.log 2>&1",
        owner="pseeej",
        retries=3,
        retry_delay=timedelta(minutes=5),
    )

    t7 = BashOperator(
        task_id="auto_crawling_7",
        bash_command="bash /opt/ml/final-project-level3-nlp-04/bash/crawling_7.sh  >> /opt/ml/final-project-level3-nlp-04/bash/log/crawling_7.log 2>&1",
        owner="pseeej",
        retries=3,
        retry_delay=timedelta(minutes=5),
    )


    [t1, t2, t3, t4, t5, t6, t7]
