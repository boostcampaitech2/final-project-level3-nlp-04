# hello_world.py

from datetime import timedelta

from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

def print_world() -> None:
    print("world")

# with 구문으로 DAG 정의 시작
with DAG(
    dag_id="hello_world",    # DAG의 식별자용 아이디
    description="My First Dag", # DAG에 대해 설명
    start_date=days_ago(2), # DAG 정의 기준 2일 전부터 시작
    schedule_interval="0 6 * * *", # 매일 06:00에 실행
    tags=["my_dags"],
) as dag:

    # 태스크 정의
    # bash 커맨드로 echo hello 실행
    t1 = BashOperator(
        task_id="print_hello",
        bash_command="echo Hello",
        owner="pseeej", # 이 작업의 오너. 보통 작업을 담당하는 사람 이름 넣음
        retries=3,  # 이 태스크가 실패한 경우, 3번 재시도
        retry_delay=timedelta(minutes=5),   # 재시도하는 시간 간격은 5분
    ) 

    # 태스크 정의
    # python 함수인 print_world 실행
    t2 = PythonOperator(
        task_id="print_world",
        python_callable=print_world,
        depends_on_past=True,
        owner="pseeej",
        retries=3,
        retry_delay=timedelta(minutes=5),
    )

    # 태스크 순서 정함
    # t1 실행 후 t2 실행
    t1 >> t2