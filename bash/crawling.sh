echo "crontab start"

export PYTHONPATH=$PATHONPATH:/opt/ml/final_project/

cd /opt/ml/final_project
pkill chrome
/opt/conda/envs/lightweight/bin/python3 /opt/ml/final_project/crawling/new_review_crawling.py --num 1
pkill chrome
/opt/conda/envs/lightweight/bin/python3 /opt/ml/final_project/crawling/new_review_crawling.py --num 2
pkill chrome
/opt/conda/envs/lightweight/bin/python3 /opt/ml/final_project/crawling/new_review_crawling.py --num 3
pkill chrome
/opt/conda/envs/lightweight/bin/python3 /opt/ml/final_project/crawling/new_review_crawling.py --num 4
pkill chrome
/opt/conda/envs/lightweight/bin/python3 /opt/ml/final_project/crawling/new_review_crawling.py --num 5
pkill chrome
/opt/conda/envs/lightweight/bin/python3 /opt/ml/final_project/crawling/new_review_crawling.py --num 6
pkill chrome
/opt/conda/envs/lightweight/bin/python3 /opt/ml/final_project/crawling/new_review_crawling.py --num 7
pkill chrome
/opt/conda/envs/lightweight/bin/python3 /opt/ml/final_project/crawling/preprocess_review.py
