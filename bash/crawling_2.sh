echo "crontab start"

pkill chrome

export PYTHONPATH=$PATHONPATH:/opt/ml/final_project/

cd /opt/ml/final_project
/opt/conda/envs/lightweight/bin/python3 /opt/ml/final_project/crawling/new_review_crawling.py --num 2
/opt/conda/envs/lightweight/bin/python3 /opt/ml/final_project/crawling/preprocess_review.py