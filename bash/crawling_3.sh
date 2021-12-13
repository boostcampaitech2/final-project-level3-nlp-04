echo "crontab start"

pkill chrome

cd /opt/ml/final_project/crawling
/opt/conda/envs/lightweight/bin/python3 /opt/ml/final_project/crawling/new_review_crawling.py --num 3
/opt/conda/envs/lightweight/bin/python3 /opt/ml/final_project/crawling/preprocess_review.py