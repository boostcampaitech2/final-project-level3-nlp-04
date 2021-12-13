echo "crontab start"

pkill chrome

cd /opt/ml/final_project
/opt/conda/envs/lightweight/bin/python3 /opt/ml/final_project/new_review_crawling.py --num 7