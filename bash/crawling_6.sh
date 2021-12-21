echo "crontab start"

pkill chrome

export PYTHONPATH="${PATHONPATH}:/opt/ml/final-project-level3-nlp-04/"

cd /opt/ml/final-project-level3-nlp-04
/opt/conda/envs/lightweight/bin/python3 /opt/ml/final-project-level3-nlp-04/crawling/new_review_crawling.py --num 6
/opt/conda/envs/lightweight/bin/python3 /opt/ml/final-project-level3-nlp-04/crawling/preprocess_review.py