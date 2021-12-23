echo "crontab start"

pkill chrome

<<<<<<< HEAD
export PYTHONPATH="${PATHONPATH}:/opt/ml/final-project-level3-nlp-04/"

cd /opt/ml/final-project-level3-nlp-04
/opt/conda/envs/lightweight/bin/python3 /opt/ml/final-project-level3-nlp-04/crawling/new_review_crawling.py --num 3
/opt/conda/envs/lightweight/bin/python3 /opt/ml/final-project-level3-nlp-04/crawling/preprocess_review.py
=======
export PYTHONPATH="${PATHONPATH}:/opt/ml/final_project/"

cd /opt/ml/final_project
/opt/conda/envs/lightweight/bin/python3 /opt/ml/final_project/crawling/new_review_crawling.py --num 3
/opt/conda/envs/lightweight/bin/python3 /opt/ml/final_project/crawling/preprocess_review.py
>>>>>>> 2db9c5b733b4b9c4c09e02e862411d62d4d22672
