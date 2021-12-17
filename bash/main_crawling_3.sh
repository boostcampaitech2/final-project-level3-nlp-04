crontab -r

mkdir -p /opt/ml/final-project-level3-nlp-04/bash/log

(crontab -l 2>/dev/null; echo "0 2 * * * bash /opt/ml/final-project-level3-nlp-04/bash/crawling_3.sh  >> /opt/ml/final-project-level3-nlp-04/bash/log/crawling_3.log 2>&1") | crontab -

service cron restart
crontab -l
