crontab -r

mkdir -p /opt/ml/final-project-level3-nlp-04/bash/log

(crontab -l 2>/dev/null; echo "20 6 * * * bash /opt/ml/final-project-level3-nlp-04/bash/crawling.sh  >> /opt/ml/final-project-level3-nlp-04/bash/log/crawling.log 2>&1") | crontab -

service cron restart
crontab -l