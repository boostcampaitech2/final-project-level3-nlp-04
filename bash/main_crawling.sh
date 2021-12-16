crontab -r

mkdir -p /opt/ml/final_project/bash/log

(crontab -l 2>/dev/null; echo "20 6 * * * bash /opt/ml/final_project/bash/crawling.sh  >> /opt/ml/final_project/bash/log/crawling.log 2>&1") | crontab -

service cron restart
crontab -l