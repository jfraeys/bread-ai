#!/usr/bin/python3
from crontab import CronTab


cron = CronTab(user='root')
job_hist = cron.new(command='/Users/eer/anaconda/bin/python3 weather_hist.py')
job_hist.dow.on('tuesday')

job_pred = cron.new(command='/Users/eer/anaconda/bin/python3 weather_pred.py')
job_pred.dow.on('tuesday')

for item in cron:
    print(item)

cron.write()

