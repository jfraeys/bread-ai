from crontabs import CronTab

cron = CronTab(user='jeremiefraeys')
job = cron.new(command='python3 weather_hist.py')
job.dow.on('tuesday')


for item in cron:
    print(item)

cron.write()
