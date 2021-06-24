#!/usr/bin/python3
import requests
from bs4 import BeautifulSoup

import datetime

now = datetime.datetime.now()

print("Hi EP, weather_pred with python runs every minute -- %s" % (now.strftime("%Y-%m-%d %H:%M:%S")))

def get_weather_forecast():

    url = 'https://weather.gc.ca/forecast/public_bulletins_e.html?Bulletin=fpcn54.cwul'

    response = requests.get(url)

    soup = BeautifulSoup(response.text, 'html.parser')

    forecast = soup.find('pre').text

    forecast = forecast.split('\n\n')

    for n in forecast:
        if(n.find("Metro MontrÃ©al - Laval") >= 0):
            forecast_location = n.split('.\n')

            for f in forecast_location:
                forecast_days = f.split('\n')

                for days in forecast_days:
                    forecast_df = pd.DataFrame(days.split('..'), days.split('.'), days.split('\n'))

                continue

                print(days)
                #parse the message and use 0 as not raining and 1 as raining

            continue