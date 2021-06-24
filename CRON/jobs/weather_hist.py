#!/usr/bin/python3

import requests
from bs4 import BeautifulSoup

import datetime

now = datetime.datetime.now()

print("Hi EP, weather_hist with python runs every minute -- %s" % (now.strftime("%Y-%m-%d %H:%M:%S")))

def get_weather_dl(limit_date):

    url = 'https://xn--montral-fya.weatherstats.ca/download.html'
    file = '~/Download/weatherstats_quebec_daily.csv'
    response = requests.get(url)

    soup = BeautifulSoup(response.text, 'html.parser')

    soup.findAll('input')

    print(soup.find(attrs={'value':'daily'}).has_attrs['checked'])

    params = {
        'limit': limit_date
    }

    response = requests.post(url,data=params)
    #print(response)
    #get the file from the download folder

    #weather_data = pd.read_csv(file)