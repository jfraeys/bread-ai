#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import datetime, timedelta, date
import pandas as pd
import datetime


from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.tsa.seasonal as sm
import itertools


#import plotly for visualization
import chart_studio.plotly as py
import plotly.offline as pyoff

#import Keras
#import keras
#from keras.layers import Dense
#from keras.models import Sequential
#from keras.optimizers import Adam
#from keras.callbacks import EarlyStopping
#from keras.utils import np_utils
#from keras.layers import LSTM

from sklearn.model_selection import KFold, cross_val_score, train_test_split

get_ipython().run_line_magic('matplotlib', 'inline')
#initiate plotly
pyoff.init_notebook_mode()


# In[2]:


#re-organize files into one file
bread_data_2014 = pd.read_csv('../data/data_bread/2014-Table.csv')
bread_data_2015 = pd.read_csv('../data/data_bread/2015-Table.csv')
bread_data_2016 = pd.read_csv('../data/data_bread/2016-Table.csv')
bread_data_2017 = pd.read_csv('../data/data_bread/2017-Table.csv')


# In[3]:


weather_data_2014 = pd.read_csv("../data/data_weather/en_climate_daily_ON_6104142_2014_P1D.csv")
weather_data_2015 = pd.read_csv("../data/data_weather/en_climate_daily_ON_6104142_2015_P1D.csv")
weather_data_2016 = pd.read_csv("../data/data_weather/en_climate_daily_ON_6104142_2016_P1D.csv")
weather_data_2017 = pd.read_csv("../data/data_weather/en_climate_daily_ON_6104142_2017_P1D.csv")


# In[4]:


bread_data = pd.DataFrame()
products = bread_data_2014['DESC_1'].unique()
bread_data_years = [bread_data_2014, bread_data_2015, bread_data_2016, bread_data_2017]

for years in bread_data_years:
    years = years.rename(columns={'DESC_1':'product_name', 'qty net' : 'quantity', 'Qty net':'quantity', 'DATE_DELIV':'date_billed', 'Date Facture': 'date_billed'})
    years = years.dropna(how='all', axis=1)
    bread_data = bread_data.append(years, ignore_index=True)

bread_data.date_billed = pd.to_datetime(bread_data.date_billed)

bread_data.info()
bread_data.head(10)


# In[5]:


print(bread_data.head())

#viz sales per week per products
plt.plot(bread_data.date_billed, bread_data.quantity)
plt.show()


# In[6]:


weather_data = pd.DataFrame()
weather_data_years = [weather_data_2014, weather_data_2015, weather_data_2016, weather_data_2017]

for weather in weather_data_years:
    weather = weather.dropna(how='all', axis = 1)
    weather = weather.drop(['Year', 'Month', 'Day'], 1)
    weather_data = weather_data.append(weather, ignore_index=True)


weather_data['Date/Time'] = pd.DatetimeIndex(weather_data['Date/Time'], freq='D')

#experimenting with retrospected weather as the demand for bread may be based on future weather.
#weather_data['Date/Time'] = weather_data['Date/Time'] - datetime.timedelta(days=2)


weather_data.head()


# In[7]:


def merge_datasets(df1, df2, left_on, right_on):

    df = pd.merge(df1, df2, left_on=left_on, right_on=right_on)#change this for something that will only do it once

    if right_on in df.columns:
        df = df.drop([right_on], axis=1)

    return df

#bread_data.info()


# In[8]:


def add_weekday_col(df, datetime):
    df['weekday'] = df[datetime].dt.day_name()

    return df


# In[9]:


def corr(df):
    corr = df.corr()
    corr.style.background_gradient(cmap='coolwarm')


# In[10]:


def fill_missing_dates(df, prod_id, prod_name, fill = False):
    #Assumption: using this to look for first column that is a date, later will be use as index.
    dt = [column for column in df.columns if pd.api.types.is_datetime64_any_dtype(df[column])]
    datetime = df[dt].squeeze()

    #create empty dataframe with all dates missing from the user input dataframe.
    #r = pd.Series(index=pd.date_range(start=date(datetime.dt.year.min(), 1, 1), end=date(datetime.dt.year.max(), 12, 31), freq='D'))
    #deciding whether or not to include the missing dew days of the year

    r = pd.Series(index=pd.date_range(start=datetime.iloc[0], end=date(datetime.dt.year.max(), 12, 31), freq='D'))

    #merge the user input with empty dataframe so that all dates are present. The dates had to be index for this step to\
    # work hence the set_index and drop date_billed so that it is not duplicated
    df = pd.concat([df.set_index(datetime),r[~r.index.isin(df.set_index(datetime).index)]]).sort_index()

    #assign product name and id to new rows
    df['PRODUCT'] = prod_id
    df['product_name'] = prod_name

    if fill:
        df = df.bfill().reset_index()

    else:
        df = df.reset_index()

    df = df.dropna(how='all', axis=1)
    #has I need to use either ffill or bfill, the first or last day of the dataset will be nan as there is nothing to fill it.
    df = df[pd.notnull(df['quantity'])]

    df = df.drop('date_billed', 1).rename(columns={'index':'date_billed'})

    return df

#print(fill_missing_dates(bread_data, 50).tail())


# In[11]:


def set_index_sort(df, index_col):
    df[index_col] = pd.DatetimeIndex(df[index_col], freq = 'D')
    return df.set_index(index_col).sort_index()
    


# In[12]:


def split_by_year(df, year):
    return df.loc[df.index.year < year], df.loc[df.index.year >= year]


# In[13]:


#maybe be better to split at rounded week number
def train_test_split_timeseries(x, y, test_size=0.33):

    years_list = x.index.year.unique().to_list()

    split_index = int(np.ceil(len(years_list) * test_size))
    split_year = years_list[-split_index]

    X_train, X_test = split_by_year(x, split_year)
    y_train, y_test = split_by_year(y, split_year)

    return X_train, X_test, y_train, y_test


# In[ ]:


#change to True to see al plots
viz = False

for prod_id in bread_data.PRODUCT.unique():

    if not prod_id == 51:
        continue

    df = pd.DataFrame()

    df = bread_data.loc[bread_data.PRODUCT == prod_id]

    df = fill_missing_dates(df, prod_id, df.product_name.unique()[0], True)

    df = merge_datasets(df, weather_data, 'date_billed', 'Date/Time')
    
    df = add_weekday_col(df, 'date_billed')

    df = set_index_sort(df, 'date_billed')

    if viz:
        #as the seasonal chart is too frequent let's see one year or month
        seas_d = sm.seasonal_decompose(df.quantity, model='additive')
        seas_d.plot()
        plt.show() #as the seasonal chart is too frequent let's see one year or month

        plt.figure(figsize = (25,4))
        seas_d.seasonal.plot()

    #split method will split at the year mark instead of percentage in as the data contains seasonality that would be lost otherewise.
    #X_train, X_test, y_train, y_test = train_test_split_timeseries(df.loc[:, df.columns != 'quantity'], df.quantity, test_size=0.25)
    X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'quantity'], df.quantity, test_size=0.25)

    ##Models and forecasting##
    p = d = q = range(0,3)
    pdq = list(itertools.product(p,d,p))#get all combination of the tuple

    p2 = d2 = q2 = range(0,2)
    pdq2 = list(itertools.product(p2,d2,p2))
    s=52
    pdqs2 = [(c[0], c[1], c[2], s) for c in pdq2]#get all combination of the tuple

    combs = {}
    aics = []

    for comb in pdq:
        for season_comb in pdqs2:
            try:
                model = SARIMAX(y_train, order=comb, seasonal_order=season_comb)
                model = model.fit()
                print(comb, " : ", season_comb, "-->", model.aic)
                aic = int(model.aic)
                print(aic)
                combs.update({aic : [comb, season_comb]})
                aics.append(aic)

            except:
                continue

    best_aic = min(aics)

    model = SARIMAX(y_train, order=combs[best_aic][0], seasonal_order=combs[best_aic][1])
    model_fit = model.fit()
    
    print(model_fit.summary())
    
    #model_fit.forecast(7)

    #pred = list()
    #hist = [x for x in y_train]

    #for t in range(len(y_test)):
        #model = SARIMAX(hist, order = (0,0,0))
        #model_fit = model.fit(disp=False)
        #output = model_fit.forecast()
        #yhat = output[0]
        #pred.append(yhat)
       # obs = y_test[t]
        #hist.append(obs)

        #print('predicted = %f, actual = %f' % (yhat, obs))

    #error = mean_squared_error(y_test, pred)
    #print('test MSE: %.3f' % error)


# In[ ]:





# In[ ]:




