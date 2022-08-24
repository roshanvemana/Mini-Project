from itertools import count
import streamlit as st
import base64
import io
import yfinance as yf
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import date

#data sets of different countries
df_us = yf.download('BTC-USD')
df_ind = yf.download('BTC-INR')
df_k = yf.download('BTC-KRW')
df_uk = yf.download('BTC-GBP')
df_rus = yf.download('BTC-RUB')
df_aus = yf.download('BTC-AUD')

############################################################################################################
# FOR USA
#splitting into train and test
to_row_us = int(len(df_us))
train_data_us = list(df_us[0:to_row_us]['Adj Close'])
test_data_us = list(df_us[to_row_us:]['Adj Close'])

def price_predictor_us(startdate, enddate, specificdate):
  model_us = ARIMA(train_data_us, order = (4, 1, 1))
  model_us = model_us.fit()
  if specificdate:
    d0 = date(int(datetime.datetime.now().strftime("%Y")), int(datetime.datetime.now().strftime("%m")), int(datetime.datetime.now().strftime("%d")))
    d1 = date(int(specificdate.strftime("%Y")), int(specificdate.strftime("%m")), int(specificdate.strftime("%d")))
    delta = d1 - d0
    price = list(model_us.predict(start=to_row_us,end=to_row_us+delta.days, type='levels'))[-1]
    return price
  index_future_dates = pd.date_range(start=startdate, end=enddate)
  
  d0 = date(int(startdate.strftime("%Y")), int(startdate.strftime("%m")), int(startdate.strftime("%d")))
  d1 = date(int(enddate.strftime("%Y")), int(enddate.strftime("%m")), int(enddate.strftime("%d")))
  delta = d1 - d0

  pred= model_us.predict(start=to_row_us,end=(to_row_us+delta.days), type='levels')
  print("prediction data:" + str(pred))
  print(len(test_data_us))
  # print(len(model_predictions))
  print(len(train_data_us))

  model_us.summary()

  plt.figure(figsize=(15,9))
  plt.grid(True)

  date_range = df_us.tail(10).index

  plt.plot(index_future_dates, pred, color = 'green', marker='o', linestyle = 'dashed', label = 'BTC predicted price')

  plt.title('Btc price prediction for US')
  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.legend()
  plt.show()
############################################################################################################

#FOR INDIA
#splitting into train and test
to_row_ind = int(len(df_ind))
train_data_ind = list(df_ind[0:to_row_ind]['Adj Close'])
test_data_ind = list(df_ind[to_row_ind:]['Adj Close'])

def price_predictor_ind(startdate, enddate, specificdate):
  model_ind = ARIMA(train_data_ind, order = (4, 1, 1))
  model_ind = model_ind.fit()
  if specificdate:
    d0 = date(int(datetime.datetime.now().strftime("%Y")), int(datetime.datetime.now().strftime("%m")), int(datetime.datetime.now().strftime("%d")))
    d1 = date(int(specificdate.strftime("%Y")), int(specificdate.strftime("%m")), int(specificdate.strftime("%d")))
    delta = d1 - d0
    price = list(model_ind.predict(start=to_row_ind,end=to_row_ind+delta.days, type='levels'))[-1]
    return price
  index_future_dates = pd.date_range(start=startdate, end=enddate)
  
  d0 = date(int(startdate.strftime("%Y")), int(startdate.strftime("%m")), int(startdate.strftime("%d")))
  d1 = date(int(enddate.strftime("%Y")), int(enddate.strftime("%m")), int(enddate.strftime("%d")))
  delta = d1 - d0

  pred = model_ind.predict(start=to_row_ind,end=(to_row_ind+delta.days), type='levels')
  print("prediction data:" + str(pred))
  print(len(test_data_ind))
  # print(len(model_predictions))
  print(len(train_data_ind))

  model_ind.summary()

  plt.figure(figsize=(15,9))
  plt.grid(True)

  date_range = df_ind.tail(10).index

  plt.plot(index_future_dates, pred, color = 'green', marker='o', linestyle = 'dashed', label = 'BTC predicted price')

  plt.title('Btc price prediction for India')
  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.legend()
  plt.show()
############################################################################################################

#FOR AUSTRALIA
#splitting into train and test
to_row_aus = int(len(df_aus))
train_data_aus = list(df_aus[0:to_row_aus]['Adj Close'])
test_data_aus = list(df_aus[to_row_aus:]['Adj Close'])

def price_predictor_aus(startdate, enddate, specificdate):
  model_aus = ARIMA(train_data_aus, order = (4, 1, 1))
  model_aus = model_aus.fit()
  if specificdate:
    d0 = date(int(datetime.datetime.now().strftime("%Y")), int(datetime.datetime.now().strftime("%m")), int(datetime.datetime.now().strftime("%d")))
    d1 = date(int(specificdate.strftime("%Y")), int(specificdate.strftime("%m")), int(specificdate.strftime("%d")))
    delta = d1 - d0
    price = list(model_aus.predict(start=to_row_aus,end=to_row_aus+delta.days, type='levels'))[-1]
    return price
  index_future_dates = pd.date_range(start=startdate, end=enddate)
  
  d0 = date(int(startdate.strftime("%Y")), int(startdate.strftime("%m")), int(startdate.strftime("%d")))
  d1 = date(int(enddate.strftime("%Y")), int(enddate.strftime("%m")), int(enddate.strftime("%d")))
  delta = d1 - d0

  pred = model_aus.predict(start=to_row_aus,end=(to_row_aus+delta.days), type='levels')
  print("prediction data:" + str(pred))
  print(len(test_data_aus))
  # print(len(model_predictions))
  print(len(train_data_aus))

  model_aus.summary()

  plt.figure(figsize=(15,9))
  plt.grid(True)

  date_range = df_aus.tail(10).index

  plt.plot(index_future_dates, pred, color = 'green', marker='o', linestyle = 'dashed', label = 'BTC predicted price')

  plt.title('Btc price prediction for Australia')
  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.legend()
  plt.show()
############################################################################################################

#FOR UK
#splitting into train and test
to_row_uk = int(len(df_uk))
train_data_uk = list(df_uk[0:to_row_uk]['Adj Close'])
test_data_uk = list(df_uk[to_row_uk:]['Adj Close'])

def price_predictor_uk(startdate, enddate, specificdate):
  model_uk = ARIMA(train_data_uk, order = (4, 1, 1))
  model_uk = model_uk.fit()
  if specificdate:
    d0 = date(int(datetime.datetime.now().strftime("%Y")), int(datetime.datetime.now().strftime("%m")), int(datetime.datetime.now().strftime("%d")))
    d1 = date(int(specificdate.strftime("%Y")), int(specificdate.strftime("%m")), int(specificdate.strftime("%d")))
    delta = d1 - d0
    price = list(model_uk.predict(start=to_row_uk,end=to_row_uk+delta.days, type='levels'))[-1]
    return price
  index_future_dates = pd.date_range(start=startdate, end=enddate)
  
  d0 = date(int(startdate.strftime("%Y")), int(startdate.strftime("%m")), int(startdate.strftime("%d")))
  d1 = date(int(enddate.strftime("%Y")), int(enddate.strftime("%m")), int(enddate.strftime("%d")))
  delta = d1 - d0

  pred = model_uk.predict(start=to_row_uk,end=(to_row_uk+delta.days), type='levels')
  print("prediction data:" + str(pred))
  print(len(test_data_uk))
  # print(len(model_predictions))
  print(len(train_data_uk))

  model_uk.summary()

  plt.figure(figsize=(15,9))
  plt.grid(True)

  date_range = df_uk.tail(10).index

  plt.plot(index_future_dates, pred, color = 'green', marker='o', linestyle = 'dashed', label = 'BTC predicted price')

  plt.title('Btc price prediction for United Kingdom')
  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.legend()
  plt.show()
############################################################################################################

#FOR S KOREA
#splitting into train and test
to_row_k = int(len(df_k))
train_data_k = list(df_k[0:to_row_k]['Adj Close'])
test_data_k = list(df_k[to_row_k:]['Adj Close'])

def price_predictor_k(startdate, enddate, specificdate):
  model_k = ARIMA(train_data_k, order = (4, 1, 1))
  model_k = model_k.fit()
  if specificdate:
    d0 = date(int(datetime.datetime.now().strftime("%Y")), int(datetime.datetime.now().strftime("%m")), int(datetime.datetime.now().strftime("%d")))
    d1 = date(int(specificdate.strftime("%Y")), int(specificdate.strftime("%m")), int(specificdate.strftime("%d")))
    delta = d1 - d0
    price = list(model_k.predict(start=to_row_k,end=to_row_k+delta.days, type='levels'))[-1]
    return price
  index_future_dates = pd.date_range(start=startdate, end=enddate)
  
  d0 = date(int(startdate.strftime("%Y")), int(startdate.strftime("%m")), int(startdate.strftime("%d")))
  d1 = date(int(enddate.strftime("%Y")), int(enddate.strftime("%m")), int(enddate.strftime("%d")))
  delta = d1 - d0

  pred = model_k.predict(start=to_row_k,end=(to_row_k+delta.days), type='levels')
  print("prediction data:" + str(pred))
  print(len(test_data_k))
  # print(len(model_predictions))
  print(len(train_data_k))

  model_k.summary()

  plt.figure(figsize=(15,9))
  plt.grid(True)

  date_range = df_k.tail(10).index

  plt.plot(index_future_dates, pred, color = 'green', marker='o', linestyle = 'dashed', label = 'BTC predicted price')

  plt.title('Btc price prediction for South Korea')
  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.legend()
  plt.show()
############################################################################################################

#FOR RUSSIA
#splitting into train and test
to_row_rus = int(len(df_rus))
train_data_rus = list(df_rus[0:to_row_rus]['Adj Close'])
test_data_rus = list(df_rus[to_row_rus:]['Adj Close'])

def price_predictor_rus(startdate, enddate, specificdate):
  model_rus = ARIMA(train_data_rus, order = (4, 1, 1))
  model_rus = model_rus.fit()
  if specificdate:
    d0 = date(int(datetime.datetime.now().strftime("%Y")), int(datetime.datetime.now().strftime("%m")), int(datetime.datetime.now().strftime("%d")))
    d1 = date(int(specificdate.strftime("%Y")), int(specificdate.strftime("%m")), int(specificdate.strftime("%d")))
    delta = d1 - d0
    price = list(model_rus.predict(start=to_row_rus,end=to_row_rus+delta.days, type='levels'))[-1]
    return price
  index_future_dates = pd.date_range(start=startdate, end=enddate)
  
  d0 = date(int(startdate.strftime("%Y")), int(startdate.strftime("%m")), int(startdate.strftime("%d")))
  d1 = date(int(enddate.strftime("%Y")), int(enddate.strftime("%m")), int(enddate.strftime("%d")))
  delta = d1 - d0

  pred = model_rus.predict(start=to_row_rus,end=(to_row_rus+delta.days), type='levels')
  print("prediction data:" + str(pred))
  print(len(test_data_rus))
  # print(len(model_predictions))
  print(len(train_data_rus))

  model_rus.summary()

  plt.figure(figsize=(15,9))
  plt.grid(True)

  date_range = df_rus.tail(10).index

  plt.plot(index_future_dates, pred, color = 'green', marker='o', linestyle = 'dashed', label = 'BTC predicted price')

  plt.title('Btc price prediction for Russia')
  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.legend()
  plt.show()
############################################################################################################

#web app
def show_predict_page():
    st.title("Bitcoin price prediction")
   
    file_ = open("bitcoin.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="bitcoin gif">', unsafe_allow_html=True,)

    st.write("""### Please Provide the following details to start predection""")

    countries = [
        'AUSTRALIA',    #aud
        'INDIA',        #inr
        'RUSSIA',       #rub
        'SOUTH KOREA',  #krw
        'USA',          #usd
        'UNITED KINGDOM'#gbp
    ]


    country = st.selectbox("Country", countries)

    startDate = st.date_input("Select Start Date for price prediction", value=datetime.datetime(2022,8,23),min_value=datetime.datetime(2014,9,17))
    endDate = st.date_input("Select End Date for price prediction", value=datetime.datetime(2022,8,30),min_value=datetime.datetime(2014,9,17))
    print("startDate:"+str(startDate)+" "+"endDate:"+str(endDate))

    #for US
    if country == 'USA':
      print('us')
      if st.button('Predict'):
        price_predictor_us(startDate, endDate,0)
        st.pyplot(plt)
        
        fn = 'plot.jpg'
        img = io.BytesIO()
        plt.savefig(img, format='jpg') #saving the graph in plot.jpg
        st.download_button(label="Download Plot", data=img, file_name=fn, mime="image/jpg")
      else:
        st.write('Hit Button to See Prediction')

      specificdate = st.date_input("Select A Day for price prediction", value=datetime.datetime(2022,8,25),min_value=datetime.datetime.now())
      if st.button('Get Price'):
        st.write("The predicted price is "+" "+"$"+str(price_predictor_us(startDate, endDate, specificdate)))
      else:
        st.write('Hit Button to See Price')

    #for IND
    elif(country == 'INDIA'):
      print('India')
      if st.button('Predict'):
        price_predictor_ind(startDate, endDate,0)
        st.pyplot(plt)
        fn = 'plot.jpg'
        img = io.BytesIO()
        plt.savefig(img, format='jpg')
        st.download_button(label="Download Plot", data=img, file_name=fn, mime="image/jpg")
      else:
        st.write('Hit Button to See Prediction')

      specificdate = st.date_input("Select A Day for price prediction", value=datetime.datetime(2022,8,25),min_value=datetime.datetime.now())
      if st.button('Get Price'):
        st.write("The predicted price is "+" "+"₹"+str(price_predictor_ind(startDate, endDate, specificdate)))
      else:
        st.write('Hit Button to See Price')

    #for AUS
    elif(country == 'AUSTRALIA'):
      print('australia')
      if st.button('Predict'):
        price_predictor_aus(startDate, endDate,0)
        st.pyplot(plt)
        fn = 'plot.jpg'
        img = io.BytesIO()
        plt.savefig(img, format='jpg')
        st.download_button(label="Download Plot", data=img, file_name=fn, mime="image/jpg")
      else:
        st.write('Hit Button to See Prediction')

      specificdate = st.date_input("Select A Day for price prediction", value=datetime.datetime(2022,8,25),min_value=datetime.datetime.now())
      if st.button('Get Price'):
        st.write("The predicted price is "+" "+"$"+str(price_predictor_aus(startDate, endDate, specificdate)))
      else:
        st.write('Hit Button to See Price')

    #for RUSSIA
    elif(country == 'RUSSIA'):
      print('russia')
      if st.button('Predict'):
        price_predictor_rus(startDate, endDate,0)
        st.pyplot(plt)
        fn = 'plot.jpg'
        img = io.BytesIO()
        plt.savefig(img, format='jpg')
        st.download_button(label="Download Plot", data=img, file_name=fn, mime="image/jpg")
      else:
        st.write('Hit Button to See Prediction')

      specificdate = st.date_input("Select A Day for price prediction", value=datetime.datetime(2022,8,25),min_value=datetime.datetime.now())
      if st.button('Get Price'):
        st.write("The predicted price is "+" "+"₽"+str(price_predictor_rus(startDate, endDate, specificdate)))
      else:
        st.write('Hit Button to See Price')

    #for UK
    elif(country == 'UNITED KINGDOM'):
      print('uk')
      if st.button('Predict'):
        price_predictor_uk(startDate, endDate,0)
        st.pyplot(plt)
        fn = 'plot.jpg'
        img = io.BytesIO()
        plt.savefig(img, format='jpg')
        st.download_button(label="Download Plot", data=img, file_name=fn, mime="image/jpg")
      else:
        st.write('Hit Button to See Prediction')

      specificdate = st.date_input("Select A Day for price prediction", value=datetime.datetime(2022,8,25),min_value=datetime.datetime.now())
      if st.button('Get Price'):
        st.write("The predicted price is "+" "+"£"+str(price_predictor_uk(startDate, endDate, specificdate)))
      else:
        st.write('Hit Button to See Price')

    #for S KOREA
    elif(country == 'SOUTH KOREA'):
      print('korea')
      if st.button('Predict'):
        price_predictor_k(startDate, endDate,0)
        st.pyplot(plt)
        fn = 'plot.jpg'
        img = io.BytesIO()
        plt.savefig(img, format='jpg')
        st.download_button(label="Download Plot", data=img, file_name=fn, mime="image/jpg")
      else:
        st.write('Hit Button to See Prediction')

      specificdate = st.date_input("Select A Day for price prediction", value=datetime.datetime(2022,8,25),min_value=datetime.datetime.now())
      if st.button('Get Price'):
        st.write("The predicted price is "+" "+"₩"+str(price_predictor_k(startDate, endDate, specificdate)))
      else:
        st.write('Hit Button to See Price')

      
    
