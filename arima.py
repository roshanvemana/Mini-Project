import streamlit as st

import yfinance as yf
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import date

df = yf.download('BTC-USD')
print(df)

plt.plot(df.index, df['Adj Close'])
# plt.show()

#splitting into train and test
to_row = int(len(df))
train_data = list(df[0:]['Adj Close'])
test_data = list(df[to_row:]['Adj Close'])
#train_data
# test_data

# plt.figure(figsize=(10,6))
# plt.grid(True)
# plt.xlabel('Dates')
# plt.ylabel('closing prices')
# plt.plot(df[0:to_row]['Adj Close'],'green', label = 'train data')
# plt.plot(df[to_row:]['Adj Close'],'blue', label = 'test data')
# plt.legend()

# model_predictions = []
# n_test_observations = len(test_data)

# print(n_test_observations)

# for i in range(10):
#   model = ARIMA(train_data, order = (4, 1, 1))
#   model_fit = model.fit()
#   output = model_fit.forecast()
#   print("output is:")
#   print(output)
#   yhat= output[0]
# #   yhat = [640]
#   model_predictions.append(yhat)
#   actual_test_val = test_data[i]

#   train_data.append(actual_test_val)

def price_predictor(startdate, enddate, specificdata):
  model2 = ARIMA(train_data, order = (4, 1, 1))
  model_fit = model2.fit()
  if specificdata:
    d0 = date(int(datetime.datetime.now().strftime("%Y")), int(datetime.datetime.now().strftime("%m")), int(datetime.datetime.now().strftime("%d")))
    d1 = date(int(specificdata.strftime("%Y")), int(specificdata.strftime("%m")), int(specificdata.strftime("%d")))
    delta = d1 - d0
    price = list(model_fit.predict(start=to_row,end=to_row+delta.days, type='levels'))[-1]
    return price
  index_future_dates = pd.date_range(start=startdate, end=enddate)
  
  d0 = date(int(startdate.strftime("%Y")), int(startdate.strftime("%m")), int(startdate.strftime("%d")))
  d1 = date(int(enddate.strftime("%Y")), int(enddate.strftime("%m")), int(enddate.strftime("%d")))
  delta = d1 - d0

  pred= model_fit.predict(start=to_row,end=(to_row+delta.days), type='levels')
  print("prediction data:" + str(pred))
  print(len(test_data))
  # print(len(model_predictions))
  print(len(train_data))

  model_fit.summary()

  plt.figure(figsize=(15,9))
  plt.grid(True)

  # date_range = df.tail(289).index
  date_range = df.tail(10).index

  # plt.plot(date_range, model_predictions[-289:], color = 'green', marker='^', linestyle = 'dashed', label = 'BTC predicted price')
  plt.plot(index_future_dates, pred, color = 'green', marker='^', linestyle = 'dashed', label = 'BTC predicted price')
  # plt.plot(date_range, test_data[-289:], color = 'blue', label = 'BTC actual price')
  # plt.plot(date_range, test_data[-10:], color = 'blue', label = 'BTC actual price')

  plt.title('Btc price prediction')
  plt.xlabel('Date')
  plt.ylabel('Price')
  plt.legend()
  plt.show()

  from sklearn.metrics import r2_score
  # mape = np.mean(np.abs(np.array(model_predictions[-289:]) - np.array(test_data[-289:]))/np.abs(test_data[-289:]))
  # print(mape) #mean absolute percentage error
  # print(r2_score(model_predictions, test_data))
  # print(mean_absolute_error(model_predictions, test_data))

#web app

def show_predict_page():
    st.title("Bitcoin price predection")

    st.write("""### Please Provide the following details to start predection""")

    countries = [
        'USA',
        'ANGOLA',
        'INDIA',
        'US',
    ]

    county = st.selectbox("County", countries)
    startDate = st.date_input("Select Start Date for price prediction", value=datetime.datetime(2022,8,23),min_value=datetime.datetime(2014,9,17))
    endDate = st.date_input("Select End Date for price prediction", value=datetime.datetime(2022,8,30),min_value=datetime.datetime(2014,9,17))
    print("startDate:"+str(startDate)+" "+"endDate:"+str(endDate))

    if st.button('Predict'):
     price_predictor(startDate, endDate,0)
     st.pyplot(plt)
    else:
     st.write('Hit Button to See Prediction')

    specificData = st.date_input("Select A Day for price prediction", value=datetime.datetime(2022,8,23),min_value=datetime.datetime.now())
    if st.button('Get Price'):
     st.write("The predicted price is:"+"  "+str(price_predictor(startDate, endDate, specificData)))
    else:
     st.write('Hit Button to See Price')
