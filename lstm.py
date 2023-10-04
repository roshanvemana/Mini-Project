import imp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential  
import datetime
import requests
import streamlit as st
from datetime import date

api_key = '8b56d8c493094900a6580e5ac32b536d'
symbol = 'BTC/USD'
interval = '5min'
order = 'asc'
startTime = '2022-07-13 00:00:00'
end_date = '2022-08-03 00:00:00'
api_url = f'https://api.twelvedata.com/time_series?apikey={api_key}&symbol={symbol}&startTime={startTime}&end_date={end_date}&interval={interval}&order={order}'
data=requests.get(api_url).json()
# print(data.json())
df = pd.DataFrame(data['values'])
# df = df['close']
to_row = int(len(df)*0.9)
train_data = list(df[0:]['close'])
test_data = list(df[to_row:]['close'])
train_data = np.array(train_data)
test_data = np.array(test_data)
# train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1],1))

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(train_data.shape[1],1), activation='relu'))
model.add(Dropout(0.4))
model.add(LSTM(64, return_sequences=True, activation='relu'))
model.add(Dropout(0.3))
model.add(LSTM(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.fit(train_data, test_data, epochs=10, batch_size=64)

def lstm_price_predictor(startTime, endTime, customTime):
    if customTime:
        d0 = datetime.datetime(2022,8,20, 00, 00)
        # d0 = date(int(datetime.datetime.now().strftime("%Y")), int(datetime.datetime.now().strftime("%m")), int(datetime.datetime.now().strftime("%d")), int(datetime.datetime.now().strftime("%H")), int(datetime.datetime.now().strftime("%M")))
        d1 = date(int(customTime.strftime("%Y")), int(customTime.strftime("%m")), int(customTime.strftime("%d")), int(customTime.strftime("%H")), int(customTime.strftime("%M")))
        delta = d1 - d0
        print(delta.seconds)
        price = list(model.predict(start=to_row,end=to_row+delta.seconds/60, type='levels'))[-1]
        return price
    index_future_dates = pd.date_range(start=startTime, end=endTime)
  
    d0 = date(int(startTime.strftime("%Y")), int(startTime.strftime("%m")), int(startTime.strftime("%d")))
    d1 = date(int(endTime.strftime("%Y")), int(endTime.strftime("%m")), int(endTime.strftime("%d")))
    delta = d1 - d0

    pred= model.predict(start=to_row,end=(to_row+delta.days), type='levels')
    print("prediction data:" + str(pred))
    print(len(test_data))
    # print(len(model_predictions))
    print(len(train_data))

#   model_fit.summary()

    plt.figure(figsize=(15,9))
    plt.grid(True)
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
lstm_price_predictor()
# def show_predict_page():
#     st.title("Bitcoin price predection")

#     st.write("""### Please Provide the following details to start predection""")

#     countries = [
#         'USA',
#         'ANGOLA',
#         'INDIA',
#         'US',
#     ]

#     country = st.selectbox("Country", countries)
#     startTime = st.date_input("Select Start Date for price prediction", value=datetime.datetime(2022,8,24,00,00),min_value=datetime.datetime(2014,9,17,00,00))
#     endTime = st.date_input("Select End Date for price prediction", value=datetime.datetime(2022,8,24,23,55),min_value=datetime.datetime(2014,9,17,00,00))
#     print("startTime:"+str(startTime)+" "+"endTime:"+str(endTime))

#     if st.button('Predict'):
#      lstm_price_predictor(startTime, endTime,0)
#      st.pyplot(plt)
#     else:
#      st.write('Hit Button to See Prediction')

#     customTime = st.date_input("Select A Day for price prediction", value=datetime.datetime(2022,8,23),min_value=datetime.datetime.now())
#     if st.button('Get Price'):
#      st.write("The predicted price is:"+"  "+str(lstm_price_predictor(startTime, endTime, customTime)))
#     else:
#      st.write('Hit Button to See Price')


