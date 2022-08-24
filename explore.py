import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from datetime import date

#data sets of different countries
df_us = yf.download('BTC-USD')
df_ind = yf.download('BTC-INR')
df_k = yf.download('BTC-KRW')
df_uk = yf.download('BTC-GBP')
df_rus = yf.download('BTC-RUB')
df_aus = yf.download('BTC-AUD')

def show_explore_page():
    st.title('Explore Bitcoin Data')
    st.write("""### Select a country and desired time line""")

    countries = [
        'AUSTRALIA',    #aud
        'INDIA',        #inr
        'RUSSIA',       #rub
        'SOUTH KOREA',  #krw
        'USA',          #usd
        'UNITED KINGDOM'#gbp
    ]

    country = st.selectbox("Country", countries)
    startDate = st.date_input("Select Start Date", value=datetime.datetime(2022,8,10),min_value=datetime.datetime(2014,9,17))
    endDate = st.date_input("Select End Date", value=datetime.datetime(2022,8,23),min_value=datetime.datetime(2014,9,17))
    
    startDate = date(int(startDate.strftime("%Y")), int(startDate.strftime("%m")), int(startDate.strftime("%d")))
    endDate = date(int(endDate.strftime("%Y")), int(endDate.strftime("%m")), int(endDate.strftime("%d")))
    #converting date to datetime for comparison
    #df.index is of type datetime64
    startDate = datetime.datetime(startDate.year, startDate.month, startDate.day)
    endDate = datetime.datetime(endDate.year, endDate.month, endDate.day)
    
    #data specific to countries
    if country == 'USA':
        if st.button('Explore'):
            df_final = df_us[(df_us.index > startDate) & (df_us.index <= endDate)]
            st.dataframe(df_final)
            csv = df_final.to_csv().encode('utf-8')
            st.download_button("Download Data", csv, "btc_data.csv", "text/csv") #download the data frame as csv
            
    elif country == 'INDIA':
        if st.button('Explore'):
            df_final = df_ind[(df_ind.index > startDate) & (df_ind.index <= endDate)]
            st.dataframe(df_final)
            csv = df_final.to_csv().encode('utf-8')
            st.download_button("Download Data", csv, "btc_data.csv", "text/csv")

    elif country == 'AUSTRALIA':
        if st.button('Explore'):
            df_final = df_aus[(df_aus.index > startDate) & (df_aus.index <= endDate)]
            st.dataframe(df_final)
            csv = df_final.to_csv().encode('utf-8')
            st.download_button("Download Data", csv, "btc_data.csv", "text/csv")

    elif country == 'RUSSIA':
        if st.button('Explore'):
            df_final = df_rus[(df_rus.index > startDate) & (df_rus.index <= endDate)]
            st.dataframe(df_final)
            csv = df_final.to_csv().encode('utf-8')
            st.download_button("Download Data", csv, "btc_data.csv", "text/csv")

    elif country == 'UNITED KINGDOM':
        if st.button('Explore'):
            df_final = df_uk[(df_uk.index > startDate) & (df_uk.index <= endDate)]
            st.dataframe(df_final)
            csv = df_final.to_csv().encode('utf-8')
            st.download_button("Download Data", csv, "btc_data.csv", "text/csv")

    elif country == 'SOUTH KOREA':
        if st.button('Explore'):
            df_final = df_k[(df_k.index > startDate) & (df_k.index <= endDate)]
            st.dataframe(df_final)
            csv = df_final.to_csv().encode('utf-8')
            st.download_button("Download Data", csv, "btc_data.csv", "text/csv")