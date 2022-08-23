import streamlit as st
from arima import show_predict_page

st.sidebar.selectbox("Choose Your Model", ("AIMA", "LTSM"))

show_predict_page()