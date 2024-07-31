import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

#Uploading Modelling
model = load_model('/Users/rowaahmed/Downloads/Stock Predictor/Rowa Stock Market Predictor.keras')

st.header('Stock Market Prediction Model')

stock =st.text_input('Enter Stock Symbol', 'GOOG')
start = '2012-01-01'
end = '2023-12-31'

data = yf.download(stock, start ,end)

st.subheader('Stock Data')
st.write(data)

train_data = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
test_data = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = train_data.tail(100)
test_data = pd.concat([pas_100_days, test_data], ignore_index=True)
data_test_scale = scaler.fit_transform(test_data)

#Figure for Price in Green vs Moving Avergage 50 Days (Red)
st.subheader('Price (G) vs Moving Avergae 50 Days (R)')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)

#Figure for Price in Green vs Moving Avergage 50 Days (Red) vs Moving Average 100 Days (Blue)

st.subheader('Price(G) vs MA50(R) vs MA100(B)')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig2)

#Figure for Price in Green vs Moving Avergage 100 Days (Red) vs Moving Average 200 Days (Blue)

st.subheader('Price(G) vs MA100(R) vs MA200(B)')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig3)

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale
y = y * scale

#Figure for Orginal Price (Red) vs Predicted Price (Green)

st.subheader('Original Price(R) vs Predicted Price(G)')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label='Original Price')
plt.plot(y, 'g', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig4)