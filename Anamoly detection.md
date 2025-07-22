!pip install pytrends

from pytrends.request import TrendReq

pytrends = TrendReq()
pytrends.build_payload(kw_list=['football'], timeframe='today 3-m')
data = pytrends.interest_over_time()

print(data.head())

import matplotlib.pyplot as plt

data['football'].plot(figsize=(15,5), title='Google Search Trends: Football')
plt.show()

from sklearn.ensemble import IsolationForest

model = IsolationForest(contamination=0.05)
data['anomaly'] = model.fit_predict(data[['football']])
data['anomaly'] = data['anomaly'].map({1: 0, -1: 1})

anomalies = data[data['anomaly'] == 1]

plt.figure(figsize=(15,5))
plt.plot(data.index, data['football'], label='Search Volume')
plt.scatter(anomalies.index, anomalies['football'], color='red', label='Anomaly')
plt.legend()
plt.title("Anomaly Detection in Google Search Queries")
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pytrends.request import TrendReq

# Fetch Google Trends data
pytrends = TrendReq()
pytrends.build_payload(kw_list=['football'], timeframe='today 3-m')
data = pytrends.interest_over_time()

# Keep only the keyword column
df = data[['football']].copy()
df.dropna(inplace=True)
df = df.reset_index()
df.rename(columns={'date': 'ds', 'football': 'y'}, inplace=True)  # For Prophet

from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.05, random_state=42)
df['anomaly_iso'] = iso.fit_predict(df[['y']])

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_y = scaler.fit_transform(df[['y']])

svm = OneClassSVM(nu=0.05, kernel="rbf", gamma=0.1)
df['anomaly_svm'] = svm.fit_predict(scaled_y)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df[['y']])

# Define autoencoder
autoencoder = Sequential([
    Dense(16, activation='relu', input_shape=(1,)),
    Dense(1, activation='linear')
])

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_scaled, X_scaled, epochs=100, batch_size=16, verbose=0)

# Get reconstruction error
reconstructions = autoencoder.predict(X_scaled)
mse = np.mean(np.square(X_scaled - reconstructions), axis=1)
threshold = np.percentile(mse, 95)

df['anomaly_auto'] = (mse > threshold).astype(int)

from prophet import Prophet

model = Prophet()
model.fit(df[['ds', 'y']])

future = model.make_future_dataframe(periods=0)
forecast = model.predict(future)

# Residuals
df['yhat'] = forecast['yhat']
df['residual'] = abs(df['y'] - df['yhat'])

# Mark anomalies as points with residuals above threshold
residual_threshold = df['residual'].quantile(0.95)
df['anomaly_prophet'] = (df['residual'] > residual_threshold).astype(int)

plt.figure(figsize=(16, 8))
plt.plot(df['ds'], df['y'], label='Search Trend', color='blue')

# Overlay anomalies
plt.scatter(df[df['anomaly_iso'] == -1]['ds'], df[df['anomaly_iso'] == -1]['y'],
            label='IsolationForest', color='red', marker='x')

plt.scatter(df[df['anomaly_svm'] == -1]['ds'], df[df['anomaly_svm'] == -1]['y'],
            label='One-Class SVM', color='green', marker='^')

plt.scatter(df[df['anomaly_auto'] == 1]['ds'], df[df['anomaly_auto'] == 1]['y'],
            label='Autoencoder', color='orange', marker='o')

plt.scatter(df[df['anomaly_prophet'] == 1]['ds'], df[df['anomaly_prophet'] == 1]['y'],
            label='Prophet', color='purple', marker='s')

plt.legend()
plt.title("Anomaly Detection in Google Search Queries - Model Comparison")
plt.xlabel("Date")
plt.ylabel("Search Interest")
plt.grid(True)
plt.show()

pip install streamlit

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pytrends.request import TrendReq
from sklearn.ensemble import IsolationForest

# Title
st.title("📈 Google Search Anomaly Detection Dashboard")

# User input
keyword = st.text_input("Enter a keyword:", "football")

# Fetch data
@st.cache_data
def get_trends(keyword):
    pytrends = TrendReq()
    pytrends.build_payload(kw_list=[keyword], timeframe='today 3-m')
    df = pytrends.interest_over_time()
    df = df[~df.isPartial]
    df = df.reset_index()
    return df

df = get_trends(keyword)
st.write("### Raw Search Trends Data", df)

# Anomaly detection
model = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = model.fit_predict(df[[keyword]])
df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})

# Anomalies Table
anomalies = df[df['anomaly'] == 1]
st.write("### 📌 Detected Anomalies", anomalies[['date', keyword]])

# Plotting
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df['date'], df[keyword], label='Search Interest', color='blue')
ax.scatter(anomalies['date'], anomalies[keyword], color='red', label='Anomalies', s=50)
ax.set_title(f"{keyword.title()} - Search Trend with Anomalies")
ax.set_xlabel("Date")
ax.set_ylabel("Search Volume")
ax.legend()
st.pyplot(fig)
