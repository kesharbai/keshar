#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os


# In[2]:


print (os.getcwd())


# In[3]:


os.chdir ('C:\\Users\\vikram\\Desktop\\New folder')
print (os.getcwd())


# In[4]:


data = pd.read_csv('BTC-USD.csv')
display(data)


# In[5]:


fig, ax = plt.subplots(figsize=(20, 8))
ax.plot(data['Date'], data['Close'], color='green')
ax.xaxis.set_major_locator(plt.MaxNLocator(15))
ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('Price in USD', fontsize=14)
plt.title('Bitcoin Prices', fontsize=18)
plt.grid()
plt.show()


# In[6]:



fig, ax = plt.subplots(figsize=(20, 8))
ax.plot(data['Date'], data['Volume'])
ax.xaxis.set_major_locator(plt.MaxNLocator(15))
ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('Volumes', fontsize=14)
plt.title('Bitcoin Volume Trends', fontsize=18)
plt.grid()
plt.show()


# In[7]:


data['Market Cap'] = data['Open'] * data['Volume']
print(data['Market Cap'])


# In[8]:


fig, ax = plt.subplots(figsize=(20, 8))
ax.plot(data['Date'], data['Market Cap'], color='Orange')
ax.xaxis.set_major_locator(plt.MaxNLocator(15))
ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('Market Cap', fontsize=14)
plt.title('Market Cap', fontsize=18)
plt.grid()
plt.show()


# In[9]:


data['vol'] = (data['Close'] / data['Close'].shift(1)) - 1
print(data['vol'])


# In[10]:


fig, ax = plt.subplots(figsize=(20, 8))
ax.plot(data['Date'], data['vol'], color='purple')
ax.xaxis.set_major_locator(plt.MaxNLocator(15))
plt.title('Volatility', fontsize=14)
plt.grid()
plt.show()


# In[11]:


data['Cumulative Return'] = (1 + data['vol']).cumprod()
print(data['Cumulative Return'])


# In[12]:


fig, ax = plt.subplots(figsize=(20, 8))
ax.bar(data['Date'], data['Cumulative Return'], color='Brown')
ax.xaxis.set_major_locator(plt.MaxNLocator(15))
ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('Cumulative Return', fontsize=14)
plt.title('Cumulative Return', fontsize=14)
plt.grid()
plt.show()


# In[13]:


data['MA for 10 days'] = data['Open'].rolling(10).mean()
data['MA for 20 days'] = data['Open'].rolling(20).mean()
data['MA for 50 days'] = data['Open'].rolling(50).mean()
data['MA for 100 days'] = data['Open'].rolling(100).mean()


# In[14]:


truncated_data = data.truncate()
truncated_data[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days', 'MA for 100 days']].plot(subplots=False, figsize=(12, 5))

plt.title('Bitcoin Stock: Adjusted Close Price and Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


# In[15]:


data = data.sort_values(by='Date')
data['Daily_Price_Change'] = data['Close'].diff()
print(data)


# In[16]:


fig1, ax = plt.subplots(figsize=(20, 6))
ax.plot(data['Date'], data['Daily_Price_Change'], color='b', label='Daily Price Change')
ax.xaxis.set_major_locator(plt.MaxNLocator(15))
plt.xlabel('Date')
plt.ylabel('Daily Price Change')
plt.title('Daily Price Change of Bitcoin', fontsize=14)
plt.grid()
plt.show()


# In[17]:


data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(by='Date')

fig = go.Figure(data=[go.Candlestick(x=data['Date'],
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'])])

fig.update_layout(title='Bitcoin Candlestick Chart',
                  xaxis_title='Date',
                  yaxis_title='Price')

fig.show()


# In[18]:


data = data.sort_values(by='Date')

data['Daily_Price_Change'] = data['Close'].diff()


def calculate_rsi(data, window=14):
    delta = data['Daily_Price_Change']
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    relative_strength = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + relative_strength))

    return rsi

data['RSI'] = calculate_rsi(data)

fig, ax = plt.subplots(figsize=(20, 8))
ax.plot(data['Date'], data['RSI'], color='b', label='RSI')
ax.axhline(y=70, color='r', linestyle='--', label='Overbought (70)')
ax.axhline(y=30, color='g', linestyle='--', label='Oversold (30)')
ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('RSI', fontsize=14)
plt.title('Relative Strength Index (RSI) of Bitcoin', fontsize=18)
plt.grid()
plt.show()


# In[ ]:




