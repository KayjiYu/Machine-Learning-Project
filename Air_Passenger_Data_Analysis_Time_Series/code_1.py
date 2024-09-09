#--------------------------------------------------------
# Import Modules
#--------------------------------------------------------
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')

#--------------------------------------------------------
# Load the Dataset
#--------------------------------------------------------
def load_data():
    df = pd.read_csv('AirPassengers.csv')
    df.set_index('Month', inplace=True)
    return df


def plot_basic():
    plt.figure(figsize=(20,5))
    df = load_data()
    plt.plot(df.index, df['#Passengers'], label='#Passengers')
    plt.xlabel('Date')
    plt.ylabel('#Passengers')
    plt.title('#Passengers Over Time')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=90)
    plt.show()
print(load_data())
print(plot_basic())
def result():
    df = load_data()
    result = seasonal_decompose(df['#Passengers'], model='multiplicative', period=12)
    return result
print(result())
def components_plot():
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(18,12))
    data = result()

    plt.subplot(411)
    sns.lineplot(data=data.trend)
    plt.title('Trend')
    plt.xticks(rotation=90)

    plt.subplot(412)
    sns.lineplot(data=data.seasonal)    
    plt.title('Seasonal')
    plt.xticks(rotation=90)

    plt.subplot(413)
    sns.lineplot(data=data.resid)
    plt.title('Residuals')
    plt.xticks(rotation=90)

    plt.subplot(414)
    df = load_data()
    sns.lineplot(data=df['#Passengers'])
    plt.title('Original Data')
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.show()
print(components_plot())

from statsmodels.tsa.stattools import adfuller
#
def adf():
    df = load_data()
    result = adfuller(df['#Passengers'], autolag='AIC')
    return result[0], result[1]
print(f"ADF Statistics: {adf()[0]}")
print(f"p-value: {adf()[1]}")

print("------First Order Differencing-------")

def adf1st():
    df = load_data()
    result = adfuller(df['#Passengers'].diff().dropna(), autolag='AIC')
    return result[0], result[1]
print(f"ADF Statistics: {adf1st()[0]}")
print(f"p-value: {adf1st()[1]}")


print("------Second Order Differencing-------")

def adf2nd():
    df = load_data()
    result = adfuller(df['#Passengers'].diff().diff().dropna(), autolag='AIC')
    return result[0], result[1]
print(f"ADF Statistics: {adf2nd()[0]}")
print(f"p-value: {adf2nd()[1]}")

# Plot the Differencing values
def plot_diff():
    fig, (ax1, ax2, ax3) = plt.subplots(3)

    df = load_data()

    ax1.plot(df)
    ax1.set_title("Original Time Series")
    ax1.axes.xaxis.set_visible(False)

    ax2.plot(df.diff())
    ax2.set_title("1st Order Differencing")
    ax2.axes.xaxis.set_visible(False)

    ax3.plot(df.diff().diff())
    ax3.set_title("2nd Order Differencing")
    ax3.axes.xaxis.set_visible(False)

    plt.show()
print(plot_diff())

def plot_acf_pacf():
    df = load_data()
    fig, ax = plt.subplots(2, 1, figsize=(12, 7))
    sm.graphics.tsa.plot_acf(df.diff().dropna(), lags=40, ax=ax[0])
    sm.graphics.tsa.plot_pacf(df.diff().dropna(), lags=40, ax=ax[1])
    plt.show()

print(plot_acf_pacf())

from statsmodels.tsa.statespace.sarimax import SARIMAX
def arima_model():
    seasonal_period = 12 
    p, d, q = 2, 1, 1
    P, D, Q = 1, 0, 3
    df = load_data()
    model = SARIMAX(df['#Passengers'], order=(p,d,q), 
                    seasonal_order=(P,D,Q,seasonal_period))
    fitted_model = model.fit()
    print(fitted_model.summary)
print(arima_model())
