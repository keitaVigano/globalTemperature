import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.seasonal import STL

def seasonal_plot(df_filtered_train):
    plt.figure(figsize=(7, 5)) 
    sns.lineplot(x=df_filtered_train.index.month, y=df_filtered_train['LandAverageTemperature'], ci=None) 
    plt.xlabel('Month') 
    plt.ylabel('Land Average Temperature') 
    plt.title('Seasonal Plot') 
    plt.xticks(range(1, 13), labels=[ 
            'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']) 
    plt.grid(True) 
    plt.show() 

def lag_plot(df_filtered_train):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    pd.plotting.lag_plot(df_filtered_train["LandAverageTemperature"], lag=1, ax=axes[0, 0])
    axes[0, 0].set_title('Lag 1')

    pd.plotting.lag_plot(df_filtered_train["LandAverageTemperature"], lag=2, ax=axes[0, 1])
    axes[0, 1].set_title('Lag 2')

    pd.plotting.lag_plot(df_filtered_train["LandAverageTemperature"], lag=3, ax=axes[1, 0])
    axes[1, 0].set_title('Lag 3')

    pd.plotting.lag_plot(df_filtered_train["LandAverageTemperature"], lag=4, ax=axes[1, 1])
    axes[1, 1].set_title('Lag 4')

    plt.tight_layout()
    plt.show()

def pacf_plot(df_filtered_train):
    plt.figure(figsize=(7, 5)) 
    plot_pacf(df_filtered_train["LandAverageTemperature"], lags=50) 
    plt.xlabel('Lags') 
    plt.ylabel('Partial Autocorrelation') 
    plt.title('Partial Autocorrelation Function (PACF) Plot') 
    plt.grid(True) 
    plt.show() 

def acf_plot(df_filtered_train):
    plt.figure(figsize=(7,5)) 
    plot_acf(df_filtered_train["LandAverageTemperature"], lags=50) 
    plt.xlabel('Lags') 
    plt.ylabel('Autocorrelation') 
    plt.title('Autocorrelation Plot') 
    plt.grid(True) 
    plt.show()

def stl_decomposition_plot(df_filtered_train):
    # StatsModel
    stl = STL(df_filtered_train["LandAverageTemperature"], seasonal=13)  # 12 months + 1 for better smoothing
    result = stl.fit()

    # Extracting the components
    trend = result.trend
    seasonal = result.seasonal
    residual = result.resid

    # Plotting the original data, trend, seasonal, and residual components
    plt.figure(figsize=(10, 8))

    plt.subplot(4, 1, 1)
    plt.plot(df_filtered_train["LandAverageTemperature"], label='Original Data')
    plt.title('Original Data')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(trend, label='Trend', color='orange')
    plt.title('Trend Component')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(seasonal, label='Seasonal Component', color='green')
    plt.title('Seasonal Component')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(residual, label='Residual Component', color='red')
    plt.title('Residual Component')
    plt.legend()

    plt.tight_layout()
    plt.show()
