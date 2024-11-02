import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd

from statsmodels.graphics.tsaplots import plot_acf

def plot_residuals_acf(residuals):
    plt.figure(figsize=(7,5)) 
    plot_acf(residuals, lags=50) 
    plt.xlabel('Lags') 
    plt.ylabel('Autocorrelation') 
    plt.title('Autocorrelation Plot') 
    plt.grid(True) 
    plt.show()

def plot_residuals_pacf(residuals):
    plt.figure(figsize=(7, 5)) 
    plot_acf(residuals, lags=50)  
    plt.xlabel('Lags') 
    plt.ylabel('Partial Autocorrelation') 
    plt.title('Partial Autocorrelation Function (PACF) Plot') 
    plt.grid(True) 
    plt.show() 