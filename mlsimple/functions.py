import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt

def plotTable(df, yCol, plotTitle='Tabular Data Plot', limitFeatures=True, featureLimit=10):
    # Need to fix if skipping plots ie if y column is not first and if non numeric row, doesnt leave blank col
    if not isinstance(df, pd.DataFrame):
        print("Dataframe passed is not a valid dataframe. Must be a Pandas DataFrame.")
        return 1

    columns = list(df.columns)
    print(yCol)
    if yCol not in columns:
        print('y column could not be found in DataFrame, double check argument.')
        return 1

    plotCount = len(columns) - 1
    
    if limitFeatures and plotCount > featureLimit:
        plotCount = featureLimit
        print(f'Number of columns ({plotCount+1}) exceeds limit, only graphing first {featureLimit}. You can disable this limit by expressing the limitFeatures=False argument, change the limit with featureLimit=N.')
    fig, axs = plt.subplots(nrows=plotCount//2+plotCount%2, ncols=2, figsize=(15,12))
    fig.suptitle(plotTitle)
    for i, ax in zip(range(plotCount), axs.ravel()):
        #if not is_numeric_dtype(df[columns[i + 1]]):
            #print(f'Skipping column "{columns[i + 1]}" since it contains non-numeric data.')
            #continue
        ax.plot(df[columns[i + 1]], df[yCol], 'o')
        ax.set_title(columns[i+1])
        ax.set_ylabel('y')
        ax.set_xlabel('x')
    plt.tight_layout()
    plt.show()
    return 0

def costPlot():
    pass
    
def linearRegression(df, yCol, learning_rate=0.01, epochs=100, doNormalize=True):
    if doNormalize:
        df = normalize(df)
    print('Doing linear regression on data with following features:')
    plotTable(df, yCol=yCol)   

    # There may exist and issue here where there is a column of 1's
    x = np.array(df.drop([yCol], axis=1), dtype=float)
    x = np.hstack((np.ones((x.shape[0],1)), x))
    y = np.array(df[yCol], dtype=float)
    y = np.reshape(y, (y.shape[0], 1))
    theta = np.zeros((x.shape[1], 1))
    theta, J_all = gradient_descent(x, y, theta, learning_rate, epochs)
    J = meanSquaredError(x, y, theta)
    print(f"Paramters: {theta}")

    n_epochs = []
    jplot = []
    count = 0
    for i in J_all

class linearRegression():
    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, x, y):
        self.m, self.n = x.shape
        self.w = np.zeros(self.n)

def normalize(df):
    return (df-df.mean())/df.std()

def meanSquaredError(x, y, theta):
    return ((np.matmul(x, theta)-y).T @ (np.matmul(x, theta)-y))/(2*y.shape[0])

def gradient_descent(x, y, theta, learning_rate, epochs):
    m = x.shape[0]
    J_all = []
    for _ in range(epochs):
        h_x = np.matmul(x, theta)
        derivative_ = (1/m)*(x.T@(h_x - y))
        theta = theta - (learning_rate) * derivative_
        J_all.append(meanSquaredError(x, y, theta))
    return theta, J_all

x = pd.read_csv('housedata.csv')
#plotTable(x, yCol='Price')

linearRegression(x.drop(['Neighborhood'], axis=1), 'Price')