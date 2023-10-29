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

class linearRegression():
    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, x, y):
        self.m, self.n = x.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.x = x
        self.y = y

        for _ in range(self.iterations):
            self.update_weights()
        return self
    
    def update_weights(self):
        y_pred = self.predict(self.x)
        dw = - (2*(self.x.T).dot(self.y - y_pred)) / self.m
        db = - 2*np.sum(self.y - y_pred) / self.m

        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

        return self
    
    def predict(self, x):
        return x.dot(self.w) + self.b

def normalize(df):
    return (df-df.mean())/df.std()

x = pd.read_csv('housedata.csv')
#plotTable(x, yCol='Price')

linearRegression(x.drop(['Neighborhood'], axis=1), 'Price')