import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt


def plotTable(df, yCol, plotTitle='Tabular Data Plot', limitFeatures=True, featureLimit=10):
    numx = df.drop(yCol, axis=1).select_dtypes(include=np.number)
    plotCount = len(numx.columns)
    columns = numx.columns

    if limitFeatures and plotCount > featureLimit:
        print(f'Number of columns ({plotCount+1}) exceeds limit, only graphing first {featureLimit}. You can disable this limit by expressing the limitFeatures=False argument, change the limit with featureLimit=N.')
        plotCount = featureLimit

    fig, axs = plt.subplots(nrows=plotCount//2+plotCount % 2, ncols=2, figsize=(15, 12))
    fig.suptitle(plotTitle)

    for i, ax in zip(range(plotCount), axs.ravel()):
        ax.plot(numx[columns[i]], df[yCol], 'o')
        ax.set_title(columns[i])
        ax.set_ylabel('y')
        ax.set_xlabel('x')

    plt.tight_layout()
    plt.show()

    return 0

class linearRegression():
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, x, y):
        self.m, self.n = x.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.x = x
        self.y = y
        self.cHistory = []

        for _ in range(self.iterations):
            self.cHistory.append(self.MSE(self.x, self.y))
            self.update_weights()
        
        print('this is hsitroy', self.cHistory)
        self.costPlot()
        
        return self

    def update_weights(self):
        y_pred = self.predict(self.x)
        dw = (1/self.m) * (self.x.T).dot(y_pred - self.y)
        db = (1/self.m) * np.sum(y_pred - self.y)

        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

        return self

    def predict(self, x):
        return x.astype(float).dot(self.w) + self.b

    def MSE(self, x, y):
        return (y - self.predict(x))**2 * (0.5*self.m)
    
    def costPlot(self):
        print('here')
        plt.plot(range(1, len(self.cHistory) + 1), self.cHistory)
        plt.title("MSE Cost Plot")
        plt.ylabel("MSE Cost")
        plt.xlabel("Iteration")
        plt.show()

def normalize(df):
    return (df-df.mean())/df.std()

df = pd.read_csv('housedata.csv')
#plotTable(df, yCol='Price')

model = linearRegression(iterations = 100, learning_rate=0.000001)

x = df.drop(["Price", "Neighborhood"], axis=1).values
y = df['Price'].values

model.fit(x, y)