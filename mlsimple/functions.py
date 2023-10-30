import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt

class linearRegression():
    '''
    Given a pandas dataframe of numerical columns, will auto-normalize and perform gradient descent,
    where predictions can then be made. Automatically presents graphs for simple data visualization.
    '''
    
    def initialize_parameters(self, lenw):
        w = np.random.randn(1, lenw)
        b = 0
        return w, b

    def predY(self, x, w, b):
        z = np.dot(w,x) + b
        return z

    def cost_function(self, z, y):
        m = y.shape[1]
        J = (1/(2*m))*np.sum(np.square(z-y))
        return J
    
    def gradientDescent(self, x, y, z, w, b, learning_rate):
        m = y.shape[1]
        dz = (1/m)*(z-y) 
        dw = np.dot(dz,x.T)
        db = np.sum(dz)

        w = w - learning_rate*dw
        b = b - learning_rate*db

        return w, b

    def cost_function(self, z, y):
        m = y.shape[1]
        J = (1/(2*m))*np.sum(np.square(z-y))
        return J
    
    def plotCost(self, costs_train):
        plt.plot(costs_train)
        plt.xlabel('iterations per ten')
        plt.ylabel('cost')
        plt.title('Cost plot')
        plt.show()

    def fit(self, x_train, y_train, x_val, y_val, learning_rate=0.01, epochs=1000):
        lenw = x_train.shape[0]
        w, b = self.initialize_parameters(lenw)

        costs_train = []

        for i in range(epochs):
            z_train = self.predY(x_train, w, b)
            cost_train = self.cost_function(z_train, y_train)
            w, b = self.gradientDescent(x_train, y_train, z_train, w, b, learning_rate)

            if i%10==0:
                costs_train.append(cost_train)

            z_val = self.predY(x_val, w, b)

            cost_val = self.cost_function(z_val, y_val)

            print('Epochs ' + str(i) + '/' + str(epochs) + ': ')
            print('Training Cost ' + str(cost_train) + '|' + 'Validation cost' + str(cost_val))

        self.plotCost(costs_train)

def normalize(df):
    return (df-df.mean())/df.std()

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

