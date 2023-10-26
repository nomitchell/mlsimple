import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt

def plotTable(df, plotTitle='Tabular Data Plot', limitFeatures=True, featureLimit=10, yCol='y'):
    # Need to fix if skipping plots ie if y column is not first and if non numeric row, doesnt leave blank col
    if not isinstance(df, pd.DataFrame):
        print("Dataframe passed is not a valid dataframe. Must be a Pandas DataFrame.")
        return 1

    columns = list(df.columns)
    print(yCol)
    if yCol not in columns:
        if yCol=='y':
            print('Please specify what column is used for y, i.e. (df, yCol="Price", ...')
            return 1
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
    
def linearRegression(df):
    pass

x = pd.read_csv('housedata.csv')
plotTable(x, yCol='Price')