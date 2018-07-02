import pandas as pd
from pandas import DataFrame
from matplotlib.pylab import *
import matplotlib.pyplot as plot
from math import exp

print("paralle")
#target_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
wine = pd.read_csv("winequality-red.csv",header=0,sep=";")

summary = wine.describe()
nrows = len(summary.columns)
tasteCol = len(summary.columns)
meanTaste = summary.iloc[1,tasteCol - 1]
sdTaste = summary.iloc[2,tasteCol-1]
nDataCol = len(wine.columns) -1

#testaera
print(nrows)

for i in range(nrows):
    dataRow = wine.iloc[i,1:nDataCol]
    normTarget = (wine.iloc[i,nDataCol] - meanTaste)/sdTaste
    labelColor = 1.0/(1.0 + exp(-normTarget))
    dataRow.plot(color = plot.cm.RdYlBu(labelColor),alpha=0.5)



plot.xlabel("Attribute Index")
plot.ylabel(("Attribute Values"))
plot.show()

wineNormalized = wine
ncols = len(wineNormalized.columns)
#testaera
print(ncols)

for i in range(ncols):
    mean = summary.iloc[1,i]
    sd = summary.iloc[2,i]
    wineNormalized.iloc[:,i:(i+1)] = (wineNormalized.iloc[:,i:(i+1)]-mean) / sd

for i in range(nrows):
    dataRow = wineNormalized.iloc[i,1:nDataCol]
    normTarget = wineNormalized.iloc[i,nDataCol]
    labelColor = 1.0/(1.0+exp(-normTarget))
    dataRow.plot(color=plot.cm.RdYlBu(labelColor),alpha=0.5)

plot.xlabel("Attribute Index")
plot.ylabel(("Attribute Values"))
plot.show()
