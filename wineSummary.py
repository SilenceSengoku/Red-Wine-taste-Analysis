print("red,wine");
import numpy as np
import pandas as pd
from pandas import DataFrame
from matplotlib.pylab import *
import matplotlib.pyplot as plot

#target_url = ("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv")
wine = pd.read_csv("winequality-red.csv",header=0,sep=";")

print(wine.head())

summary = wine.describe()
print(summary);

wineNormalized = wine
ncols = len(wineNormalized.columns)

for i in range(ncols):
    mean = summary.iloc[1,i]
    sd = summary.iloc[2,i]

wineNormalized.iloc[:,i:(i+1)] = \
    (wineNormalized.iloc[:,i:(i+1)]-mean) / sd
array = wineNormalized.values
boxplot(array)
plot.xlabel("attribute Index")
plot.ylabel(("Quartile Ranges - Normalized"))

plot.show()