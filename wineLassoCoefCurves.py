import pandas as pd
import urllib2
import numpy
from sklearn import datasets ,linear_model
from sklearn.linear_model import LassoCV
from math import sqrt
import matplotlib.pyplot as plot

#print("red wine")

target_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = urllib2.urlopen(target_url)
#data = pd.read_csv("winequality-red.csv",header=0,sep=";")

xList = []
labels = []
names= []
firstLine = True
for line in data:
    if firstLine:
        names = line.strip().split(";")
        firstLine = False
    else:
        row = line.strip().split(";")
        labels.append(float(row[-1]))
        row.pop()
        floatRow = [float(num) for num in row]
        xList.append(floatRow)

nrows = len(xList)
ncols = len(xList[0])

xMeans = []
xSD = []

for i in range(ncols):
    col = [xList[j][i] for j in range(nrows)]
    mean = sum(col)/nrows
    xMeans.append(mean)
    colDiff = [(xList[j][i]-mean) for j in range(nrows)]
    sumSq = sum([colDiff[i] * colDiff[i] for i in range(nrows)])
    stdDev = sqrt(sumSq/nrows)
    xSD.append(stdDev)

xNormalized = []
for i in range(nrows):
        rowNormalized = [(xList[i][j]- xMeans[j])/xSD[j] for j in range(ncols)]
        xNormalized.append(rowNormalized)

meanLabel = sum(labels)/nrows
sdLabel = sqrt(sum([(labels[i]-meanLabel) * (labels[i] - meanLabel)for i in range(nrows)])/nrows)

laberNormalized = [(labels[i]-meanLabel) /sdLabel for i in range(nrows)]

Y = numpy.array(labels)

Y = numpy.array(laberNormalized)

X = numpy.array(laberNormalized)

X = numpy.array(xList)

X = numpy.array(xNormalized)

alphas, coefs , _ = linear_model.lasso_path(X,Y,return_models = False)
plot.plot(alphas,coefs.T)

plot.xlabel('alpha')
plot.xlabel('Coefficients')
plot.xlabel('tight')
plot.semilogx()
ax = plot.gca()
ax.invert_xaxis()
plot.show()