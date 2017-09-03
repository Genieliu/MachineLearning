
# coding: utf-8

# In[1]:

from numpy import *
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0,0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), 
                             key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    arrayOflines = fr.readlines()
    numberOfLines = len(arrayOflines)
    returnMat = zeros((numberOfLines,4))
    classLabelVector = []
    index = 0
    for line in arrayOflines:
        line = line.strip()
        listFromLine = line.split(',')
        returnMat[index,:] = listFromLine[0:4]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals-minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet-tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('iris.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],\
                                    datingLabels[numTestVecs:m],3)
        print ("the classifier came back with: %d, the real answer is: %d"\
                %(classifierResult, datingLabels[i]))
        if(classifierResult != datingLabels[i]):
            errorCount += 1
    print("the total error rate is: %f" %(errorCount/float(numTestVecs)))

def classifyIrsi():
    resultList = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

    sepal_len = float(input("sepal_len:"))
    sepal_wid = float(input("sepal_wid:"))
    petal_len = float(input("petal_len:"))
    petal_wid = float(input("petal_wid:"))

    irisMat, irisLabels = file2matrix("iris.txt")
    normMat, ranges, minVals = autoNorm(irisMat)
    inArr = array([sepal_len, sepal_wid, petal_len, petal_wid])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, irisLabels, 3)
    print("You will probably see the Iris: " , resultList[classifierResult-1])




#test
if __name__ == "__main__":
    group, labels = kNN.createDataSet()
    kNN.classify0([1.5,0], group, labels,3)
