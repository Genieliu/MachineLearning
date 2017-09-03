#!/usr/bin/python

# 
from os import listdir
from numpy import *
import operator

def img2vector(filename):
    """
        transfer img file to a vector
    """
    returnVect = zeros([1,1024])
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

# This is main kNN do, input: testData, data, labels and K
def classify0(inX, dataSet, labels, k):
    """
        main kNN program
    """
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

def handWritingClassTest():
    """
        test the hand writing program 
    """
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNamestr = trainingFileList[i]
        fileStr = fileNamestr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % (fileNamestr))
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNamestr = testFileList[i]
        fileStr = fileNamestr.split('.')[0]
        classNumStr = int(fileNamestr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNamestr)
        classifierResult = classify0(vectorUnderTest, \
                trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" \
                % (classifierResult, classNumStr))
        if(classifierResult != classNumStr):
            errorCount += 1.0
    print("\nthe total number of error is: %d" %(errorCount))
    print("\nthe total error rate of error is: %f" %(errorCount/float(mTest)))



if __name__ == "__main__":

    handWritingClassTest()
