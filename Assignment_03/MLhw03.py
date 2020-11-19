# -*- coding: utf-8 -*-
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import datasets 
import numpy as np
import collections
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 使用加载器读取数据并且存入变量iris。
iris = datasets.load_iris()


def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()

def KNN(testX,k):# K  value change
    predY = []
    for x in testX:
        ###### 计算样本点训练集的欧氏距离 ######
        distance = [np.sqrt(np.sum(np.power(x_train - x, 2))) for x_train in trainX]

        ##### 计算曼哈顿距离 #####
        #distance = [(np.sum(x_train - x)) for x_train in trainX]


        # 从小到大排序，每个数的索引位置
        indexSort = np.argsort(distance)
        # 获得距离样本点最近的K个点的标记值y
        nearK_y = [trainY[i] for i in indexSort[:k]]
        # 统计邻近K个点标记值的数量
        cntY = collections.Counter(nearK_y)
        ###### 返回标记值最多的那个标记,majority voting scheme#####
        y_predict = cntY.most_common(1)[0][0]
        # #####返回最近的那个点的标记  nearest voting scheme#####
        #y_predict = nearK_y[0]
        predY.append(y_predict)
    return predY
#计算测试的准确率

def autoNorm(dataSet):
    minVals = dataSet.min(0)    #参数0可以从选取每一列的最小值组成向量
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals,(m,1))
    normDataSet = normDataSet/np.tile(ranges,(m,1))
    return normDataSet

def evaluateKNN():
    cnt = np.sum(preY == testY)
    acc=np.divide(cnt,len(testX))
    print("the accurate of knn:",round(acc,4))

if __name__ == '__main__':
    print(iris["feature_names"]) #返回特征值的字段名称
    print(iris.target_names)
    classes = iris.target_names
    #trainX, trainY, testX, testY=loadSplitDataSet(rate=0.8)
    rate=0.8
    iris = datasets.load_iris()
    #确定训练集和测试集的比例
    shuffleIndex=np.random.permutation(len(iris.data))
    trainSize=int(rate * len(iris.data))
    trainX= iris.data[shuffleIndex[:trainSize]]
    trainY=iris.target[shuffleIndex[:trainSize]]
    testX= iris.data[shuffleIndex[trainSize:]]
    testY = iris.target[shuffleIndex[trainSize:]]
    std = StandardScaler()

    #z score 归一化
    trainX = std.fit_transform(trainX)
    testX = std.transform(testX)

    #min max 归一化
    #trainX=autoNorm(trainX)
    #testX=autoNorm(testX)

    accs = []
    krange = range(1,30)
    #k = 0
    for k in krange:
        preY = KNN(testX, k)
        cnt = np.sum(preY == testY)
        acc = np.divide(cnt, len(testX))
        accs.append(acc)
    plt.figure(figsize=(12, 6))
    plt.plot(krange, accs, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.xlabel('K Value')
    plt.ylabel('Acc')
    plt.title('Classifier Accuracy under different K values')
    #plt.show()

    #cm=confusion_matrix(testY, preY)
    #plot_confusion_matrix(cm, 'confusion_matrix.png', title='confusion matrix')

    evaluateKNN()
    #print(preY)
    #print(testY)
    #print(trainX) #此处看归一化效果
    #print(confusion_matrix(testY, preY))
    #print(classification_report(testY, preY))

