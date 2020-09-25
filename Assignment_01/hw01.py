#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from random import sample


plt.close('all')  # close any open plots

# In[2]:


""" ======================  Function definitions ========================== """


def plotData(x1, t1, x2=None, t2=None, x3=None, t3=None, legend=[]):
    '''plotData(x1,t1,x2,t2,x3=None,t3=None,legend=[]): Generate a plot of the
       training data, the true function, and the estimated function'''
    p1 = plt.plot(x1, t1, 'bo')  # plot training data
    if (x2 is not None):
        p2 = plt.plot(x2, t2, 'g')  # plot true value
    if (x3 is not None):
        p3 = plt.plot(x3, t3, 'r')  # plot training data

    # add title, legend and axes labels
    plt.ylabel('t')  # label x and y axes
    plt.xlabel('x')

    if (x2 is None):
        plt.legend((p1[0]), legend)
    if (x3 is None):
        plt.legend((p1[0], p2[0]), legend)
    else:
        plt.legend((p1[0], p2[0], p3[0]), legend)


def fitdata(x, t, M):
    '''fitdata(x,t,M): Fit a polynomial of order M to the data (x,t)'''
    # This needs to be filled in

    X = np.array([x ** m for m in range(M + 1)]).T
    w = np.linalg.inv(X.T @ X) @ X.T @ t
    return w


def fitdataRBF1(x, t, M, s):
    up = max(x)
    low = min(x)
    interval = (up - low) / M

    miu = np.arange(low + interval / M, up, interval)
    X = np.array([np.exp((x - miu[i]) ** 2 / (-2 * s ** 2)) for i in range(miu.size)]).T
    w = np.linalg.inv(X.T @ X) @ X.T @ t
    return w, miu


def fitdataRBF2(x, t, M, s):
    miu = sample(list(x), M)
    X = np.array([np.exp((x - miu[i]) ** 2 / (-2 * s ** 2)) for i in range(M)]).T
    w = np.linalg.inv(X.T @ X) @ X.T @ t
    return w, miu


# In[3]:


""" =======================  Load Training Data ======================= """
data_uniform = np.load('TrainData.npy')
x1 = data_uniform[:, 0]
t1 = data_uniform[:, 1]
plt.title('training data')
plt.plot(x1, t1, 'bo')  # plot training data >>>>>>>>>>1

# In[4]:


l = -4
u = 4
x3 = np.arange(l, u, 0.001)  # get equally spaced points in the xrange
t3 = np.sinc(x3)  # compute the true function value

fig = plt.figure()
plt.title('true function')
plotData(x1, t1, x3, t3, legend=['Training Data', 'True Function'])  ### plot true function
plt.xlim(-4.5, 4.5)
plt.ylim(-2, 2)

# In[5]:


""" ========================  Train the Model ============================= """
"""This is where you call functions to train your model with different RBF kernels   """

M = 7
s = 1
w = fitdata(x1, t1, M)
xrange = np.arange(-4, 4, 0.001)  # get equally spaced points in the xrange
X = np.array([xrange ** m for m in range(w.size)]).T
esty = X @ w  # compute the predicted value
plt.title('curve with polynomial')
plotData(x1, t1, x3, t3, xrange, esty, legend=['Training Data', 'True Function', 'Estimated\nPolynomial'])
plt.xlim(-4.5, 4.5)
plt.ylim(-2, 2)

# In[6]:


w_rbf1, miu1 = fitdataRBF1(x1, t1, M, s)
xrange = np.arange(-4, 4, 0.001)  # get equally spaced points in the xrange
X1 = np.array([np.exp((xrange - miu1[i]) ** 2 / (-2 * s ** 2)) for i in range(w_rbf1.size)]).T
esty_rbf1 = X1 @ w_rbf1  # compute the predicted value
plt.title('curve with rbf1')
plotData(x1, t1, xrange, esty_rbf1, legend=['Training Data', 'Estimated\RBF1'])  # >>>>>>>>>>>>>>>2
plt.xlim(-4.5, 4.5)
plt.ylim(-2, 2)

# In[7]:


w_rbf2, miu2 = fitdataRBF2(x1, t1, M, s)
xrange = np.arange(-4, 4, 0.001)  # get equally spaced points in the xrange

X2 = np.array([np.exp((xrange - miu2[i]) ** 2 / (-2 * s ** 2)) for i in range(w_rbf2.size)]).T
esty_rbf2 = X2 @ w_rbf2  # compute the predicted value
plt.title('curve with rbf2')
plotData(x1, t1, x3, t3, xrange, esty_rbf2,
         legend=['Training Data', 'True Function', 'Estimated\RBF2'])  # >>>>>>>>>>>>>>>2
plt.xlim(-4.5, 4.5)
plt.ylim(-2, 2)

# In[8]:


""" ======================== Load Test Data  and Test the Model =========================== """
"""This is where you should load the testing data set. You shoud NOT re-train the model   """

data2_uniform = np.load('TestData.npy')
x2 = data2_uniform[:, 0]
t2 = data2_uniform[:, 1]
# plt.plot(x2, t2, 'bo') #plot test data


# In[9]:


s = 1
Mrange = np.arange(2, 20, 1)
error_rbf1 = Mrange.copy()
error_rbf2 = Mrange.copy()
for M in Mrange:
    w_rbf1, miu1 = fitdataRBF1(x1, t1, M, s)
    w_rbf2, miu2 = fitdataRBF2(x1, t1, M, s)
    X2_rbf1 = np.array([np.exp((x2 - miu1[i]) ** 2 / (-2 * s ** 2)) for i in range(w_rbf1.size)]).T
    esty_rbf1_test = X2_rbf1 @ w_rbf1
    error1 = abs(esty_rbf1_test - t2)
    error_rbf1[M - 2] = sum(error1)
    X2_rbf2 = np.array([np.exp((x2 - miu2[i]) ** 2 / (-2 * s ** 2)) for i in range(w_rbf2.size)]).T
    esty_rbf2_test = X2_rbf2 @ w_rbf2
    error2 = abs(esty_rbf2_test - t2)
    error_rbf2[M - 2] = sum(error2)

plt.plot(Mrange, error_rbf1, 'r')
plt.plot(Mrange, error_rbf2, 'b')
plt.xlim(0, 21)
plt.ylim(0, 1000)
plt.ylabel('error')
plt.xlabel('M')

# In[10]:


sum(error1)

# In[11]:


sum(error1)

# In[ ]:


# !/usr/bin/env python
# coding: utf-8




import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from random import sample

plt.close('all')  # close any open plots




""" ======================  Function definitions ========================== """


def plotData(x1, t1, x2=None, t2=None, x3=None, t3=None, legend=[]):
    '''plotData(x1,t1,x2,t2,x3=None,t3=None,legend=[]): Generate a plot of the
       training data, the true function, and the estimated function'''
    p1 = plt.plot(x1, t1, 'bo')  # plot training data
    if (x2 is not None):
        p2 = plt.plot(x2, t2, 'g')  # plot true value
    if (x3 is not None):
        p3 = plt.plot(x3, t3, 'r')  # plot training data

    # add title, legend and axes labels
    plt.ylabel('t')  # label x and y axes
    plt.xlabel('x')

    if (x2 is None):
        plt.legend((p1[0]), legend)
    if (x3 is None):
        plt.legend((p1[0], p2[0]), legend)
    else:
        plt.legend((p1[0], p2[0], p3[0]), legend)


def fitdata(x, t, M):
    '''fitdata(x,t,M): Fit a polynomial of order M to the data (x,t)'''
    # This needs to be filled in

    X = np.array([x ** m for m in range(M + 1)]).T
    w = np.linalg.inv(X.T @ X) @ X.T @ t
    return w


def fitdataRBF1(x, t, M, s):
    up = max(x)
    low = min(x)
    interval = (up - low) / M

    miu = np.arange(low + interval / M, up, interval)
    X = np.array([np.exp((x - miu[i]) ** 2 / (-2 * s ** 2)) for i in range(miu.size)]).T
    w = np.linalg.inv(X.T @ X) @ X.T @ t
    return w, miu


def fitdataRBF2(x, t, M, s):
    miu = sample(list(x), M)
    X = np.array([np.exp((x - miu[i]) ** 2 / (-2 * s ** 2)) for i in range(M)]).T
    w = np.linalg.inv(X.T @ X) @ X.T @ t
    return w, miu





""" =======================  Load Training Data ======================= """
data_uniform = np.load('TrainData.npy')
x1 = data_uniform[:, 0]
t1 = data_uniform[:, 1]
plt.title('training data')
plt.plot(x1, t1, 'bo')  # plot training data >>>>>>>>>>1
plt.show()




l = -4
u = 4
x3 = np.arange(l, u, 0.001)  # get equally spaced points in the xrange
t3 = np.sinc(x3)  # compute the true function value

fig = plt.figure()
plt.title('true function')
plotData(x1, t1, x3, t3, legend=['Training Data', 'True Function'])  ### plot true function
plt.xlim(-4.5, 4.5)
plt.ylim(-2, 2)
plt.show()




""" ========================  Train the Model ============================= """
"""This is where you call functions to train your model with different RBF kernels   """

M = 7
s = 1
w = fitdata(x1, t1, M)
xrange = np.arange(-4, 4, 0.001)  # get equally spaced points in the xrange
X = np.array([xrange ** m for m in range(w.size)]).T
esty = X @ w  # compute the predicted value
plt.title('curve with polynomial')
plotData(x1, t1, x3, t3, xrange, esty, legend=['Training Data', 'True Function', 'Estimated\nPolynomial'])
plt.xlim(-4.5, 4.5)
plt.ylim(-2, 2)
plt.show()




w_rbf1, miu1 = fitdataRBF1(x1, t1, M, s)
xrange = np.arange(-4, 4, 0.001)  # get equally spaced points in the xrange
X1 = np.array([np.exp((xrange - miu1[i]) ** 2 / (-2 * s ** 2)) for i in range(w_rbf1.size)]).T
esty_rbf1 = X1 @ w_rbf1  # compute the predicted value
plt.title('curve with rbf1')
plotData(x1, t1, xrange, esty_rbf1, legend=['Training Data', 'Estimated\RBF1'])  # >>>>>>>>>>>>>>>2
plt.xlim(-4.5, 4.5)
plt.ylim(-2, 2)
plt.show()




w_rbf2, miu2 = fitdataRBF2(x1, t1, M, s)
xrange = np.arange(-4, 4, 0.001)  # get equally spaced points in the xrange

X2 = np.array([np.exp((xrange - miu2[i]) ** 2 / (-2 * s ** 2)) for i in range(w_rbf2.size)]).T
esty_rbf2 = X2 @ w_rbf2  # compute the predicted value
plt.title('curve with rbf2')
plotData(x1, t1, x3, t3, xrange, esty_rbf2,
         legend=['Training Data', 'True Function', 'Estimated\RBF2'])  # >>>>>>>>>>>>>>>2
plt.xlim(-4.5, 4.5)
plt.ylim(-2, 2)
plt.show()




""" ======================== Load Test Data  and Test the Model =========================== """
"""This is where you should load the testing data set. You shoud NOT re-train the model   """

data2_uniform = np.load('TestData.npy')
x2 = data2_uniform[:, 0]
t2 = data2_uniform[:, 1]
# plt.plot(x2, t2, 'bo') #plot test data





s = 1
Mrange = np.arange(2, 20, 1)
error_rbf1 = Mrange.copy()
error_rbf2 = Mrange.copy()
for M in Mrange:
    w_rbf1, miu1 = fitdataRBF1(x1, t1, M, s)
    w_rbf2, miu2 = fitdataRBF2(x1, t1, M, s)
    X2_rbf1 = np.array([np.exp((x2 - miu1[i]) ** 2 / (-2 * s ** 2)) for i in range(w_rbf1.size)]).T
    esty_rbf1_test = X2_rbf1 @ w_rbf1
    error1 = abs(esty_rbf1_test - t2)
    error_rbf1[M - 2] = sum(error1)
    X2_rbf2 = np.array([np.exp((x2 - miu2[i]) ** 2 / (-2 * s ** 2)) for i in range(w_rbf2.size)]).T
    esty_rbf2_test = X2_rbf2 @ w_rbf2
    error2 = abs(esty_rbf2_test - t2)
    error_rbf2[M - 2] = sum(error2)

plt.plot(Mrange, error_rbf1, 'r')
plt.plot(Mrange, error_rbf2, 'b')
plt.title('red:rbf1 and blue:rbf2 with error')
plt.xlim(0, 21)
plt.ylim(0, 1000)
plt.ylabel('error')
plt.xlabel('M')
plt.show()




