'''
加载调用数据库
'''
import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings('ignore')

'''
#得到5个基本模型作为stacking进行预测
'''
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import KFold

'''
import第一个py文件中特征工程处理后的train、test
'''
from A_Feature_Engineering import train,test

##############################################################################################################################

'''
创建集合和堆叠模型
    一、构建一个SklearnHelper类，允许扩展所有Sklearn分类器的共同的内置方法（如训练，预测和拟合）。
    二、定义一个get_oof函数。
    此后，如果要调用五个不同的分类器，将减少冗余，不需要多次编写相同的代码。
'''

'''
（1）准备参数
'''
ntrain = train.shape[0]     #shape函数查看矩阵维数，直接print(.shape)即可得到行列尺寸。shape[0]为行数，shape[1]为列数。
ntest = test.shape[0]
SEED = 0                    # 随机数生成种子，为了可重复性
NFOLDS = 5                  # 设置k折交叉验证的k值
kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)      #在训练样本train中使用交叉验证，最终会采用所有NFOLDS行的均值
'''
KFold()方法：
    KFold(n, n_folds=3, shuffle=False, random_state=None)
    第一个参数为数据总量尺寸，第二个参数为k，shuffle表示在划分前要不要打乱数据，
    random_state：随机数生成种子。若设置，保证每次随机生成的数是一致的（即使是随机的）；若不设置，即默认None，则每次生成的随机数都不同。

注意：其实交叉验证的是 0到ntrain-1 的标记，通过标记标定出交叉验证出来的训练子集和测试子集。


例：
from sklearn.cross_validation import KFold

kf = KFold(12,n_folds=5,shuffle=False)
for i,(train_index,test_index) in enumerate(kf):
    print(i,train_index,test_index)

输出为：
0 [ 3  4  5  6  7  8  9 10 11] [0 1 2]
1 [ 0  1  2  6  7  8  9 10 11] [3 4 5]
2 [ 0  1  2  3  4  5  8  9 10 11] [6 7]
3 [ 0  1  2  3  4  5  6  7 10 11] [8 9]
4 [0 1 2 3 4 5 6 7 8 9] [10 11]
'''


'''
（2）定义一个辅助类来扩展Sklearn分类器
    类的简单方法，它简单地调用sklearn分类器中已经存在的相应方法。 
    本质上创建了一个包装类来扩展各种Sklearn分类器，这样可以帮助我们在实现到堆栈器时，减少编写相同的代码。
'''
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):   #clf：输入的sklearn分类器， seed：随机种子， params：分类器的参数
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):  #fit()是调用的通用方法。fit(X)表示用数据X来训练某种模型。函数返回值一般为调用fit方法的对象本身。
        self.clf.fit(x_train, y_train)  #fit(X,y=None)为无监督学习算法，fit(X,y)为监督学习算法

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self,x,y):
        return self.clf.fit(x,y)

    def feature_importances(self,x,y):
        return self.clf.fit(x,y).feature_importances_       #RF包里的写法.feature_importances_


'''
（3）定义get_oof函数
    Out-of-Fold Predictions预测函数，用来生成第一级预测结果。

    之后堆叠使用基础分类器的预测作为训练到二级模型的输入。
    然而，不能简单地对完整的训练数据进行基本模型的训练，在完整的测试集上产生预测，然后输出这些用于二级训练。
    因为基本模型预测很有可能已经具有“看到”测试集的风险，并因此在馈送这些预测时过度拟合。
'''
def get_oof(clf_obj, x_train, y_train, x_test):     # clf_obj：各个clf分类器根据上一步SklearnHelper()类定义出来的对象
    #先申请3个空数组
    oof_train = np.zeros((ntrain,))         #zeros创建零矩阵。()内参数代表尺寸。逗号前代表行数，逗号后代表列数，如果没有列数，则创建的是一维数组
    oof_test = np.zeros((ntest,))           #zeros默认元素为float型，zeros((5,3),int) 则变成整型。注意zeros后两个括号：尺寸的括号可以()也可以[]
    oof_test_skf = np.empty((NFOLDS, ntest))    #empty创建的数组，元素表示无意义的数值

    '''注意：其实交叉验证的是 0到ntrain-1 的标记，通过标记标定出交叉验证出来的训练子集和测试子集'''
    for i, (train_index, test_index) in enumerate(kf):  
        x_tr = x_train[train_index]         #把交叉验证中第i组获得的训练集在原训练数据集train的输入集x_train中找出来，定义为x_tr
        y_tr = y_train[train_index]         #把交叉验证中第i组获得的训练集在原训练数据集train的输出集y_train中找出来，定义为y_tr
        x_te = x_train[test_index]          #把交叉验证中第i组获得的测试集在原训练数据集train的输入集x_train中找出来，定义为x_te

        clf_obj.train(x_tr, y_tr)               #其实就是 fit(x_tr, y_tr)
        
        '''
        注意：predict()括号内的是要输入的x，预测出来的结果即为y。
        此处y被存进数组oof_train的对应位置[test_index]上。
        '''
        oof_train[test_index] = clf_obj.predict(x_te)   #最终循环结束oof_train矩阵就是交叉验证中通过各组训练子集fit出来的的模型，分别针对各组测试子集预测出来并连接起来的所有结果。即为对全部训练数据集的预测。
        oof_test_skf[i, :] = clf_obj.predict(x_test)    #分别保存 用每一组交叉验证fit出来的模型去预测真正的测试数据集而得到的结果

    oof_test[:] = oof_test_skf.mean(axis=0)         #axis=0为按列计算均值
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
    '''
    reshape方法:
        把矩阵内所有元素全部按新的尺寸依次排布，两个参数分别代表行列。
        若其中一个参数是-1，则代表其值依据另一个参数确定而确定。
        比如总共有10个元素，一个参数为-1，另一个为5，则为-1的那个参数实际表现值为2
        若只有一个参数，且为-1，则数组为一行
        reshape方法是暂时性的排布
    '''

