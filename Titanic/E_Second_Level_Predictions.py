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
import第四个py文件中交叉验证处理后的数据
'''
import D_Generating_Base_First_Level_Models

##############################################################################################################################

'''
（1）将一级预测构建出的一组新特征（即针对训练集、测试集的预测结果y），作为训练数据训练二级分类器
'''
base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),
     'ExtraTrees': et_oof_train.ravel(),
     'AdaBoost': ada_oof_train.ravel(),
     'GradientBoost': gb_oof_train.ravel()
    })
print(base_predictions_train.head())

'''
（2）二级训练集相关热图 — Plotly图
    模型之间要 “好而不同”，彼此间相关性越低的模型越好。
'''
data5 = [
    go.Heatmap(
        z = base_predictions_train.astype(float).corr().values ,        # z即强度颜色
        x = base_predictions_train.columns.values,
        y = base_predictions_train.columns.values,
        colorscale = 'Viridis',
        showscale = True,
        reversescale = True
    )
]
layout5 = go.Layout(
    autosize = True,
    title = 'Labelled Heatmap',
)
fig5 = go.Figure(data = data5, layout = layout5)
py.plot(fig5, filename = 'labelled-heatmap')

'''
（3）拟合二级学习模型
    把五个一级模型针对训练集的预测结果y结合起来，
    把五个一级模型针对测试集的预测结果y结合起来。
    以上两个结果分别作为二次模型的训练和测试集的输入x，拟合二级学习模型。
'''
x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)

'''
用XGBoost算法拟合二级学习模型
    XGBoost参数：
        max_depth： 想要增长的树的深度。 若设置过高，可能会过拟合。 default = 6
        gamma： L1正则化项中叶结点数T前面乘的系数。 越大，算法越保守。 default = 0
        min_child_weight： L2正则化项中叶结点样本权重和（欧米伽之和）的最小值。 default = 1
        lambda： L2正则化项前的系数。 default = 1
        eta: 在每个增压步骤中使用的步骤尺寸缩小以防止过拟合，即提高鲁棒性。 default = 0.3
        subsample： 控制每棵树随机采样的比例。减小该值，算法更保守，避免过拟合。但若该值设置过小，可能会导致欠拟合。典型值：0.5-1。 default = 1
        colsample_bytree： 控制每棵树随机采样的列数的占比(每一列是一个特征)。类似于RF随机选取特征子集。 典型值：0.5-1。 default = 1
        scale_pos_weight： 在各类别样本十分不平衡时，把这个参数设定为一个正值，可以使算法更快收敛。 default = 1

        nthread： 用来进行多线程控制，应当输入系统的核数。如果你希望使用CPU全部的核，就不要输入这个参数，或者设置为-1，算法会自动检测它。类似于n_jobs

        objective： 该参数定义需要被最小化的损失函数。default = 'reg:linear'
                    最常用的值有：
                        binary:logistic： 二分类的逻辑回归，返回预测的概率(不是类别)。
                        multi:softmax： 使用softmax的多分类器，返回预测的类别(不是概率)。在这种情况下，还需多设一个参数：num_class(类别数目)。 
                        multi:softprob： 和multi:softmax参数一样，但是返回的是每个数据属于各个类别的概率。
'''
gbm = xgb.XGBClassifier(
 learning_rate = 0.02,
 n_estimators= 2000,
 max_depth = 4,
 min_child_weight = 2,
 #gamma = 1,
 gamma = 0.9,
 subsample = 0.8,
 colsample_bytree = 0.8,
 objective = 'binary:logistic',
 nthread = -1,
 scale_pos_weight = 1 ).fit(x_train, y_train)
predictions = gbm.predict(x_test)

