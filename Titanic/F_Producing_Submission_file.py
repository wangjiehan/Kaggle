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

import E_Second_Level_Predictions

##############################################################################################################################

'''
根据已训练和fit出的所有的一级和二级模型，将预测输出到适用于Titanic比赛的格式，生成提交文件
'''
StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,
                            'Survived': predictions })
StackingSubmission.to_csv("StackingSubmission.csv", index = False)

##############################################################################################################################

'''
Summary总结：
    以上是一种非常简单的集成堆叠模型的方式。
    在高级别的Kaggle比赛中，该方法被广泛应用。
    步骤为：
    （1）选择多个“好而不同”的一级分类器，
    （2）用这些一级分类器分别交叉验证并拟合得到的预测结果y（分训练集上的和测试集上的）作为新的训练集和测试集输入x，为之后堆叠作准备
    （3）对一级分类器组合以及超过2级的堆叠。

    采取一些额外的步骤提高泛化性能：
    （1）在训练模型中实现良好的交叉验证策略，以找到最佳参数值（调参）；
    （2）引入更多种基础模型进行学习。 结果越不相关，最终得分越好。“好而不同”

比如引入LR：
from sklearn.linear_model import LogisticRegression

#LR参数设置
lr_params = {
	'C':1.0, 
	'penalty':'l1', 
	'tol':1e-6
}

注：此问题引入LR模型之前准确率为79.9%，引入后却只有77.5%
'''