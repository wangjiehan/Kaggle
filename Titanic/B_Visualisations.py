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
import第一个py文件中特征工程处理后的train
'''
from A_Feature_Engineering import train

##############################################################################################################################

'''
（1）观察进行过特征清洗,筛选过的新特征数据
'''
print(train.head(3))

'''
（2）Pearson Correlation Heatmap(皮尔森相关热图)：
    生成特征的相关图，看看一个特征和另一个特征的相关程度，.corr()方法。
    利用Seaborn绘图软件包的.heatmap()方法，非常方便地绘制皮尔森相关热图
'''
colormap = plt.cm.viridis
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
#.heatmap()为画热图方法，.astype(float)把所有数据转换成float类型，.corr()即为求相关性的方法
# linewidths，linecolor划分线的宽度和颜色，annot是否在方格里注释数据，vmin, vmax图例相关度最大值和最小值

plt.show()

'''
（3）生成一些配对图来观察一个特征和另一个特征的数据分布
'''
g = sns.pairplot(train[[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare', u'Embarked',      #一行写不下时，直接换行并缩进两个tab（或者使用\）
                        u'FamilySize', u'Title']], hue='Survived', palette = 'seismic',size=1.2,
                        diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )
g.set(xticklabels=[])
'''
sns.pairplot()方法：
    kind : {‘scatter’, ‘reg’}      scatter散点图，reg回归图
    diag_kind : {‘hist’, ‘kde’}    hist柱状图，kde密度图

    hue : 使用指定变量为分类变量画图。参数类型：string (变量名)。即用一个特征来显示图像上的颜色，类似于打标签。
    palette : 调色板颜色
    size : 默认 6，图的尺度大小（正方形）参数类型：numeric
    markers : 使用不同的形状。参数类型：list
    {plot, diag, grid}_kws : 指定其他参数。参数类型：dicts
'''

plt.show()