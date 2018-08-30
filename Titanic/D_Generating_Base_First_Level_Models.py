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
import第一、三个py文件
'''
from A_Feature_Engineering import train,test
import C_Ensembling_and_Stacking_models as C

##############################################################################################################################

'''
sklear模块中的5个分类模型（作为一级分类）
    1. Random Forest classifier 
    2. Extra Trees classifier 分类树
    3. AdaBoost classifer 
    4. Gradient Boosting classifer 梯度提升
    5. Support Vector Machine 
'''

'''
（1）列出基学习器的参数Parameters 
    n_jobs : 用于训练过程的核心数量。 如果设置为-1，则使用所有内核。
    n_estimators : 学习模型中的分类树数（默认设置为10，default=10）。
    max_depth : 树的最大深度，或者应该扩展多少节点。 如果设置得太高，请注意，如果树太深，则会有过度拟合的风险。
    verbose : 控制是否要在学习过程中输出任何文本。 值0将禁止所有文本，而值3在每次迭代时输出树学习过程。
    min_samples_leaf : 或min_samples_split，叶子结点最少的样本数，控制叶节点上的样本数量。
    max_features : 分割节点时要考虑的特征的随机子集的大小。
                针对RF： 默认是"None"，即划分时考虑所有的特征数；
                        "log2" 划分时最多考虑log2N个特征；
                        "sqrt"或者"auto" 划分时最多考虑√N个特征。
                        整数，代表考虑的特征绝对数。
                        浮点数，代表考虑特征百分比，即考虑（百分比xN）取整后的特征数。
                其中N为样本总特征数。
    learning_rate : 针对Adaboost，即每个弱学习器的权重缩减系数ν：v*ak。
    warm_start=False ： 热启动，决定是否使用上次调用该类的结果然后增加新的。
    

所有参数以字典形式集合写出来，注意字典形式参量名要加''

这些参数可去看sklearn文档！
'''
# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025         #C 即1/lambda，描述正则项系数。
    }                   #还有个参数gamma，和sigama负相关，描述核函数宽度，即回归问题中欠拟合和过拟合平衡性



'''
（2）创建5个SklearnHelper辅助类的对象来表示5个模型
'''
#此处D文件和C文件分开时要 imort C文件里内容（C.）
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

'''
（3）将训练集、测试集和目标集转化为Numpy数组以输入模型
'''
y_train = train['Survived'].ravel()     #ravel函数（临时性操作）：对数组降维（拍平成一行），默认是行序优先（一行行走）
train = train.drop(['Survived'], axis=1)
x_train = train.values                  # 创建训练数据集数组
x_test = test.values                    # 创建测试数据集数组

'''
（4）Output of the First level Predictions
    将训练集和测试集送入模型，然后采用交叉验证方式进行预测，这些预测结果将作为二级模型的新特征。
    每个模型的两个预测结果分别为：
        针对训练数据集的预测结果 ， 针对测试数据集的预测结果
'''
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test)        # Random Forest
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test)       # Extra Trees
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test)    # AdaBoost 
gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test)        # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test)     # Support Vector Classifier

print("Training is complete")


#到此刻训练完成

'''
（5）Feature importances generated from the different classifiers
    利用Sklearn模型.featureimportances()功能，得出训练和测试集中各种特征的重要性。
'''
rf_features = rf.feature_importances(x_train,y_train)
et_features = et.feature_importances(x_train, y_train)
ada_features = ada.feature_importances(x_train, y_train)
gb_features = gb.feature_importances(x_train,y_train)
#注：SVC没有feature_importances_

rf_features = list(rf_features)
et_features = list(et_features)
ada_features = list(ada_features)
gb_features = list(gb_features)

print(rf_features)
print(et_features)
print(ada_features)
print(gb_features)

'''
（6）创建DataFrame数据框，为使用Plotly包绘制图像作准备：
    DataFrame是pandas模块里的方法，表示建立数据框，为画图做准备
    数据框以字典形式
'''
cols = train.columns.values
feature_dataframe = pd.DataFrame( {u'features': cols,
     u'Random Forest feature importances': rf_features,
     u'Extra Trees  feature importances': et_features,
     u'AdaBoost feature importances': ada_features,
     u'Gradient Boost feature importances': gb_features
    })


'''
（7）Interactive feature importances via Plotly scatterplots
    使用交互式Plotly软件包，通过调用Scatter，生成散点图来显示不同分类器的特征重要性
'''
# Scatter plot 1
trace1 = go.Scatter(
    y = feature_dataframe['Random Forest feature importances'].values,
    x = feature_dataframe['features'].values,
    mode = 'markers',
    marker = dict(
        sizemode = 'diameter',  #控制图像整体尺寸问题
        sizeref = 1,
        size = 25,              #单个散点的尺寸
#       size = feature_dataframe['Random Forest feature importances'].values,
#       color = np.random.randn(500), #set color equal to a variable
        color = feature_dataframe['Random Forest feature importances'].values,  # 设置颜色随y轴上值的大小而变化
        colorscale = 'Portland',
#       line = dict(            #每个散点加上黑线圈
#           width = 2,
#           color = 'rgb(0, 0, 0)'
#       ),
        showscale = True          # 显示图例

    ),
    text = feature_dataframe['features'].values
)
data1 = [trace1]

layout1 = go.Layout(
    autosize = True,
    title = 'Random Forest Feature Importance',
    hovermode = 'closest',
#   xaxis= dict(
#       title = 'Pop',
#       ticklen = 5,
#       zeroline = False,
#       gridwidth = 2,
#   ),
    yaxis = dict(
        title = 'Feature Importance',
        ticklen = 5,         #y轴上刻度线的长度
        gridwidth = 2        #平行于x轴的每条网格线的宽度
    ),
    showlegend = False
)
fig1 = go.Figure(data = data1, layout = layout1)
py.plot(fig1, filename = 'scatter1')

# Scatter plot 2
trace2 = go.Scatter(
    y = feature_dataframe['Extra Trees  feature importances'].values,
    x = feature_dataframe['features'].values,
    mode = 'markers',
    marker = dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
        color = feature_dataframe['Extra Trees  feature importances'].values,
        colorscale = 'Portland',
        showscale = True
    ),
    text = feature_dataframe['features'].values
)
data2 = [trace2]

layout2 = go.Layout(
    autosize = True,
    title = 'Extra Trees Feature Importance',
    hovermode = 'closest',
#   xaxis= dict(
#       title= 'Pop',
#       ticklen= 5,
#       zeroline= False,
#       gridwidth= 2,
#   ),
    yaxis=dict(
        title = 'Feature Importance',
        ticklen = 5,
        gridwidth = 2
    ),
    showlegend = False
)
fig2 = go.Figure(data = data2, layout = layout2)
py.plot(fig2, filename = 'scatter2')

# Scatter plot 3
trace3 = go.Scatter(
    y = feature_dataframe['AdaBoost feature importances'].values,
    x = feature_dataframe['features'].values,
    mode = 'markers',
    marker = dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
        color = feature_dataframe['AdaBoost feature importances'].values,
        colorscale = 'Portland',
        showscale = True
    ),
    text = feature_dataframe['features'].values
)
data3 = [trace3]

layout3 = go.Layout(
    autosize = True,
    title= 'AdaBoost Feature Importance',
    hovermode= 'closest',
#   xaxis = dict(
#       title = 'Pop',
#       ticklen = 5,
#       zeroline = False,
#       gridwidth = 2,
#   ),
    yaxis = dict(
        title = 'Feature Importance',
        ticklen = 5,
        gridwidth = 2
    ),
    showlegend = False
)
fig3 = go.Figure(data = data3, layout = layout3)
py.plot(fig3, filename = 'scatter3')

# Scatter plot 4
trace4 = go.Scatter(
    y = feature_dataframe['Gradient Boost feature importances'].values,
    x = feature_dataframe['features'].values,
    mode = 'markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
        color = feature_dataframe['Gradient Boost feature importances'].values,
        colorscale = 'Portland',
        showscale = True
    ),
    text = feature_dataframe['features'].values
)
data4 = [trace4]

layout4 = go.Layout(
    autosize = True,
    title = 'Gradient Boosting Feature Importance',
    hovermode = 'closest',
#   xaxis= dict(
#       title= 'Pop',
#       ticklen= 5,
#       zeroline= False,
#       gridwidth= 2,
#   ),
    yaxis=dict(
        title = 'Feature Importance',
        ticklen = 5,
        gridwidth = 2
    ),
    showlegend= False
)
fig4 = go.Figure(data=data4, layout=layout4)
py.plot(fig4, filename='scatter4')


'''
（8）计算所有特征重要性的平均值，并将其作为特征重要性数据框中的新列存储
'''
feature_dataframe['mean'] = feature_dataframe.mean(axis= 1)     #在字典里新添关键词'mean'的项，axis = 1 按行求均值
print(feature_dataframe.head(3))


'''
（9）绘制平均特征重要性的柱状图
'''
# Bar plot
trace5 = go.Bar(
    y = feature_dataframe['mean'].values,
    x = feature_dataframe['features'].values,
    width = 0.5,        #柱状宽度
    marker = dict(
        color = feature_dataframe['mean'].values,
        colorscale = 'Portland',
        showscale = True,
        reversescale = False    #反转图例的表示强度的颜色
    ),
    opacity = 0.6       #不透明度
)
data5 = [trace5]

layout5 = go.Layout(
    autosize = True,
    title = 'Barplots of Mean Feature Importance',
    hovermode = 'closest',
#   xaxis= dict(
#       title = 'Pop',
#       ticklen = 5,
#       zeroline = False,
#       gridwidth = 2,
#   ),
    yaxis = dict(
        title = 'Mean Feature Importance',
        ticklen = 5,
        gridwidth = 2
    ),
    showlegend = False
)
fig5 = go.Figure(data = data5, layout = layout5)
py.plot(fig5, filename ='Bar')