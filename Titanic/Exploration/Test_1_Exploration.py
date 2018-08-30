import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import seaborn as sns

data_train = pd.read_csv("D:/kaggle/Titanic/data/train.csv")
data_train
data_train.info()
print(data_train.describe())

import matplotlib.pyplot as plt
'''#此处引用确保中文不会变成乱码'''
plt.rcParams[ 'font.sans-serif'] = [ 'Microsoft YaHei']
plt.rcParams[ 'axes.unicode_minus'] = False

'''
缺失值画图分析
'''
fig = plt.figure()
fig.set(alpha=0.2)  # alpha参数表示图表颜色透明度，越小越透明

#Survived - Age散点分布图
plt.scatter(data_train.Survived, data_train.Age)    #scatter散点图，x、y轴分别为Survived和Age值
plt.ylabel(u"年龄")         #设定纵坐标名称
''' grid()方法设置网格线。b布尔值表示是否显示网格线。
    which取值为'major','minor'，'both'。默认为'major'。
    axis: 取值为'both'，'x'，'y'，输入的是哪条轴，则会隐藏哪条轴'''
plt.grid(b=True, which='major', axis='y')
plt.title(u"按年龄看获救分布 (1为获救)")
plt.show()

#各年龄段人数及获救情况 频数直方图
'''
hist()括号里第一个参数就是频率分布直方图横坐标数据，横坐标数据不需要value_counts()
bin代表直方图条形个数，edgecolor代表条形边界颜色，alpha代表透明度
Survived为0或1同时hist()就会放在一张图中
'''
data_train.dropna(subset=['Age'], inplace= True)        #原数据中Age有缺失值，此条语句去掉缺失值样本。若在缺失值填补后，画直方图则不需要添加此语句

plt.hist(data_train.Age[data_train.Survived == 0],bins=30,edgecolor='black')
plt.hist(data_train.Age[data_train.Survived == 1],bins=30,edgecolor='black',alpha=0.8)
plt.xlabel(u'年龄')
plt.ylabel(u'人数')
plt.title(u'各年龄段人数及获救情况')
plt.legend((u'0', u'1'),loc='best')
plt.show()

#Cabin堆积柱状图
Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
df=pd.DataFrame({u'Exist':Survived_cabin, u'Null':Survived_nocabin}).transpose()    #transpose更换横纵堆积方向
df.plot(kind='bar', stacked=True)   #设置stacked=True即可为DataFrame生成堆积柱状图，这样每行的值就会被堆积在一起
plt.title(u"按Cabin有无看获救情况")
plt.xlabel(u"Cabin有无") 
plt.ylabel(u"人数")
plt.show()

#各个Embarked的存活情况
Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'1':Survived_1, u'0':Survived_0})     #关心的是P(c|x)，即港口条件下的存活率
df.plot(kind='bar', stacked=True)
plt.title(u"各登录港口乘客的获救情况")
plt.xlabel(u"登录港口") 
plt.ylabel(u"人数") 
plt.show()

'''
DataFrame是pandas模块里的方法，表示建立数据框，为画图做准备
数据框以字典形式
'''

'''
数据清洗前的相关性分析
'''
colormap = plt.cm.viridis
plt.figure(figsize=(14,12))
sns.heatmap(data_train[data_train.columns[1:]].corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
plt.title('Pearson Correlation of Features', y=1.05, size=15)
plt.show()