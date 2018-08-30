import pandas as pd
import numpy as np
from pandas import Series, DataFrame

data_train = pd.read_csv("D:/kaggle/Titanic/data/train.csv")

data_train.info()
print(data_train.isnull().sum())


import matplotlib.pyplot as plt
'''#此处引用确保中文不会变成乱码'''
plt.rcParams[ 'font.sans-serif'] = [ 'Microsoft YaHei']
plt.rcParams[ 'axes.unicode_minus'] = False


'''
缺失值填补
'''
from sklearn.ensemble import RandomForestRegressor
 
### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):
 
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
 
    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
 
    # y即目标年龄
    y = known_age[:, 0]
 
    # X即特征属性值
    X = known_age[:, 1:]
 
    # fit到RandomForestRegressor之中
    '''
    n_estimators：基学习器决策树的个数，越多越好，但是性能就会越差，至少100左右可以达到可接受的性能和误差率。
    n_jobs：并行job个数。这个参数告诉引擎有多少处理器是它可以使用。 “-1”意味着没有限制，而“1”值意味着它只能使用一个处理器。
    '''
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)
 
    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1:])
 
    # 用得到的预测结果填补原缺失数据
    df.loc[ df.Age.isnull(), 'Age' ] = predictedAges 
 
    return df, rfr
 
def set_Cabin_type(df):
    df.loc[ df.Cabin.notnull(), 'Cabin' ] = "Yes"
    df.loc[ df.Cabin.isnull(), 'Cabin' ] = "No"
    return df

def set_missing_Embarked(df):
    df.loc[ df.Embarked.isnull(), 'Embarked' ] = 'S'
    return df
 
data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)
data_train = set_missing_Embarked(data_train)

data_train.info()

'''
探索性可视化
'''
fig = plt.figure()
fig.set(alpha=0.2)  # alpha参数表示图表颜色透明度，越小越透明

'''subplot2grid()方法：在一张大图里分列几个小图：(2,3)表示大图总尺寸为两行三列，(0,0)表示当前图的起始下标''' 
#（1）Survived人数
plt.subplot2grid((2,2),(0,0))
data_train.Survived.value_counts().plot(kind='bar') # kind = bar 柱状图 
plt.title(u"获救情况 (1为获救)") # 标题
plt.ylabel(u"人数")  
 
#（2）Pclass人数
plt.subplot2grid((2,2),(0,1))
data_train.Pclass.value_counts().plot(kind="bar")
plt.title(u"乘客等级分布")
plt.ylabel(u"人数")

#（3）不同Pclass的Age密度分布图
plt.subplot2grid((2,2),(1,0), colspan=2)                  #在大图中从(1,0)位置开始，占两个列宽
data_train.Age[data_train.Pclass == 1].plot(kind='kde')   # kde 密度图
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel(u"年龄")# plots an axis lable
plt.ylabel(u"密度") 
plt.title(u"各等级的乘客年龄分布")
plt.legend((u'Pclass 1', u'Pclass 2',u'Pclass 3'),loc='best')   # 图例写法！！

plt.show()


#不同Pclass获救情况
Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'0':Survived_0, u'1':Survived_1})         #此处只能两段数据堆积柱状图，若要三段及以上需transpose()方法
df.plot(kind='bar', stacked=True)
plt.title(u"船舱等级获救情况")
plt.xlabel(u"Pclass")
plt.ylabel(u"人数")
plt.show()

#不同Sex获救情况
Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df=pd.DataFrame({u'male':Survived_m, u'female':Survived_f}).transpose()
df.plot(kind='bar', stacked=True)
plt.title(u"不同性别获救情况")
plt.xlabel(u"性别")
plt.ylabel(u"人数")
plt.show()

'''画子图：pyplot的方式中plt.subplot()参数和面向对象中的add_subplot()参数和含义都相同'''
fig=plt.figure()
fig.set(alpha=0.65)
plt.title(u"根据舱等级和性别的获救情况")
 
ax1=fig.add_subplot(141)        #141代表：总共1行、4列，当前子图处于第1个位置，之后的子图位置标记按每一行逐列再逐行的顺序往下排布
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar', label="female, high class", color='#FA2479')
ax1.set_xticklabels([u"1", u"0"], rotation=0)		#1和0的顺序：Pclass是1,2的female从上到下先遇见Survived是1
ax1.legend([u"female/Pclass = 1,2"], loc='best')
 
ax2=fig.add_subplot(142, sharey=ax1)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')
ax2.set_xticklabels([u"1", u"0"], rotation=0)
plt.legend([u"female/Pclass = 3"], loc='best')
 
ax3=fig.add_subplot(143, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')
ax3.set_xticklabels([u"0", u"1"], rotation=0)
plt.legend([u"male/Pclass = 1,2"], loc='best')
 
ax4=fig.add_subplot(144, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male, low class', color='steelblue')
ax4.set_xticklabels([u"0", u"1"], rotation=0)
plt.legend([u"male/Pclass = 3"], loc='best')

plt.show()
