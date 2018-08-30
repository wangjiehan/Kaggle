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

train = pd.read_csv("D:/kaggle/Titanic/data/train.csv")
test = pd.read_csv("D:/kaggle/Titanic/data/test.csv")
full_data = [train, test]

#（测试数据集里没有Survived类标记项，存在gender_submission里）

##############################################################################################################################

train.info()
test.info()

print(train.isnull().sum())
print(test.isnull().sum())



'''
存储乘客的ID号
'''
PassengerId = test['PassengerId']
print(train.head(3))        #查看前三行数据

'''
(1)加入新特征"Name_length"，并给出名字的长度
'''
train['Name_length'] = train['Name'].apply(len)     #方法apply的返回值就是()内函数对.apply前数据的返回值
test['Name_length'] = test['Name'].apply(len)


'''
（2）加入新特征是否存在Cabin
'''
train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)   #lambda为匿名函数，":"前为输入变量，":"后为返回值
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)     #NaN在python3里是float类型


'''
（3）创造新的家庭成员特征FamilySize作为SibSp和Parch的组合
'''
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1


'''
（4）根据新的特征FamilySize,创造新特征IsAlone表示是否一个人
'''
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1      # 方法data.loc[y, 'x'] = a：表示把data数据中，y条件下的'x'的值赋予新值a


'''
（5）消除登船地点缺失的数据,并用频率高的'S'代替
'''
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')       # 或者dataset.loc[dataset['Embarked'].isnull(), 'Embarked'] = 'S'
                                                                #fillna()会填充NaN数据，返回()内的填充后结果。如果希望在原DataFrame中修改，则在()内添加inplace=True


'''
（6）去除费用特征缺失数据,并以他们中位数代替,创建新的特征CategoricalFare费用类别
'''
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())    #.median()代表.之前数据的中位数
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)                    #.qcut()方法，第一个参数为要分的数据，第二个数为被分成的份数
                                                                        #cut均分值的分布间隔来选择桶的均匀间隔，qcut是均分值的频数来选择桶的均匀间隔。
print(train['CategoricalFare'])     #找划分点

'''
（6）补齐年龄缺失值，创建新的年龄分类特征CategoricalAge
'''
'''
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
'''
from sklearn.ensemble import RandomForestRegressor              # 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]     # 把已有的数值型特征取出来丢进Random Forest Regressor中
 
    known_age = age_df[age_df.Age.notnull()].as_matrix()        # 乘客分成已知年龄和未知年龄两部分
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
 
    y = known_age[:, 0]     # y即目标年龄，Age在第1列
    X = known_age[:, 1:]    # X即特征属性值

    '''
    n_estimators：基学习器决策树的个数，越多越好，但是性能就会越差，至少100左右可以达到可接受的性能和误差率。
    n_jobs：并行job个数。这个参数告诉引擎有多少处理器是它可以使用。 “-1”意味着没有限制，而“1”值意味着它只能使用一个处理器。
    '''
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)   # 将X和y用RFR算法fit后存入至rfr
    rfr.fit(X, y)
    
    predictedAges = rfr.predict(unknown_age[:, 1:])     # 用得到的模型rfr进行.predict()方法预测未知年龄结果，predict方法括号内输入的是其余特征属性X'
    df.loc[ df.Age.isnull(), 'Age' ] = predictedAges    # 用得到的预测结果填补原缺失数据
    return df

train = set_missing_ages(train)
test = set_missing_ages(test)
for dataset in full_data:
    dataset['Age'] = dataset['Age'].astype(int)         #把所有Age数据int化，否则是float格式
train['CategoricalAge'] = pd.cut(train['Age'], 5)       #cut均分值的分布间隔来选择桶的均匀间隔，均分为5段

print(train['CategoricalAge'])      #找划分点

'''
（7）定义消除乘客名字中的特殊字符，并创建新的名字特征Title,包含乘客名字主要信息
'''
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)', name)    #re.search 扫描整个字符串并返回第一个成功的匹配的位置。第一个参数为要匹配的目标，第二个参数为搜索匹配的区域
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
'''
*re.search()和group()使用方式：
    import re
    （1）
    m2 = re.search('c', 'abcdefcg')         #只匹配第一个c，返回<_sre.SRE_Match object; span=(2, 3), match='c'>
    （2）
    a = re.search('([a-z])','231422sadf')
    print(a.group(0))                       #返回2
    （3）
    a = re.search('([a-z]+)','231422sadf')
    print(a.group(0))                       #返回231422
    （4）
    a = "123abc456"
    print( re.search("([0-9]*)([a-z]*)([0-9]*)",a).group(0) )    #123abc456,返回整体
    print( re.search("([0-9]*)([a-z]*)([0-9]*)",a).group(1) )    #123，列出第一个括号匹配内容
    print( re.search("([0-9]*)([a-z]*)([0-9]*)",a).group(2) )    #abc，列出第二个括号匹配内容
    print( re.search("([0-9]*)([a-z]*)([0-9]*)",a).group(3) )    #456，列出第三个括号匹配内容。若字符串a中没有456，则group(3)就不会返回内容
'''

for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)


'''
将所有非常见的Title分组成一个单独的“稀有”组，并将Title中的非常规写法常规化
'''
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



'''
（8）对特征进行绘制，把所有特征都转化成数值型
        map(function, x) 会根据提供的函数对指定序列做映射。
        第二个参数序列中的每一个元素调用第一个参数 function函数，返回包含每次 function 函数返回值的新列表。
        def square(x) :
            return x ** 2
        print( list(map(square, [1,2,3,4,5])) )
        返回：[1, 4, 9, 16, 25]
'''
for dataset in full_data:
    '''
    对性别进行绘制
    '''
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)     #字典描述映射，类似于函数

    '''
    对Title进行绘制
    '''
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

    '''
    对登船地点绘制
    '''
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    '''
    对费用进行绘制
    '''
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare']   = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1     #不能用and
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare']  = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

    '''
    对年龄进行绘制
    '''
    dataset.loc[ dataset['Age'] <= 16, 'Age']  = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1 
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4


'''
（9）除去冗余的特征属性标签
'''
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
train = train.drop(drop_elements, axis = 1)                             #drop()方法删除，axis=1表示按列删除
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)     #axis=0代表往跨行（down)，而axis=1代表跨列（across)，作为方法动作
test  = test.drop(drop_elements, axis = 1)

'''
axis=0:
    df.mean(axis=0)：按列计算均值
    df.drop(name, axis=0)：按行删除
    （列均行删）
axis=1:
    df.mean(axis=1)：按行计算均值
    df.drop(name, axis=1)：按列删除
    （列删行均）
'''

train.info()
test.info()

