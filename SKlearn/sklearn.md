# RFR

# linear
## 逻辑回归
from sklearn import linear_model
```
# 训练
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X, y)
# 预测
predictions = clf.predict(test)

```
带 L1 正则的 logistic 回归,tol为停止误差,求解器默认为sag和saga基于平均随机梯度下降算法


# 数据处理

## 使用rfr补全缺失数据

```
from sklearn.ensemble import RandomForestRegressor
def set_missing_ages(df):
    # 先把特征值条件放在函数中
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    # 把缺失数据列分成已知和未知两个数据
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # 把已知数据分成代求和已知属性两个数据
    y = known_age[:, 0]
    x = known_age[:, 1:]

    # fit到rfr中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1) 
    # 决策树个数2000；并行job个数1=不并行；n：n个并行；-1：CPU有多少core，就启动多少job
    rfr.fit(x, y)

    # 用得到的模型预测未知年龄
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补缺失
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges

return df, rfr
```

## 数据归化
把数据归化到[-1, 1]之间浮动
```
import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df['Age'].values.reshape(-1, 1))
df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1, 1), age_scale_param)
```
把Age归化到（-1，1）之间，并加入到表中。其中reshape表示设置n个数为n行一列表

## 数据与结果相关度
```
pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(clf.coef_.T)})
```
coef即clf训练出的结果显示相关度，正往1，负往0

## 数据交叉验证与分割
### 交叉
```
from sklearn import cross_validation
#简单看看打分情况
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
all_data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
X = all_data.as_matrix()[:,1:]
y = all_data.as_matrix()[:,0]
print (cross_validation.cross_val_score(clf, X, y, cv=5)) #五组
```
### 分割
```
# 分割数据，按照 训练数据:cv数据 = 7:3的比例
split_train, split_cv = cross_validation.train_test_split(df, test_size=0.3, random_state=0)
train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# 生成模型
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(train_df.as_matrix()[:,1:], train_df.as_matrix()[:,0])

# 对cross validation数据进行预测

cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(cv_df.as_matrix()[:,1:])
```

## 与其他数据对比
```
origin_data_train = pd.read_csv("D:/CS/ML/Kaggle/Titanic/Train.csv")
bad_cases = origin_data_train.loc[origin_data_train['PassengerId'].isin(split_cv[predictions != cv_df.as_matrix()[:,0]]['PassengerId'].values)]
bad_cases
```