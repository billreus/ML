# Pandas基本用法

## 读取、存储文件
读取csv文件时，可使用`pd.read_csv("path")`
存储csv文件时，使用`data.to_csv("path")`

## 表格信息
每行列数据信息统计可以使用`info()`
```
data.info()
```
每列数据的总数、平均数等统计信息可使用`describe()`
```
data.describe()
```

## 数据叠加
一列需要统计总数累加时可以使用`data.value_counts()`
```
data.survived.value_counts()
```
即表示data数据的survived列的数据和

## onehotb编码
```
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')

dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')

dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')

dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')

df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1) # 数组拼接
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df
```
对`Cabin,Embarked,Sex,Pclass`进行onehot编码且编码名称取本身数据名用0,1表示，再进行拼接和删除

