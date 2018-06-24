# Pandas基本用法

## 创建数据集

### 创建数据集
直接生成一个6行*4列的随机列表，列名分别为A.B.C.D

`data = pd.DataFrame(np.random.randn(6,4), columns=list('ABCD'))`

添加行索引时间类
```
data = pd.date_range('20180621', periods)
df = pd.DataFrame(np.random.randn(6,4), index=data, columns=list('ABCD'))
```
使用字典创建数据
`df = pd.DataFrame({'A':np.random.randn(3)})`即行索引默认3行，列索引为A
```
df2 = pd.DataFrame({'A':pd.Timestamp('20170101'), 'B':np.random,randn(3)})
```
此为AB两列索引。

## 查看数据集

### 查看数据格式
`df.dtypes`

### 查看前几行数据
`df.head(5)`默认五行

### 查看后几行数据
`df.tail(5)`

### 查看数据行索引
`df.index`

### 查看数据列索引
`df.columns`

### 查看数据值
`df.values`

### 查看数据描述
`df.describe`

### 数据排序
`df.sort_values(by='列名')`

## 数据选择

### 数据切片
`df[a:b, c:d]`前行后列，使用数字
下面几种可以不使用：直接指定行列
`df.loc[a:b, c:d]`行列索引，使用行列标签
`df.iloc[a:b, c:d]`使用数字
`df.ix[a:b, c:d]`数字标签都可以使用

### 特定取某值
`df.at[dates[0], 'B']`

## 读取保存文件

### csv
读取csv文件时，可使用`pd.read_csv("path")`
存储csv文件时，使用`data.to_csv("path", encoding='utf-8', index=False)`

## 筛选数据

### 使用符号限制条件
`df[df.D>0]`筛选D列数据中大于0的数据
`df[(df.D>0)&(df.C<0)]`筛选D列中大于0且C列小于0的所有行
`df[(df.D>0)|(df.C<0)]`筛选D列中大于0或C列小于0的所有行

### 使用isin筛选特定值，把筛选的值写到另一个表中
```
goal_list = [0.232, 0.243, 0.9843]
df['D'].isin(goal_list)
```
会判断df的D列中是否有符合条件的，有返回True，没有返回false

## 增加和删除

### 增加列
例如增加一列E
`df['E'] = pd.Series(np.random.randn(6), index=df.index)`
插入任何位置(例如插入第2列名为a)
`df.insert(1,'a', np.random.randn(6))`

### 删除列
永久删除一列
`del df['a']`
返回一个删除了的数据(axis=1表示删除列)
`df1 = df.drop(['D','E'],axis=1)`

## 数据分组

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

