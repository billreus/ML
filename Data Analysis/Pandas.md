<!-- TOC -->

- [Pandas基本用法](#pandas基本用法)
    - [创建数据集](#创建数据集)
        - [创建数据集](#创建数据集-1)
    - [查看数据集](#查看数据集)
        - [查看数据格式](#查看数据格式)
        - [查看前几行数据](#查看前几行数据)
        - [查看后几行数据](#查看后几行数据)
        - [查看数据行索引](#查看数据行索引)
        - [查看数据列索引](#查看数据列索引)
        - [查看数据值](#查看数据值)
        - [查看数据描述](#查看数据描述)
        - [数据排序](#数据排序)
    - [数据选择](#数据选择)
        - [数据切片](#数据切片)
        - [特定取某值](#特定取某值)
    - [读取保存文件](#读取保存文件)
        - [csv](#csv)
    - [筛选数据](#筛选数据)
        - [使用符号限制条件](#使用符号限制条件)
        - [使用isin筛选特定值，把筛选的值写到另一个表中](#使用isin筛选特定值把筛选的值写到另一个表中)
    - [增加和删除](#增加和删除)
        - [增加列](#增加列)
        - [删除列](#删除列)
    - [数据分组](#数据分组)
        - [利用groupby（）进行分组](#利用groupby进行分组)
        - [aggregate()实现数据分组计算](#aggregate实现数据分组计算)
        - [size()查看各组数据量](#size查看各组数据量)
    - [处理缺失值](#处理缺失值)
        - [用固定值代替](#用固定值代替)
        - [用统计值代替](#用统计值代替)
        - [用插值法填补缺失值](#用插值法填补缺失值)
        - [删除缺失值](#删除缺失值)
    - [排序与合并](#排序与合并)
        - [排序sort_value()](#排序sort_value)
        - [合并merge()](#合并merge)
        - [合并concat()](#合并concat)
    - [表格信息](#表格信息)
    - [数据叠加](#数据叠加)
    - [onehotb编码](#onehotb编码)

<!-- /TOC -->

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
```
df = pd.DataFrame({'A':np.random.randn(3)}) # 即行索引默认3行，列索引为A
df2 = pd.DataFrame({'A':pd.Timestamp('20170101'), 'B':np.random,randn(3)}) # 此为AB两列索引
```

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
前行后列，使用数字
```
df[a:b, c:d]
```
下面几种可以不使用：直接指定行列
```
df.loc[a:b, c:d] # 行列索引，使用行列标签
df.iloc[a:b, c:d] # 使用数字
df.ix[a:b, c:d] # 数字标签都可以使用
```

### 特定取某值
`df.at[dates[0], 'B']`

## 读取保存文件

### csv
读取csv文件
```
pd.read_csv("path")
```
存储csv文件
```
data.to_csv("path", encoding='utf-8', index=False)
```

## 筛选数据

### 使用符号限制条件
```
df[df.D>0] # 筛选D列数据中大于0的数据
df[(df.D>0)&(df.C<0)] # 筛选D列中大于0且C列小于0的所有行
df[(df.D>0)|(df.C<0)] # 筛选D列中大于0或C列小于0的所有行
```

### 使用isin筛选特定值，把筛选的值写到另一个表中
```
goal_list = [0.232, 0.243, 0.9843]
df['D'].isin(goal_list)
```
会判断df的D列中是否有符合条件的，有返回True，没有返回false

## 增加和删除

### 增加列
例如增加一列E
```
df['E'] = pd.Series(np.random.randn(6), index=df.index)
```
插入任何位置(例如插入第2列名为a)
```
df.insert(1,'a', np.random.randn(6))
```

### 删除列
永久删除一列
```
del df['a']
```
返回一个删除了的数据(axis=1表示删除列)
```
df1 = df.drop(['D','E'],axis=1)
```

## 数据分组

### 利用groupby（）进行分组
即以纵索引的形式把一个列相同的放在一个大行内
```
gruop_ID = df.groupy('ID')
```

以两列以上进行分组
```
gruop_id = df.groupby(['ID','bill'])
```

### aggregate()实现数据分组计算
在同一组内计算可以使用该函数实现
```
df.groupby(['ID', 'bill']).aggregate(np.sum)
```

### size()查看各组数据量
```
df.size() # 即可看到每个分组内的数据个数
```

## 处理缺失值

### 用固定值代替
```
df.fillna(0) # 用0代替缺失值
df.fillna('missing') # 用一个字符串代替缺失值
df.fillna(method='pad') # 用前一个数据代替
df.fillna(method='bfill') # 用后一个数据代替
df.fillna(method='pad', limit=1) # 用limit限制每列代替NaN的数目
```

### 用统计值代替
用平均数代替
```
df.fillna(df.mean())
```
可以选择哪一列进行缺值处理
```
df.fillna(df.mean()['one':'two'])
```

###用插值法填补缺失值
插值法即通过两点间估计中间点的值
```
df.interpolate()
```

### 删除缺失值
`df.dropna(axis=0,1)`

## 排序与合并

### 排序sort_value()
按照指定标签排序
```
df2 = df.sort_value(by='ID')
```

### 合并merge()
将两个表进行合并可使用该函数
例如将over和user两个表按ID进行合并，当ID不同时按右边为基准
```
data = pd.merge(over, user, how='right', on='ID')
```

### 合并concat()
直接拼接，默认列索引拼接缺失部分用NaN填充,行索引使用axis=1
```
pd.concat([df1,df2])
```
拼接后行索引忽略之前重新生成
```
pd.concat([df1,df2],ignore_index=True)
```

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

