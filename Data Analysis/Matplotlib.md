<!-- TOC -->

- [初始设置](#初始设置)
- [图表位置设置](#图表位置设置)
- [各种图形设置](#各种图形设置)
    - [柱状图](#柱状图)
    - [绘制箱形图](#绘制箱形图)
    - [散点图](#散点图)
    - [密度图](#密度图)
- [图表标题和轴的注释](#图表标题和轴的注释)
- [图像显示](#图像显示)
- [显示中文](#显示中文)

<!-- /TOC -->

## 初始设置
`import matplotlib.pyplot as plt`
`import seaborn as sns`

对于图标进行初始设置和背景设置
```
fig = plt.figure()
fig.set(alpha=0.2)
```

## 图表位置设置
图表布局有多种方式
```
# 单纯调整画布大小
plt.subplots(figsize=(15, 9))
# 表示一块放置两行三列，以下表为第0行0列且占用两列
plt.subplot2grid((2,3),(0,0), colspan=2)
# 表示两行三列第一个
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232， sharey=ax1)
```


## 各种图形设置
### 柱状图
传统柱状图
```
plt = data.plot(kind='bar')
```
在同一横轴点需要两个柱状对比时，可以对上面的传统柱状图加上字典数据使其实现一个柱分上下
例如：
```
Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
df.plot(kind='bar', stacked=True)
```
如果需要两个柱子对比可以使用sns的绘图(自带统计累加)
```
sns.countplot('数据标签', hue='对比标签', data='整体数据名')
```

绘制堆积直方图
```
plt = data.plot(kind='bar', stacked=True)
```
### 直方图（一般用于一个标签内数据统计个数和，如age等）
```
g = sns.FacetGrid(data, col='', size=5) # size为纵坐标轴刻度
g.map(plt.hist, 'data中数据标签', bins=)
```

### 绘制箱形图
（一般用于显示数据的统计量如：中位数，平均数，四分位数等）
```
data.boxplot
```

### 散点图
使用x,y数据集时可以使用
```
plt.scatter(x,y)
```
使用已知整个数据集时使用
```
plt = data.plot(kind='scatter', x='X', y='Y')
```

### 密度图
`plot(kind='kde')`

### 热力图
`sns.heatmap(data, vmin=-1, vmax=1, annot=True, square=True)`

## 图表标题和轴的注释
图表标题使用`plt.title("")`
坐标轴使用`plt.xlabel('')`
显示图例`plt.legend(('图例解释1','解释2','解释3'),loc='best')`

## 图像显示
`plt.show()`
在开头输入：`%matplotlib inline`可以不用显示命令自动显示

## 显示中文
在jupyter中表格显示中文需要加上以下文字
```
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
```