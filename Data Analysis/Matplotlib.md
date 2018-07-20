<!-- TOC -->

- [初始设置](#初始设置)
- [图表布局优化](#图表布局优化)
    - [整体布局](#整体布局)
    - [subplot周围间距](#subplot周围间距)
    - [坐标轴优化](#坐标轴优化)
- [刻度、标签、图例](#刻度标签图例)
    - [标签](#标签)
    - [刻度](#刻度)
    - [图例](#图例)
- [各种图形设置](#各种图形设置)
    - [线型图](#线型图)
    - [柱状图](#柱状图)
    - [直方图（一般用于一个标签内数据统计个数和，如age等）](#直方图一般用于一个标签内数据统计个数和如age等)
    - [绘制箱形图](#绘制箱形图)
    - [散点图](#散点图)
    - [密度图](#密度图)
    - [热力图](#热力图)
- [图像显示](#图像显示)
- [显示中文](#显示中文)
- [图片保存](#图片保存)

<!-- /TOC -->

## 初始设置
```
import matplotlib.pyplot as plt
import seaborn as sns
```

对于图标进行初始设置和背景设置
```
fig = plt.figure() # 设置一个画图区域
fig = plt.figure(figsize=(长， 宽)) # 可以在设置fig时直接设置画布大小
fig, ax = plt.subplots() # 使用ax画图
fig.set(alpha=0.2)
```

## 图表布局优化
### 整体布局
图表布局有多种方式
```
plt.subplots(figsize=(15, 9)) # 单纯调整画布大小
```
指定框图位置方法一
```
plt.subplot2grid((2,3),(0,0), colspan=2) # 表示一块放置两行三列，下表为第0行0列且占用两列
```
指定框图位置方法二
```
# 表示两行三列第一个
ax1 = fig.add_subplot(2,3,1)
ax2 = fig.add_subplot(2,3,2， sharey=ax1) # y轴刻度
```
### subplot周围间距
```
subplots_adjust(wspace=宽度百分比, hspace=高度百分比)
```
### 坐标轴优化
```
plt.xticks(rotation=角度) # x轴坐标注释角度
```

## 刻度、标签、图例
### 标签 
图表标题
```
plt.title(" ")
```
### 刻度 
轴的注释
```
plt.xlabel('')
```
轴的刻度范围
```
ticks = ax.set_xticks([0, 250, 500, 750, 1000]) # x刻度值
labels = ax.set_xticklabels(['one', 'two', 'three'], rotation=30, fontsize='small') # x刻度值取的名字
```

### 图例
显示图例
```
plt.legend(('图例解释1','解释2','解释3'),loc='best')
```

## 各种图形设置
### 线型图
```
data.plot()
```
其中label可用于标签，ax可设置对象，alpha可设置透明度，kind可以是line、bar、bath、kde，xticks可以设置x轴刻度值，xlim可以设置x轴界限
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
也可以在行标签有类别名时使用,如下可以实现每个one数据在一起，two数据在一起。
```
df = DataFrame(np.random.rand(6, 4), index=['one', 'two', 'three'], columns=pd.Index(['A', 'B', 'C'], name='number'))
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
```
data.plot(kind='kde')
```

### 热力图
```
sns.heatmap(data, vmin=-1, vmax=1, annot=True, square=True)
```

## 图像显示
`plt.show()`
在开头输入：`%matplotlib inline`可以不用显示命令自动显示

## 显示中文
在jupyter中表格显示中文需要加上以下文字
```
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
```

## 图片保存
```
plt.savefig('figpath.svg')
```