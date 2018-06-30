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

对于图标进行初始设置和背景设置
```
fig = plt.figure()
fig.set(alpha=0.2)
```

## 图表位置设置
图表布局有多种方式
```
plt.subplot2grid((2,3),(0,0), colspan=2)
```
即表示一块放置两行三列，以下表为第0行0列且占用两列
```
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232， sharey=ax1)
```
表示两行三列第一个

## 各种图形设置
### 柱状图
传统柱状图
```
plt = data.plot(kind='bar')
```
绘制堆积直方图
```
plt = data.plot(kind='bar', stacked=True)
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

## 图表标题和轴的注释
图表标题使用`plt.title("")`
坐标轴使用`plt.xlabel('')`
显示图例`plt.legend(('图例解释1','解释2','解释3'),loc='best')`

## 图像显示
`plt.show()`

## 显示中文
在jupyter中表格显示中文需要加上以下文字
```
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
```