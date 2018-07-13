# Numpy的基本用法
## 创建ndarray 
创建最简单的方法就是array函数
```
data1 = [6, 7.5, 5, 0, 1]
```
### 零一矩阵
 `zeros((x, y))`和`ones((x, y))`分别可以创造指定长度形状的全0，1数组

### 等差数组 
```
arrange（a）# 创建一组从0到a-1的数组
```

## 等差矩阵
```
np.linspace(startNumber, 等差数, 个数)
```

### 随机矩阵
对数组进行打乱可以使用
```
np.random.random((x, y)) # 创造一个随机x行y列的矩阵
random.shuffle(x, y) # 原数组上进行打乱
random.permutation(x, y) # 返回一个新数组

```
## 矩阵属性
### 行列查看
```
data.shape()
```
### 矩阵数据格式
```
data.dtype
```

## 矩阵变换
### 行列设置
```
reshape(x, y)
```
### 切片和索引
基本切片行列
```
arr[x: y]
```


## 数学表达式

### 协方差
`corr()`

## 读取文件
### 读取TXT文件
```
data = np.genfromtxt('path', delimiter='分割标志', dtype=str)
```