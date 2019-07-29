# 数据分析
## numpy
### 基础
#### 创建简单数组
函数|作用
|-|-|
arr=np.array(data) |根据data创建数组
arr.ndim |获取维度
arr.shape |获取形状
np.zeros(n) |创建n个0的数组,n也可以是元组,书P95
np.empty(n) | 创建n个空值,n也可以是元组,书P95
np.arange(15) |创建0-14的数组
更多函数|P96

#### ndarray的数据类型
dtype是一个数据类型的对象
主要有浮点,证书,布尔,字符串类型
更详细的类型P97
函数|作用
|-|-|
arr.astype(np.float64)|将array转换为float64类型
`注意,在进行转化时会有精度损失`
#### 数组和标量之间的运算
``` PYTHON
arr([[1., 2., 3.],
    [4., 5., 6.]])

arr * arr
array([[ 1.,  4.,  9.],
       [16., 25., 36.]])

arr - arr
array([[0., 0., 0.],
       [0., 0., 0.]])

1/arr
array([[1.        , 0.5       , 0.33333333],
       [0.25      , 0.2       , 0.16666667]])

arr ** 0.5
array([[1.        , 1.41421356, 1.73205081],
       [2.        , 2.23606798, 2.44948974]])
```
#### 基本的索引和切片

> 切片是跟列表的区别在于,数组切片是原始数组的视图,这意味着数据不会被复制,任何修改都会直接反应到原始的数据源上
> 

```python
# 赋值
arr = ([0,1,2,3,4,12,123,51,23])
arr[5:7] = 12
arr
array([ 0,  1,  2,  3,  4, 12, 12, 51, 23])
```

```python
# 如果想得到切片的副本,需要使用
arr[5:8].copy()
```

```python
# 多维度下的索引
data = [[1.,2.,3.],[4.,5.,6.]]
arr = np.array(data)
arr[1]
array([4., 5., 6.])

# 也可以使用逗号
arr[1,2]
6.0
```

#### 切片索引
```python
data = [[1.,2.,3.],
        [4.,5.,6.],
        [7.,8.,9.]]

arr[:2]
array([[1., 2., 3.],
       [4., 5., 6.]])

arr[:2,1:]
array([[2., 3.],
       [5., 6.]])

arr[:,1:]
array([[2., 3.],
       [5., 6.],
       [8., 9.]])
```
#### 布尔型索引
```python
names = ['黄小明','佟大为','郭冬临','宋丹丹','吴青峰']
arr == '吴青峰'
array([False, False, False, False,  True])

names = ['黄小明','佟大为','郭冬临']
data = [[1.,2.,3.,7.,4.],
        [4.,5.,6.,2,6],
        [7.,8.,9.,7,2]]
arr1[arr == '佟大为']
arr1([[4. 5. 6. 2. 6.]])

arr1[~(arr == '佟大为')]
arr1([[1. 2. 3. 7. 4.]
      [7. 8. 9. 7. 2.]])
arr1[(arr == '佟大为') | (arr=='黄小明')]
[[1. 2. 3. 7. 4.]
 [4. 5. 6. 2. 6.]]

```

```python
arr = np.zeros((3,4))
arr[1] = 2
array([[0., 0., 0., 0.],
       [2., 2., 2., 2.],
       [0., 0., 0., 0.]])

arr[[1,2]]
array([[2., 2., 2., 2.],
       [0., 0., 0., 0.]])

```
函数|作用
|-|-|
np.ix_()|将两个一维数组转换为一个用于选取方形区域的索引器
```python
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14],
       [15, 16, 17, 18, 19],
       [20, 21, 22, 23, 24]])
arr[np.ix_([2,1,0,3],[3,4,1,2])]
array([[13, 14, 11, 12],
       [ 8,  9,  6,  7],
       [ 3,  4,  1,  2],
       [18, 19, 16, 17]])
```
#### 数组的转置和轴兑换
```python
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14],
       [15, 16, 17, 18, 19],
       [20, 21, 22, 23, 24]])
arr.T
array([[ 0,  5, 10, 15, 20],
       [ 1,  6, 11, 16, 21],
       [ 2,  7, 12, 17, 22],
       [ 3,  8, 13, 18, 23],
       [ 4,  9, 14, 19, 24]])
```
函数|作用
|-|-|
np.dot()|计算矩阵内积

```python
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])
np.dot(arr.T,arr)
array([[45, 54, 63],
       [54, 66, 78],
       [63, 78, 93]])
```
对于高纬度数组需要用transpose才能得到转置,由于基本上只在二维进行,我就步研究了P108

#### 通用函数
函数|作用
|-|-|
一元ufunc|只需要一个arr
二元ufunc|需要两个arr
np.sqrt(arr)|给arr里每个数据开方
np.exp(arr)|以e为底数的幂
np.maximun(x,y)|得到对应位置的最大值
np.modf(arr)|返回两个arr一个整数,一个小数,都带符号
更多函数详见P110|更多函数详见P110
```python
# np.maximun(x,y)
x:
array([-0.28025284, -0.84807051, -0.61452975,  0.66172767,  0.87234505,
        0.37468237, -1.0642168 , -0.18541553])
y:
array([-0.19353078, -0.24381745,  0.1518192 ,  0.29788911,  0.03196816,
       -0.51104235, -1.35755972,  1.07000497])
np.maximum(x,y)
array([-0.19353078, -0.24381745,  0.1518192 ,  0.66172767,  0.87234505,
        0.37468237, -1.0642168 ,  1.07000497])

# np.modf(arr)
np.modf(x)
(array([-0.28025284, -0.84807051, -0.61452975,  0.66172767,  0.87234505,
         0.37468237, -0.0642168 , -0.18541553]),
 array([-0., -0., -0.,  0.,  0.,  0., -1., -0.]))
```

### 利用数组进行数据处理
函数|作用
|-|-|
np.meshgrid(x,y)|通过x,y生成两二维矩阵


```python
array([4, 5, 6])
xs,ys = np.meshgrid(x,y)
xs
array([[4, 5, 6],
       [4, 5, 6],
       [4, 5, 6]])
ys
array([[4, 4, 4],
       [5, 5, 5],
       [6, 6, 6]])
```

#### 将条件逻辑运算
```python
x = np.array([1.1,1.2,1.3])
y = np.array([2.1,2.2,2.3])
cond = np.array([True,True,False])
result = [x if c else y for x,y,c in zip(x,y,cond)]
[1.1, 1.2, 2.3]
```
上面有两个问题
1. 它对大数组的处理速度不是很快
2. 无法用于多数组.可以用np.where代替

函数|作用
|-|-|
np.where(cond,x,y)|和上面功能差不多
```python
np.where(cond,x,y)
array([1.1, 1.2, 2.3])

# np.where的第二个和第三个不必是数组,也可以是标量值,比如
array([[ 1.01124455, -0.28142553, -0.93801783, -0.55833355]])

np.where(arr > 0,2,-2)
array([[ 2, -2, -2, -2]])

# 也可以使用嵌套语句
np.where(cond1 & cond2 ,0,
       np.where(cond1,1,
              np.where(cond2,2,3)))
```
#### 数学和统计方法
可以通过sum.mean以及标准差std等聚合计算对arr使用
函数|作用
|-|-|
arr.mean()或np.mean(arr)|求平均数,接受axis参数,计算结果
arr.sum()|求和,接受axis参数,计算结果
```python
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])
arr.mean(axis=1)  
array([1., 4., 7.])

arr.sum(0)     
array([ 9, 12, 15])
```
其他计算在P115
#### 用于布尔型数组的方法
函数|作用
|-|-|
arr.any()|检测数组只是否存在至少一个True
arr.all()|检测数组是否所有值都是True

```python
arr = np.random.randn(100)
(arr>0).sum()
47
```
#### 排序
函数|作用
|-|-|
arr.sort()|排序,可以指定轴排序

#### 唯一化及其其他的集合逻辑
函数|作用
|-|-|
np.unique(arr)|返回一个去重集合
np.in1d(arr,[])|测试一个数组在另一个成员数组是否存在
其他方法P118|其他方法P118
```python
values = np.array([6,9,1,2,6])
np.in1d(values,[1,6])
array([ True, False,  True, False,  True])
```
### 用于数组的文件输入输出
#### 将数组以二进制格式保存到磁盘
函数|作用
|-|-|
np.save('file_name',arr)|保存文件 file_name.npy
np.load('file_nmae')|读取文件
np.savez('file_name.npz',a=arr...)|可以保存多个数组到一个亚索文件中,将数组以关键字形式传入
```python
np.savez('x_npz',a=arr) 
np.load('x_npz.npz')['a']            
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```
#### 存取文本文件
函数|作用
|-|-|
np.loadtxt('file_name.txt',delimiter=',')|读取文本,以','号分割数据

###　线性代数
函数|作用
|-|-|
x.dot(y)　或　np.dot(x,y)|矩阵ｘ乘以ｙ

numpy.linalg 有求逆矩阵和行列式之类的东西
函数|作用
|-|-|
np.linalg.inv(arr)|求arr的逆矩阵
np.linalg.qr(arr)|不知道干嘛用,以后看看
更多函数P121|更多函数P121

### 随机数生成
函数|作用
|-|-|
np.random.normal(size=(4,4))|得到一个标准正态分布的4x4样本数组
更多函数P122|更多函数P122

#### 随机漫步
书P123

## Pandas
### pandas数据结构介绍
#### Series
Series是类似于一维数组的对象,它由一组数据和一组与之关系的数据标签组成.仅由一组数据即可产生简单的Series
<pre><font color="#008700">In [</font><font color="#8AE234"><b>41</b></font><font color="#008700">]: </font><font color="#008700"><b>from</b></font> <font color="#0087D7"><b>pandas</b></font> <font color="#008700"><b>import</b></font> Series,DataFrame                                                       

<font color="#008700">In [</font><font color="#8AE234"><b>42</b></font><font color="#008700">]: </font><font color="#008700"><b>import</b></font> <font color="#0087D7"><b>pandas</b></font> <font color="#008700"><b>as</b></font> <font color="#0087D7"><b>pd</b></font>                                                                                                   

<font color="#008700">In [</font><font color="#8AE234"><b>43</b></font><font color="#008700">]: </font>obj = Series([<font color="#008700">4</font>,<font color="#008700">6</font>,<font color="#008700">1</font>,-<font color="#008700">5</font>])                                                                                              

<font color="#008700">In [</font><font color="#8AE234"><b>44</b></font><font color="#008700">]: </font>obj                                                                                                                   
<font color="#870000">Out[</font><font color="#EF2929"><b>44</b></font><font color="#870000">]: </font>
0    4
1    6
2    1
3   -5
dtype: int64

<font color="#008700">In [</font><font color="#8AE234"><b>45</b></font><font color="#008700">]: </font>obj.values                                                                                                            
<font color="#870000">Out[</font><font color="#EF2929"><b>45</b></font><font color="#870000">]: </font>array([ 4,  6,  1, -5])

<font color="#008700">In [</font><font color="#8AE234"><b>5</b></font><font color="#008700">]: </font>obj.index                                                                                                              
<font color="#870000">Out[</font><font color="#EF2929"><b>5</b></font><font color="#870000">]: </font>RangeIndex(start=0, stop=4, step=1)
</pre>

也可以自定义index
<pre><font color="#008700">In [</font><font color="#8AE234"><b>9</b></font><font color="#008700">]: </font>obj2 = Series([<font color="#008700">4</font>,<font color="#008700">7</font>,<font color="#008700">6</font>,-<font color="#008700">3</font>],index=[<font color="#AF5F00">&apos;d&apos;</font>,<font color="#AF5F00">&apos;b&apos;</font>,<font color="#AF5F00">&apos;e&apos;</font>,<font color="#AF5F00">&apos;g&apos;</font>])                                                                      

<font color="#008700">In [</font><font color="#8AE234"><b>10</b></font><font color="#008700">]: </font>obj2                                                                                                                  
<font color="#870000">Out[</font><font color="#EF2929"><b>10</b></font><font color="#870000">]: </font>
d    4
b    7
e    6
g   -3
dtype: int64
</pre>

取值
<pre><font color="#008700">In [</font><font color="#8AE234"><b>11</b></font><font color="#008700">]: </font>obj2[<font color="#AF5F00">&apos;d&apos;</font>]                                                                                                             
<font color="#870000">Out[</font><font color="#EF2929"><b>11</b></font><font color="#870000">]: </font>4

<font color="#008700">In [</font><font color="#8AE234"><b>12</b></font><font color="#008700">]: </font>obj2[[<font color="#AF5F00">&apos;d&apos;</font>,<font color="#AF5F00">&apos;e&apos;</font>]]                                                                                                       
<font color="#870000">Out[</font><font color="#EF2929"><b>12</b></font><font color="#870000">]: </font>
d    4
e    6
dtype: int64
</pre>

计算
<pre><font color="#008700">In [</font><font color="#8AE234"><b>13</b></font><font color="#008700">]: </font>obj2[obj2&gt;<font color="#008700">0</font>]                                                                                                          
<font color="#870000">Out[</font><font color="#EF2929"><b>13</b></font><font color="#870000">]: </font>
d    4
b    7
e    6
dtype: int64

<font color="#008700">In [</font><font color="#8AE234"><b>14</b></font><font color="#008700">]: </font>obj2 * <font color="#008700">2</font>                                                                                                              
<font color="#870000">Out[</font><font color="#EF2929"><b>14</b></font><font color="#870000">]: </font>
d     8
b    14
e    12
g    -6
dtype: int64

<font color="#008700">In [</font><font color="#8AE234"><b>17</b></font><font color="#008700">]: </font>np.exp(obj2)                                                                                                          
<font color="#870000">Out[</font><font color="#EF2929"><b>17</b></font><font color="#870000">]: </font>
d      54.598150
b    1096.633158
e     403.428793
g       0.049787
dtype: float64
</pre>

Series 中的index就像是字典的key
<pre><font color="#008700">In [</font><font color="#8AE234"><b>18</b></font><font color="#008700">]: </font><font color="#AF5F00">&apos;b&apos;</font> <font color="#AF00FF"><b>in</b></font> obj2                                                                                                           
<font color="#870000">Out[</font><font color="#EF2929"><b>18</b></font><font color="#870000">]: </font>True
</pre>
也可以使用字典直接创建Series
<pre><font color="#008700">In [</font><font color="#8AE234"><b>20</b></font><font color="#008700">]: </font>sdata = {<font color="#AF5F00">&apos;aa&apos;</font>:<font color="#008700">2323</font>,<font color="#AF5F00">&apos;bb&apos;</font>:<font color="#008700">5343</font>,<font color="#AF5F00">&apos;cc&apos;</font>:<font color="#008700">231</font>}          
<font color="#008700">In [</font><font color="#8AE234"><b>21</b></font><font color="#008700">]: </font>obj3 = Series(sdata)            
<font color="#008700">In [</font><font color="#8AE234"><b>22</b></font><font color="#008700">]: </font>obj3                            
<font color="#870000">Out[</font><font color="#EF2929"><b>22</b></font><font color="#870000">]: </font>
aa    2323
bb    5343
cc     231
dtype: int64
</pre>
传入一个字典和一个index
<pre>
<font color="#008700">In [</font><font color="#8AE234"><b>23</b></font><font color="#008700">]: </font>states = [<font color="#AF5F00">&apos;aa&apos;</font>,<font color="#AF5F00">&apos;cc&apos;</font>,<font color="#AF5F00">&apos;dd&apos;</font>]                
<font color="#008700">In [</font><font color="#8AE234"><b>24</b></font><font color="#008700">]: </font>obj4 = Series(sdata,index=states)                       
<font color="#008700">In [</font><font color="#8AE234"><b>25</b></font><font color="#008700">]: </font>obj4                               
<font color="#870000">Out[</font><font color="#EF2929"><b>25</b></font><font color="#870000">]: </font>
aa    2323.0
cc     231.0
dd       NaN
dtype: float64
</pre>
函数|作用
|-|-|
pd.isnull(ser) 或 ser.isnull()|为null的index为True 不为null为Fasle
pd.notnull(ser)|和上面相反
`Series最重要的功能是可以自动对其不同索引的数据`
<pre><font color="#008700">In [</font><font color="#8AE234"><b>26</b></font><font color="#008700">]: </font>obj3 + obj4                                                                                                           
<font color="#870000">Out[</font><font color="#EF2929"><b>26</b></font><font color="#870000">]: </font>
aa    4646.0
bb       NaN
cc     462.0
dd       NaN
dtype: float64
</pre>

还可以设置Series的名字.index也是.该属性和其他pandas功能非常密切
<pre><font color="#008700">In [</font><font color="#8AE234"><b>27</b></font><font color="#008700">]: </font>obj4.name = <font color="#AF5F00">&apos;population&apos;</font>                        
<font color="#008700">In [</font><font color="#8AE234"><b>28</b></font><font color="#008700">]: </font>obj4.index.name = <font color="#AF5F00">&apos;state&apos;</font>                                
<font color="#008700">In [</font><font color="#8AE234"><b>29</b></font><font color="#008700">]: </font>obj4                                   
<font color="#870000">Out[</font><font color="#EF2929"><b>29</b></font><font color="#870000">]: </font>
state
aa    2323.0
cc     231.0
dd       NaN
Name: population, dtype: float64
</pre>

Series的index也可以通过赋值的方式修改
<pre><font color="#008700">In [</font><font color="#8AE234"><b>31</b></font><font color="#008700">]: </font>obj4.index = [<font color="#AF5F00">&apos;bob&apos;</font>,<font color="#AF5F00">&apos;jame&apos;</font>,<font color="#AF5F00">&apos;jeff&apos;</font>]                               
<font color="#008700">In [</font><font color="#8AE234"><b>32</b></font><font color="#008700">]: </font>obj4                                   
<font color="#870000">Out[</font><font color="#EF2929"><b>32</b></font><font color="#870000">]: </font>
bob     2323.0
jame     231.0
jeff       NaN
Name: population, dtype: float64
</pre>
###DataFrame
DataFrame是表个性数据.包含一个有序列,每列的数值可以是不同的值类型.有行索引和列索引.就是二位数组了
<pre><font color="#008700">In [</font><font color="#8AE234"><b>34</b></font><font color="#008700">]: </font>data = {<font color="#AF5F00">&apos;state&apos;</font>:[<font color="#AF5F00">&apos;a&apos;</font>,<font color="#AF5F00">&apos;b&apos;</font>,<font color="#AF5F00">&apos;c&apos;</font>],<font color="#AF5F00">&apos;year&apos;</font>:[<font color="#008700">1</font>,<font color="#008700">2</font>,<font color="#008700">3</font>],<font color="#AF5F00">&apos;pop&apos;</font>:[<font color="#008700">1.1</font>,<font color="#008700">1.2</font>,<font color="#008700">1.3</font>]}   
<font color="#008700">In [</font><font color="#8AE234"><b>35</b></font><font color="#008700">]: </font>frame = DataFrame(data)                       
<font color="#008700">In [</font><font color="#8AE234"><b>36</b></font><font color="#008700">]: </font>frame                        
<font color="#870000">Out[</font><font color="#EF2929"><b>36</b></font><font color="#870000">]: </font>
  state  year  pop
0     a     1  1.1
1     b     2  1.2
2     c     3  1.3
</pre>
按指定列进行排序
<pre><font color="#008700">In [</font><font color="#8AE234"><b>37</b></font><font color="#008700">]: </font>frame = DataFrame(data,index=[<font color="#AF5F00">&apos;year&apos;</font>,<font color="#AF5F00">&apos;state&apos;</font>,<font color="#AF5F00">&apos;pop&apos;</font>])               
<font color="#008700">In [</font><font color="#8AE234"><b>38</b></font><font color="#008700">]: </font>frame                      
<font color="#870000">Out[</font><font color="#EF2929"><b>38</b></font><font color="#870000">]: </font>
      state  year  pop
year      a     1  1.1
state     b     2  1.2
pop       c     3  1.3
</pre>
同Series一样,要是传入找不到的值就会产生NA值
<pre><font color="#008700">In [</font><font color="#8AE234"><b>39</b></font><font color="#008700">]: </font>frame.columns               
<font color="#870000">Out[</font><font color="#EF2929"><b>39</b></font><font color="#870000">]: </font>Index([&apos;state&apos;, &apos;year&apos;, &apos;pop&apos;], dtype=&apos;object&apos;)
</pre>
取列值
<pre><font color="#008700">In [</font><font color="#8AE234"><b>40</b></font><font color="#008700">]: </font>frame[<font color="#AF5F00">&apos;state&apos;</font>]                                                                                                        
<font color="#870000">Out[</font><font color="#EF2929"><b>40</b></font><font color="#870000">]: </font>
year     a
state    b
pop      c
Name: state, dtype: object

<font color="#008700">In [</font><font color="#8AE234"><b>41</b></font><font color="#008700">]: </font>frame.state              
<font color="#870000">Out[</font><font color="#EF2929"><b>41</b></font><font color="#870000">]: </font>
year     a
state    b
pop      c
Name: state, dtype: object
</pre>

取行值
<pre><font color="#008700">In [</font><font color="#8AE234"><b>47</b></font><font color="#008700">]: </font>frame.ix[<font color="#AF5F00">&apos;year&apos;</font>]                
<font color="#870000">Out[</font><font color="#EF2929"><b>47</b></font><font color="#870000">]: </font>
state      a
year       1
pop      1.1
Name: year, dtype: object
</pre>

赋值
<pre><font color="#008700">In [</font><font color="#8AE234"><b>48</b></font><font color="#008700">]: </font>frame[<font color="#AF5F00">&apos;year&apos;</font>] = <font color="#008700">12</font>       
<font color="#008700">In [</font><font color="#8AE234"><b>49</b></font><font color="#008700">]: </font>frame                                                   
<font color="#870000">Out[</font><font color="#EF2929"><b>49</b></font><font color="#870000">]: </font>
      state  year  pop
year      a    12  1.1
state     b    12  1.2
pop       c    12  1.3
</pre>
将列表或者数组赋值给摸个列时,必须保证跟DataFrame的长度匹配.如果赋值的是一个Series就会精确匹配DataFrame的索引,所有的空位都被填上缺失值
<pre><font color="#008700">In [</font><font color="#8AE234"><b>53</b></font><font color="#008700">]: </font>frame[<font color="#AF5F00">&apos;aa&apos;</font>]=val                  
<font color="#008700">In [</font><font color="#8AE234"><b>54</b></font><font color="#008700">]: </font>frame                   
<font color="#870000">Out[</font><font color="#EF2929"><b>54</b></font><font color="#870000">]: </font>
      state  year  pop   aa
year      a    12  1.1  1.1
state     b    12  1.2  NaN
pop       c    12  1.3 -1.5
</pre>
用del删除列
<pre><font color="#008700">In [</font><font color="#8AE234"><b>55</b></font><font color="#008700">]: </font><font color="#008700"><b>del</b></font> frame[<font color="#AF5F00">&apos;aa&apos;</font>]                      
<font color="#008700">In [</font><font color="#8AE234"><b>56</b></font><font color="#008700">]: </font>frame                                                
<font color="#870000">Out[</font><font color="#EF2929"><b>56</b></font><font color="#870000">]: </font>
      state  year  pop
year      a    12  1.1
state     b    12  1.2
pop       c    12  1.3
</pre>
`通过索引得到的Series也是视图,任何修改都会影响到源数据`
字典创建
<pre><font color="#008700">In [</font><font color="#8AE234"><b>57</b></font><font color="#008700">]: </font>pop = {<font color="#AF5F00">&apos;Nevada&apos;</font>:{<font color="#008700">1</font>:<font color="#008700">2.4</font>,<font color="#008700">2</font>:<font color="#008700">3.5</font>},<font color="#AF5F00">&apos;Ohio&apos;</font>:{<font color="#008700">2</font>:<font color="#008700">33</font>,<font color="#008700">5</font>:<font color="#008700">123</font>}}  
<font color="#008700">In [</font><font color="#8AE234"><b>58</b></font><font color="#008700">]: </font>frame3 = DataFrame(pop                
<font color="#008700">In [</font><font color="#8AE234"><b>60</b></font><font color="#008700">]: </font>frame3                  
<font color="#870000">Out[</font><font color="#EF2929"><b>60</b></font><font color="#870000">]: </font>
   Nevada   Ohio
1     2.4    NaN
2     3.5   33.0
5     NaN  123.0
</pre>
转置
<pre><font color="#008700">In [</font><font color="#8AE234"><b>61</b></font><font color="#008700">]: </font>frame3.T                  
<font color="#870000">Out[</font><font color="#EF2929"><b>61</b></font><font color="#870000">]: </font>
          1     2      5
Nevada  2.4   3.5    NaN
Ohio    NaN  33.0  123.0
</pre>
                                
其他赋值
<pre><font color="#008700">In [</font><font color="#8AE234"><b>62</b></font><font color="#008700">]: </font>pdata = {<font color="#AF5F00">&apos;Ohio&apos;</font>:frame3[<font color="#AF5F00">&apos;Ohio&apos;</font>][:-<font color="#008700">1</font>],<font color="#AF5F00">&apos;Nevada&apos;</font>:frame3[<font color="#AF5F00">&apos;Nevada&apos;</font>][:<font color="#008700">2</font>]}    
<font color="#008700">In [</font><font color="#8AE234"><b>63</b></font><font color="#008700">]: </font>DataFrame(pdata)       
<font color="#870000">Out[</font><font color="#EF2929"><b>63</b></font><font color="#870000">]: </font>
   Ohio  Nevada
1   NaN     2.4
2  33.0     3.5
</pre>
更多构造方法P134
column和index也同样可以被设置name
<pre><font color="#008700">In [</font><font color="#8AE234"><b>68</b></font><font color="#008700">]: </font>frame3.columns.name=<font color="#AF5F00">&apos;yy&apos;</font>                  
<font color="#008700">In [</font><font color="#8AE234"><b>69</b></font><font color="#008700">]: </font>frame3                                               
<font color="#870000">Out[</font><font color="#EF2929"><b>69</b></font><font color="#870000">]: </font>
yy  Nevada   Ohio
xx               
1      2.4    NaN
2      3.5   33.0
5      NaN  123.0
</pre>

<pre><font color="#008700">In [</font><font color="#8AE234"><b>70</b></font><font color="#008700">]: </font>frame.values           
<font color="#870000">Out[</font><font color="#EF2929"><b>70</b></font><font color="#870000">]: </font>
array([[&apos;a&apos;, 12, 1.1],
       [&apos;b&apos;, 12, 1.2],
       [&apos;c&apos;, 12, 1.3]], dtype=object)
</pre>
#### 索引对象
索引对象负责管理轴标签和其他元数据(轴名称等).构建Series或者DateFrame时,任何数组或其他序列的标签都会被转成一个indeex
<pre><font color="#008700">In [</font><font color="#8AE234"><b>4</b></font><font color="#008700">]: </font>obj = Series(<font color="#008700">range</font>(<font color="#008700">3</font>),index=[<font color="#AF5F00">&apos;a&apos;</font>,<font color="#AF5F00">&apos;b&apos;</font>,<font color="#AF5F00">&apos;c&apos;</font>])                              
<font color="#008700">In [</font><font color="#8AE234"><b>5</b></font><font color="#008700">]: </font>index = obj.index                                                      
<font color="#008700">In [</font><font color="#8AE234"><b>6</b></font><font color="#008700">]: </font>index                                                                 
<font color="#870000">Out[</font><font color="#EF2929"><b>6</b></font><font color="#870000">]: </font>Index([&apos;a&apos;, &apos;b&apos;, &apos;c&apos;], dtype=&apos;object&apos;)
<font color="#008700">In [</font><font color="#8AE234"><b>7</b></font><font color="#008700">]: </font>index[<font color="#008700">1</font>:]                                                               
<font color="#870000">Out[</font><font color="#EF2929"><b>7</b></font><font color="#870000">]: </font>Index([&apos;b&apos;, &apos;c&apos;], dtype=&apos;object&apos;)
</pre>
`注意:index里面的数据不可修改`
<pre><font color="#008700">In [</font><font color="#8AE234"><b>11</b></font><font color="#008700">]: </font>index = pd.Index(np.arange(<font color="#008700">3</font>))                                         
<font color="#008700">In [</font><font color="#8AE234"><b>12</b></font><font color="#008700">]: </font>obj2 = Series([<font color="#008700">1.5</font>,-<font color="#008700">2.5</font>,<font color="#008700">0</font>],index=index)                         
<font color="#008700">In [</font><font color="#8AE234"><b>13</b></font><font color="#008700">]: </font>obj2.index <font color="#AF00FF"><b>is</b></font> index                                 
<font color="#870000">Out[</font><font color="#EF2929"><b>13</b></font><font color="#870000">]: </font>True
</pre>
更多P136,知道基本的其实就好

同样的.index可以进行索引
index的方法和属性P137
### 基本功能
#### 重新索引
<pre><font color="#008700">In [</font><font color="#8AE234"><b>14</b></font><font color="#008700">]: </font>obj = Series([<font color="#008700">4.5</font>,<font color="#008700">7.2</font>,-<font color="#008700">5.3</font>,<font color="#008700">3.6</font>],index=[<font color="#AF5F00">&apos;d&apos;</font>,<font color="#AF5F00">&apos;b&apos;</font>,<font color="#AF5F00">&apos;a&apos;</font>,<font color="#AF5F00">&apos;c&apos;</font>])               

<font color="#008700">In [</font><font color="#8AE234"><b>15</b></font><font color="#008700">]: </font>obj                                                                    
<font color="#870000">Out[</font><font color="#EF2929"><b>15</b></font><font color="#870000">]: </font>
d    4.5
b    7.2
a   -5.3
c    3.6
dtype: float64
<font color="#008700">In [</font><font color="#8AE234"><b>16</b></font><font color="#008700">]: </font>obj2 = obj.reindex([<font color="#AF5F00">&apos;a&apos;</font>,<font color="#AF5F00">&apos;b&apos;</font>,<font color="#AF5F00">&apos;c&apos;</font>,<font color="#AF5F00">&apos;d&apos;</font>,<font color="#AF5F00">&apos;e&apos;</font>])                              
<font color="#008700">In [</font><font color="#8AE234"><b>17</b></font><font color="#008700">]: </font>obj2                                                                   
<font color="#870000">Out[</font><font color="#EF2929"><b>17</b></font><font color="#870000">]: </font>
a   -5.3
b    7.2
c    3.6
d    4.5
e    NaN
dtype: float64
<font color="#008700">In [</font><font color="#8AE234"><b>18</b></font><font color="#008700">]: </font>obj                                                                    
<font color="#870000">Out[</font><font color="#EF2929"><b>18</b></font><font color="#870000">]: </font>
d    4.5
b    7.2
a   -5.3
c    3.6
dtype: float64
</pre>
<pre><font color="#008700">In [</font><font color="#8AE234"><b>19</b></font><font color="#008700">]: </font>obj.reindex([<font color="#AF5F00">&apos;a&apos;</font>,<font color="#AF5F00">&apos;b&apos;</font>,<font color="#AF5F00">&apos;c&apos;</font>,<font color="#AF5F00">&apos;d&apos;</font>,<font color="#AF5F00">&apos;e&apos;</font>],fill_value=<font color="#008700">0</font>)                        
<font color="#870000">Out[</font><font color="#EF2929"><b>19</b></font><font color="#870000">]: </font>
a   -5.3
b    7.2
c    3.6
d    4.5
e    0.0
dtype: float64
</pre>

填充
<pre><font color="#008700">In [</font><font color="#8AE234"><b>22</b></font><font color="#008700">]: </font>obj3.reindex(<font color="#008700">range</font>(<font color="#008700">6</font>),method=<font color="#AF5F00">&apos;ffill&apos;</font>)                                  
<font color="#870000">Out[</font><font color="#EF2929"><b>22</b></font><font color="#870000">]: </font>
0      blue
1      blue
2    purple
3    purple
4    yellow
5    yellow
dtype: object
</pre>
更多填充的method选项P138

修改columns索引
<pre><font color="#008700">In [</font><font color="#8AE234"><b>24</b></font><font color="#008700">]: </font>frame = DataFrame(np.arange(<font color="#008700">9</font>).reshape((<font color="#008700">3</font>,<font color="#008700">3</font>)),index=[<font color="#AF5F00">&apos;a&apos;</font>,<font color="#AF5F00">&apos;c&apos;</font>,<font color="#AF5F00">&apos;d&apos;</font>],colum
<font color="#008700">    ...: </font>ns=[<font color="#AF5F00">&apos;Ohio&apos;</font>,<font color="#AF5F00">&apos;Texas&apos;</font>,<font color="#AF5F00">&apos;California&apos;</font>])                                      

<font color="#008700">In [</font><font color="#8AE234"><b>25</b></font><font color="#008700">]: </font>frame                                                                  
<font color="#870000">Out[</font><font color="#EF2929"><b>25</b></font><font color="#870000">]: </font>
   Ohio  Texas  California
a     0      1           2
c     3      4           5
d     6      7           8

<font color="#008700">In [</font><font color="#8AE234"><b>26</b></font><font color="#008700">]: </font>states = [<font color="#AF5F00">&apos;Texas&apos;</font>,<font color="#AF5F00">&apos;Utah&apos;</font>,<font color="#AF5F00">&apos;California&apos;</font>]                                 

<font color="#008700">In [</font><font color="#8AE234"><b>27</b></font><font color="#008700">]: </font>frame.reindex(columns=states)                                          
<font color="#870000">Out[</font><font color="#EF2929"><b>27</b></font><font color="#870000">]: </font>
   Texas  Utah  California
a      1   NaN           2
c      4   NaN           5
d      7   NaN           8
</pre>
同时进行index,column修改,进行method填充时只能进行轴0填充
更多reindex参数 P140

#### 丢弃指定轴上的项
<pre><font color="#008700">In [</font><font color="#8AE234"><b>31</b></font><font color="#008700">]: </font>obj = Series(np.arange(<font color="#008700">5</font>),index=[<font color="#AF5F00">&apos;a&apos;</font>,<font color="#AF5F00">&apos;b&apos;</font>,<font color="#AF5F00">&apos;c&apos;</font>,<font color="#AF5F00">&apos;d&apos;</font>,<font color="#AF5F00">&apos;e&apos;</font>])                 

<font color="#008700">In [</font><font color="#8AE234"><b>32</b></font><font color="#008700">]: </font>new_obj = obj.drop(<font color="#AF5F00">&apos;c&apos;</font>)                                                

<font color="#008700">In [</font><font color="#8AE234"><b>33</b></font><font color="#008700">]: </font>new_obj                                                                
<font color="#870000">Out[</font><font color="#EF2929"><b>33</b></font><font color="#870000">]: </font>
a    0
b    1
d    3
e    4
dtype: int64
</pre>
删除column
<pre><font color="#008700">In [</font><font color="#8AE234"><b>36</b></font><font color="#008700">]: </font>data.drop<span style="background-color:#AFD7D7"><font color="#000000">(</font></span><font color="#AF5F00">&apos;two&apos;</font>,axis=<font color="#008700">1</font><span style="background-color:#870000"><font color="#FF8787">)</font></span>     </pre>
#### 索引,选取和过滤
<pre>a    0.0
b    1.0
c    2.0
d    3.0
dtype: float64

<font color="#008700">In [</font><font color="#8AE234"><b>39</b></font><font color="#008700">]: </font>obj[<font color="#AF5F00">&apos;b&apos;</font>]                                                               
<font color="#870000">Out[</font><font color="#EF2929"><b>39</b></font><font color="#870000">]: </font>1.0

<font color="#008700">In [</font><font color="#8AE234"><b>40</b></font><font color="#008700">]: </font>obj[<font color="#008700">1</font>]                                                                 
<font color="#870000">Out[</font><font color="#EF2929"><b>40</b></font><font color="#870000">]: </font>1.0

<font color="#008700">In [</font><font color="#8AE234"><b>41</b></font><font color="#008700">]: </font>obj[<font color="#008700">2</font>:<font color="#008700">4</font>]                                                               
<font color="#870000">Out[</font><font color="#EF2929"><b>41</b></font><font color="#870000">]: </font>
c    2.0
d    3.0
dtype: float64

<font color="#008700">In [</font><font color="#8AE234"><b>42</b></font><font color="#008700">]: </font>obj[[<font color="#008700">1</font>,<font color="#008700">3</font>]]                                                             
<font color="#870000">Out[</font><font color="#EF2929"><b>42</b></font><font color="#870000">]: </font>
b    1.0
d    3.0
dtype: float64

<font color="#008700">In [</font><font color="#8AE234"><b>43</b></font><font color="#008700">]: </font>obj[[<font color="#AF5F00">&apos;a&apos;</font>,<font color="#AF5F00">&apos;c&apos;</font>]]                                                         
<font color="#870000">Out[</font><font color="#EF2929"><b>43</b></font><font color="#870000">]: </font>
a    0.0
c    2.0
dtype: float64

<font color="#008700">In [</font><font color="#8AE234"><b>44</b></font><font color="#008700">]: </font>obj[obj&lt;<font color="#008700">2</font>]                                                             
<font color="#870000">Out[</font><font color="#EF2929"><b>44</b></font><font color="#870000">]: </font>
a    0.0
b    1.0
dtype: float64
</pre>

`标签切片运算和普通的切片运算是不同的,它的末端是包含的'
<pre><font color="#008700">In [</font><font color="#8AE234"><b>45</b></font><font color="#008700">]: </font>obj[<font color="#AF5F00">&apos;b&apos;</font>:<font color="#AF5F00">&apos;c&apos;</font>]                                                           
<font color="#870000">Out[</font><font color="#EF2929"><b>45</b></font><font color="#870000">]: </font>
b    1.0
c    2.0
dtype: float64
</pre>
赋值
<pre><font color="#008700">In [</font><font color="#8AE234"><b>46</b></font><font color="#008700">]: </font>obj[<font color="#AF5F00">&apos;b&apos;</font>:<font color="#AF5F00">&apos;c&apos;</font>] = <font color="#008700">5</font>                                                       

<font color="#008700">In [</font><font color="#8AE234"><b>47</b></font><font color="#008700">]: </font>obj                                                                    
<font color="#870000">Out[</font><font color="#EF2929"><b>47</b></font><font color="#870000">]: </font>
a    0.0
b    5.0
c    5.0
d    3.0
dtype: float64
</pre>

<pre><font color="#008700">In [</font><font color="#8AE234"><b>49</b></font><font color="#008700">]: </font>data                                                                   
<font color="#870000">Out[</font><font color="#EF2929"><b>49</b></font><font color="#870000">]: </font>
          one  two  three  four
Ohio        0    1      2     3
Colorado    4    5      6     7
Utah        8    9     10    11
New York   12   13     14    15

<font color="#008700">In [</font><font color="#8AE234"><b>50</b></font><font color="#008700">]: </font>data[<font color="#AF5F00">&apos;two&apos;</font>]                                                            
<font color="#870000">Out[</font><font color="#EF2929"><b>50</b></font><font color="#870000">]: </font>
Ohio         1
Colorado     5
Utah         9
New York    13
Name: two, dtype: int64

<font color="#008700">In [</font><font color="#8AE234"><b>51</b></font><font color="#008700">]: </font>data[[<font color="#AF5F00">&apos;three&apos;</font>,<font color="#AF5F00">&apos;two&apos;</font>]]                                                  
<font color="#870000">Out[</font><font color="#EF2929"><b>51</b></font><font color="#870000">]: </font>
          three  two
Ohio          2    1
Colorado      6    5
Utah         10    9
New York     14   13
</pre>
<pre><font color="#008700">In [</font><font color="#8AE234"><b>52</b></font><font color="#008700">]: </font>data[:<font color="#008700">2</font>]                                                               
<font color="#870000">Out[</font><font color="#EF2929"><b>52</b></font><font color="#870000">]: </font>
          one  two  three  four
Ohio        0    1      2     3
Colorado    4    5      6     7

<font color="#008700">In [</font><font color="#8AE234"><b>53</b></font><font color="#008700">]: </font>data[data[<font color="#AF5F00">&apos;three&apos;</font>]&gt;<font color="#008700">5</font>]                                                  
<font color="#870000">Out[</font><font color="#EF2929"><b>53</b></font><font color="#870000">]: </font>
          one  two  three  four
Colorado    4    5      6     7
Utah        8    9     10    11
New York   12   13     14    15
</pre>
布尔运算
<pre><font color="#008700">In [</font><font color="#8AE234"><b>54</b></font><font color="#008700">]: </font>data[data &lt; <font color="#008700">5</font>] = <font color="#008700">0</font>                                                     

<font color="#008700">In [</font><font color="#8AE234"><b>55</b></font><font color="#008700">]: </font>data                                                                   
<font color="#870000">Out[</font><font color="#EF2929"><b>55</b></font><font color="#870000">]: </font>
          one  two  three  four
Ohio        0    0      0     0
Colorado    0    5      6     7
Utah        8    9     10    11
New York   12   13     14    15
</pre>
更多索引手段
<pre><font color="#008700">In [</font><font color="#8AE234"><b>56</b></font><font color="#008700">]: </font>data.ix[<font color="#AF5F00">&apos;Colorado&apos;</font>,[<font color="#AF5F00">&apos;two&apos;</font>,<font color="#AF5F00">&apos;three&apos;</font>]]                                    
/home/wangjuncheng/.local/bin/ipython:1: DeprecationWarning: 
.ix is deprecated. Please use
.loc for label based indexing or
.iloc for positional indexing

See the documentation here:
http://pandas.pydata.org/pandas-docs/stable/indexing.html#ix-indexer-is-deprecated
  #!/usr/bin/python3
<font color="#870000">Out[</font><font color="#EF2929"><b>56</b></font><font color="#870000">]: </font>
two      5
three    6
Name: Colorado, dtype: int64

<font color="#008700">In [</font><font color="#8AE234"><b>57</b></font><font color="#008700">]: </font>data.ix[<font color="#008700">2</font>]                                                             
/home/wangjuncheng/.local/bin/ipython:1: DeprecationWarning: 
.ix is deprecated. Please use
.loc for label based indexing or
.iloc for positional indexing

See the documentation here:
http://pandas.pydata.org/pandas-docs/stable/indexing.html#ix-indexer-is-deprecated
  #!/usr/bin/python3
<font color="#870000">Out[</font><font color="#EF2929"><b>57</b></font><font color="#870000">]: </font>
one       8
two       9
three    10
four     11
Name: Utah, dtype: int64

<font color="#008700">In [</font><font color="#8AE234"><b>58</b></font><font color="#008700">]: </font>data.ix[:<font color="#AF5F00">&apos;Utah&apos;</font>,<font color="#AF5F00">&apos;two&apos;</font>]                                                 
/home/wangjuncheng/.local/bin/ipython:1: DeprecationWarning: 
.ix is deprecated. Please use
.loc for label based indexing or
.iloc for positional indexing

See the documentation here:
http://pandas.pydata.org/pandas-docs/stable/indexing.html#ix-indexer-is-deprecated
  #!/usr/bin/python3
<font color="#870000">Out[</font><font color="#EF2929"><b>58</b></font><font color="#870000">]: </font>
Ohio        0
Colorado    5
Utah        9
Name: two, dtype: int64

<font color="#008700">In [</font><font color="#8AE234"><b>59</b></font><font color="#008700">]: </font>data.ix[data.three &gt; <font color="#008700">5</font>,:<font color="#008700">3</font>]                                             
/home/wangjuncheng/.local/bin/ipython:1: DeprecationWarning: 
.ix is deprecated. Please use
.loc for label based indexing or
.iloc for positional indexing

See the documentation here:
http://pandas.pydata.org/pandas-docs/stable/indexing.html#ix-indexer-is-deprecated
  #!/usr/bin/python3
<font color="#870000">Out[</font><font color="#EF2929"><b>59</b></font><font color="#870000">]: </font>
          one  two  three
Colorado    0    5      6
Utah        8    9     10
New York   12   13     14
</pre>
更多索引选项P144

#### 算术运算和数据对齐
<pre><font color="#008700">In [</font><font color="#8AE234"><b>60</b></font><font color="#008700">]: </font>s1 = Series([<font color="#008700">7.3</font>,-<font color="#008700">2.6</font>,<font color="#008700">3.5</font>,<font color="#008700">1.5</font>],index=[<font color="#AF5F00">&apos;a&apos;</font>,<font color="#AF5F00">&apos;c&apos;</font>,<font color="#AF5F00">&apos;e&apos;</font>,<font color="#AF5F00">&apos;d&apos;</font>])                

<font color="#008700">In [</font><font color="#8AE234"><b>61</b></font><font color="#008700">]: </font>s2 = Series([<font color="#008700">1</font>,<font color="#008700">2</font>,<font color="#008700">3</font>,<font color="#008700">4</font>],index=[<font color="#AF5F00">&apos;a&apos;</font>,<font color="#AF5F00">&apos;b&apos;</font>,<font color="#AF5F00">&apos;c&apos;</font>,<font color="#AF5F00">&apos;d&apos;</font>])                         

<font color="#008700">In [</font><font color="#8AE234"><b>62</b></font><font color="#008700">]: </font>s1+s2                                                                  
<font color="#870000">Out[</font><font color="#EF2929"><b>62</b></font><font color="#870000">]: </font>
a    8.3
b    NaN
c    0.4
d    5.5
e    NaN
dtype: float64
</pre>
`注意DataFrame和Series操作不一样`
<pre><font color="#008700">In [</font><font color="#8AE234"><b>6</b></font><font color="#008700">]: </font>df1 = DataFrame(np.arange(<font color="#008700">9.</font>).reshape((<font color="#008700">3</font>,<font color="#008700">3</font>)),columns=<font color="#008700">list</font>(<font color="#AF5F00">&apos;bcd&apos;</font>),index=[
<font color="#008700">   ...: </font><font color="#AF5F00">&apos;Ohio&apos;</font>,<font color="#AF5F00">&apos;Texas&apos;</font>,<font color="#AF5F00">&apos;Colorado&apos;</font>])                                             

<font color="#008700">In [</font><font color="#8AE234"><b>7</b></font><font color="#008700">]: </font>df2 = DataFrame(np.arange(<font color="#008700">12.</font>).reshape((<font color="#008700">4</font>,<font color="#008700">3</font>)),columns=<font color="#008700">list</font>(<font color="#AF5F00">&apos;bde&apos;</font>),index=
<font color="#008700">   ...: </font>[<font color="#AF5F00">&apos;Ohio&apos;</font>,<font color="#AF5F00">&apos;Texas&apos;</font>,<font color="#AF5F00">&apos;Colorado&apos;</font>,<font color="#AF5F00">&apos;Oregon&apos;</font>])                                   

<font color="#008700">In [</font><font color="#8AE234"><b>8</b></font><font color="#008700">]: </font>df1 + df2                                                               
<font color="#870000">Out[</font><font color="#EF2929"><b>8</b></font><font color="#870000">]: </font>
             b   c     d   e
Colorado  12.0 NaN  15.0 NaN
Ohio       0.0 NaN   3.0 NaN
Oregon     NaN NaN   NaN NaN
Texas      6.0 NaN   9.0 NaN
</pre>
算术填充
<pre><font color="#008700">In [</font><font color="#8AE234"><b>9</b></font><font color="#008700">]: </font>df1.add(df2,fill_value=<font color="#008700">0</font>)                                               
<font color="#870000">Out[</font><font color="#EF2929"><b>9</b></font><font color="#870000">]: </font>
             b    c     d     e
Colorado  12.0  7.0  15.0   8.0
Ohio       0.0  1.0   3.0   2.0
Oregon     9.0  NaN  10.0  11.0
Texas      6.0  4.0   9.0   5.0
</pre>
函数|作用
|-|-|
df.add()|加
df.sub()|减
df.div()|除
df.mul()|乘
#### DataFrame和Series之间的运算
<pre><font color="#008700">In [</font><font color="#8AE234"><b>11</b></font><font color="#008700">]: </font>arr = np.arange(<font color="#008700">12</font>).reshape((<font color="#008700">3</font>,<font color="#008700">4</font>))                                     

<font color="#008700">In [</font><font color="#8AE234"><b>12</b></font><font color="#008700">]: </font>arr                                                                    
<font color="#870000">Out[</font><font color="#EF2929"><b>12</b></font><font color="#870000">]: </font>
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])

<font color="#008700">In [</font><font color="#8AE234"><b>13</b></font><font color="#008700">]: </font>arr[<font color="#008700">0</font>]                                                                 
<font color="#870000">Out[</font><font color="#EF2929"><b>13</b></font><font color="#870000">]: </font>array([0, 1, 2, 3])

<font color="#008700">In [</font><font color="#8AE234"><b>14</b></font><font color="#008700">]: </font>arr-arr[<font color="#008700">0</font>]                                                             
<font color="#870000">Out[</font><font color="#EF2929"><b>14</b></font><font color="#870000">]: </font>
array([[0, 0, 0, 0],
       [4, 4, 4, 4],
       [8, 8, 8, 8]])
</pre>

DataFrame-Series

<pre><font color="#008700">In [</font><font color="#8AE234"><b>13</b></font><font color="#008700">]: </font>df1                                                                    
<font color="#870000">Out[</font><font color="#EF2929"><b>13</b></font><font color="#870000">]: </font>
            b     c     d
Ohio      0.0   1.0   2.0
Texas     3.0   4.0   5.0
Colorado  6.0   7.0   8.0
aaa       9.0  10.0  11.0
<font color="#008700">In [</font><font color="#8AE234"><b>14</b></font><font color="#008700">]: </font>series = df1.ix[<font color="#008700">0</font>]
<font color="#008700">In [</font><font color="#8AE234"><b>15</b></font><font color="#008700">]: </font>df1 - series                                                           
<font color="#870000">Out[</font><font color="#EF2929"><b>15</b></font><font color="#870000">]: </font>
            b    c    d
Ohio      0.0  0.0  0.0
Texas     3.0  3.0  3.0
Colorado  6.0  6.0  6.0
aaa       9.0  9.0  9.0
</pre>
DataFrame+Series
<pre><font color="#870000">Out[</font><font color="#EF2929"><b>15</b></font><font color="#870000">]: </font>
            b    c    d
Ohio      0.0  0.0  0.0
Texas     3.0  3.0  3.0
Colorado  6.0  6.0  6.0
aaa       9.0  9.0  9.0
<font color="#008700">In [</font><font color="#8AE234"><b>20</b></font><font color="#008700">]: </font>series2                                                                
<font color="#870000">Out[</font><font color="#EF2929"><b>20</b></font><font color="#870000">]: </font>
b    0
e    1
f    2
<font color="#008700">In [</font><font color="#8AE234"><b>23</b></font><font color="#008700">]: </font>df1 + series2                                                          
<font color="#870000">Out[</font><font color="#EF2929"><b>23</b></font><font color="#870000">]: </font>
           b     c     d   f
Ohio     NaN   1.0   3.0 NaN
Texas    NaN   4.0   6.0 NaN
Colorado NaN   7.0   9.0 NaN
aaa      NaN  10.0  12.0 NaN
</pre>
列上广播
<pre>            b     c     d
Ohio      0.0   1.0   2.0
Texas     3.0   4.0   5.0
Colorado  6.0   7.0   8.0
aaa       9.0  10.0  11.0

<font color="#008700">In [</font><font color="#8AE234"><b>27</b></font><font color="#008700">]: </font>series3                                                                
<font color="#870000">Out[</font><font color="#EF2929"><b>27</b></font><font color="#870000">]: </font>
Ohio         2.0
Texas        5.0
Colorado     8.0
aaa         11.0
Name: d, dtype: float64

<font color="#008700">In [</font><font color="#8AE234"><b>28</b></font><font color="#008700">]: </font>df1.sub(series3,axis=<font color="#008700">0</font>)                                                
<font color="#870000">Out[</font><font color="#EF2929"><b>28</b></font><font color="#870000">]: </font>
            b    c    d
Ohio     -2.0 -1.0  0.0
Texas    -2.0 -1.0  0.0
Colorado -2.0 -1.0  0.0
aaa      -2.0 -1.0  0.0
</pre>
#### 函数应用和映射
numpy的ufuncs函数也可以操作pandas对象
<pre><font color="#008700">In [</font><font color="#8AE234"><b>31</b></font><font color="#008700">]: </font>np.abs(frame)                                                                                                       
<font color="#870000">Out[</font><font color="#EF2929"><b>31</b></font><font color="#870000">]: </font>
          b         d         e
u  1.935225  0.275252  0.265381
o  2.420749  0.141706  0.056011
t  1.517044  0.141860  1.389832
e  0.403460  1.177187  0.336314</pre>

lambda
<pre><font color="#008700">In [</font><font color="#8AE234"><b>32</b></font><font color="#008700">]: </font>f = <font color="#008700"><b>lambda</b></font> x: x.max() - x.min()                                                                                     

<font color="#008700">In [</font><font color="#8AE234"><b>33</b></font><font color="#008700">]: </font>frame.apply(f)                                                                                                      
<font color="#870000">Out[</font><font color="#EF2929"><b>33</b></font><font color="#870000">]: </font>
b    4.355974
d    1.035481
e    1.726146
dtype: float64

<font color="#008700">In [</font><font color="#8AE234"><b>34</b></font><font color="#008700">]: </font>frame.apply(f,axis=<font color="#008700">1</font>)                                                                                               
<font color="#870000">Out[</font><font color="#EF2929"><b>34</b></font><font color="#870000">]: </font>
u    1.669844
o    2.562455
t    2.906876
e    1.580647
dtype: float64
</pre>
处理浮点型数字
<pre><font color="#008700">In [</font><font color="#8AE234"><b>35</b></font><font color="#008700">]: format</font> = <font color="#008700"><b>lambda</b></font> x:<font color="#AF5F00">&apos;</font><font color="#AF5F87"><b>%.2f</b></font><font color="#AF5F00">&apos;</font> %x                                                                                         

<font color="#008700">In [</font><font color="#8AE234"><b>36</b></font><font color="#008700">]: </font>frame.applymap(<font color="#008700">format</font>)                                                                                              
<font color="#870000">Out[</font><font color="#EF2929"><b>36</b></font><font color="#870000">]: </font>
       b     d      e
u   1.94  0.28   0.27
o  -2.42  0.14  -0.06
t  -1.52  0.14   1.39
e  -0.40  1.18  -0.34
<font color="#008700">In [</font><font color="#8AE234"><b>37</b></font><font color="#008700">]: </font>frame[<font color="#AF5F00">&apos;e&apos;</font>].map(<font color="#008700">format</font>)                                                                                              
<font color="#870000">Out[</font><font color="#EF2929"><b>37</b></font><font color="#870000">]: </font>
u     0.27
o    -0.06
t     1.39
e    -0.34
Name: e, dtype: object
</pre>
#### 排序和排名
index排序
<pre><font color="#008700">In [</font><font color="#8AE234"><b>41</b></font><font color="#008700">]: </font>obj = Series(<font color="#008700">range</font>(<font color="#008700">4</font>),index=[<font color="#AF5F00">&apos;d&apos;</font>,<font color="#AF5F00">&apos;a&apos;</font>,<font color="#AF5F00">&apos;b&apos;</font>,<font color="#AF5F00">&apos;c&apos;</font>])                                                                      

<font color="#008700">In [</font><font color="#8AE234"><b>42</b></font><font color="#008700">]: </font>obj.sort_index()                                                                                                    
<font color="#870000">Out[</font><font color="#EF2929"><b>42</b></font><font color="#870000">]: </font>
a    1
b    2
c    3
d    0
dtype: int64
</pre>
改轴就加个axis参数,降序,ascending=False

值排序
`注意版本.有些高版本貌似用series.sort_values()代替了series.oder()`
<pre><font color="#008700">In [</font><font color="#8AE234"><b>46</b></font><font color="#008700">]: </font>obj.sort_values()                                                                                                   
<font color="#870000">Out[</font><font color="#EF2929"><b>46</b></font><font color="#870000">]: </font>
2   -3
3    2
0    4
1    7
</pre>
Nan值都会被排到末尾

指定列排序
<pre><font color="#008700">In [</font><font color="#8AE234"><b>50</b></font><font color="#008700">]: </font>frame.sort_index(by=[<font color="#AF5F00">&apos;b&apos;</font>,])                                                                                         
/home/wangjuncheng/.local/bin/ipython:1: FutureWarning: by argument to sort_index is deprecated, please use .sort_values(by=...)
  #!/usr/bin/python3
<font color="#870000">Out[</font><font color="#EF2929"><b>50</b></font><font color="#870000">]: </font>
   b   c
1  1  17
3  2   7
0  3   5
2  7   3
</pre>
#### 排名(ranking)
P151 大概就是讲的是索引由某种计算过后的大小值吧
<pre><font color="#008700">In [</font><font color="#8AE234"><b>51</b></font><font color="#008700">]: </font>obj = Series([<font color="#008700">7</font>,-<font color="#008700">5</font>,<font color="#008700">6</font>,<font color="#008700">3</font>,<font color="#008700">6</font>,<font color="#008700">6</font>])                                                                                        

<font color="#008700">In [</font><font color="#8AE234"><b>52</b></font><font color="#008700">]: </font>obj.rank()                                                                                                          
<font color="#870000">Out[</font><font color="#EF2929"><b>52</b></font><font color="#870000">]: </font>
0    6.0
1    1.0
2    4.0
3    2.0
4    4.0
5    4.0
dtype: float64
</pre>
更多rankP151
#### 带有重复值的轴索引
<pre><font color="#008700">In [</font><font color="#8AE234"><b>53</b></font><font color="#008700">]: </font>obj = Series(<font color="#008700">range</font>(<font color="#008700">5</font>),index=[<font color="#AF5F00">&apos;a&apos;</font>,<font color="#AF5F00">&apos;a&apos;</font>,<font color="#AF5F00">&apos;b&apos;</font>,<font color="#AF5F00">&apos;b&apos;</font>,<font color="#AF5F00">&apos;c&apos;</font>])                                                                  

<font color="#008700">In [</font><font color="#8AE234"><b>54</b></font><font color="#008700">]: </font>obj                                                                                                                 
<font color="#870000">Out[</font><font color="#EF2929"><b>54</b></font><font color="#870000">]: </font>
a    0
a    1
b    2
b    3
c    4
dtype: int64

<font color="#008700">In [</font><font color="#8AE234"><b>55</b></font><font color="#008700">]: </font><font color="#5F8787"><i># 检测是否唯一</i></font>                                                                                                      

<font color="#008700">In [</font><font color="#8AE234"><b>56</b></font><font color="#008700">]: </font>obj.index.is_unique                                                                                                 
<font color="#870000">Out[</font><font color="#EF2929"><b>56</b></font><font color="#870000">]: </font>False

<font color="#008700">In [</font><font color="#8AE234"><b>57</b></font><font color="#008700">]: </font>obj[<font color="#AF5F00">&apos;a&apos;</font>]                                                                                                            
<font color="#870000">Out[</font><font color="#EF2929"><b>57</b></font><font color="#870000">]: </font>
a    0
a    1
dtype: int64

</pre>
对于DataFrame也是一样


