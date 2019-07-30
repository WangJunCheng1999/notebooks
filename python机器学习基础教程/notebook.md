## 基础
`基本的数据都在　sklearn.datasets里`
加载数据
```python
from sklearn.datasets import load_iris   # 载入数据
iris_dataset = load_iris()
```
得到对面的keys
```python
iris_dataset.keys()
dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])
```
其中
>Descr:是对数据集进行简单说明
>target_names:赌赢的是一个字符串数组,里面是包含我们预测的结果
```python
Target names: ['setosa' 'versicolor' 'virginica']
```
>feature_names:键对应的值是一个字符串列表，对每一个特征进行了说明：
```python
print("Feature names: \n{}".format(iris_dataset['feature_names']))
Feature names:
['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
 'petal width (cm)']
```
> target:样本的输出结果集合
>data:样本的特征集合

```python
In[15]:
print("Type of data: {}".format(type(iris_dataset['data'])))
Out[15]:
Type of data: <class 'numpy.ndarray>
#data 数组的每一行对应一个样本，列代表样本的特征测量数据
In[16]:
print("Shape of data: {}".format(iris_dataset['data'].shape))
Out[16]:
Shape of data: (150, 4)
```
`机器学习中的个体叫作样
本（sample），其属性叫作特征（feature）。data 数组的形状（shape）是样本数乘以特征
数。这是 scikit-learn 中的约定，你的数据形状应始终遵循这个约定`

`一部分数据用于构建机器学习模型，叫作训练数据（training data）或训练
集（training set）。其余的数据用来评估模型性能，叫作测试数据（test data）、测试集（test 
set）或留出集（hold-out set）。
`

`scikit-learn 中的 train_test_split 函数可以打乱数据集并进行拆分。这个函数将 75% 的
行数据及对应标签作为训练集，剩下 25% 的数据及其标签作为测试集。训练集与测试集的
分配比例可以是随意的，但使用 25% 的数据作为测试集是很好的经验法则
`
### 划分训练集和测试集
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
 iris_dataset['data'], iris_dataset['target'], random_state=0)
```

为了确保多次运行同一函数能够得到相同的输出，我们利用 random_state 参数指定了随机
数生成器的种子。这样函数输出就是固定不变的，所以这行代码的输出始终相同。本书用
到随机过程时，都会用这种方法指定 random_state。

### 第一个预测模型
```python
# 导入
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)# 设置邻居数量
knn.fit(X_train, y_train) # 给数据
# 到这里模型已经建好了,代入数据就行

X_new = np.array([[5, 2.9, 1, 0.2]])

prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(
        iris_dataset['target_names'][prediction]))
Out[28]:
Prediction: [0]
Predicted target name: ['setosa']

#　评估模型
y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))
Out[29]:
    Test set predictions:
    [2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 1 1 0 0 2 1 0 0 2 0 0 1 1 0 2 1 0 2 2 1 0 2]
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
Out[30]:
    Test set score: 0.97
#　我们还可以使用 knn 对象的 score 方法来计算测试集的精度：
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
Out[31]:
    Test set score: 0.97
```
`scikit-learn的输入数据必须是二维数组`

|算法|导入|
--|--
k 近邻分类算法|from sklearn.neighbors import KNeighborsClassifier