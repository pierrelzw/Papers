# Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles

## 要点

- 应该是deep ensembles for uncertainty estimation的开山之作
- 

## 摘要



## 问题建模



## 方案 recipe

1) 使用一个合适的score function来作为训练的criterion（目标）

2）使用adversarial training对抗训练来平滑预测分布（prediction distribution）

3)  训练一个ensemble



### 训练ensemble

训练ensemble主要有两大类方法：random-based和boosting-based方法。

- boosting：渐进式的
- random：多个模型同步训练，相互没有interaction(？？？)，适合分布式，并行计算
  - 有论文证明【8】，通过某些方式可以把random forest拆开，使得用几个独立的tree模型，也能达到近似forest 模型的效果
  - 一种典型的方法就是：bagging。但是这不适用NN

 <img src="/Users/lizhiwei/Library/Application Support/typora-user-images/image-20201021204132206.png" alt="image-20201021204132206" style="zoom:50%;" />

我们发现bagging对我们的实验有害。

我们使用全量数据，因为NN总是希望有更多的数据（效果更好）。

我们把ensemble当作是一个均权的混合模型。对分类任务，它就是预测prob的平均。对回归任务，预测结果就是一个混合高斯模型。其中mean和variance就是混合的mean和variance



## 实验结果和结论

### 回归任务

<img src="/Users/lizhiwei/Documents/papers.active.unceratinty/image-20201021205009182.png" alt="image-20201021205009182" style="zoom:50%;" />

使用非常简单的回归任务数据做实验，第四个M=5 ensemble的结果。可以看出，在看过的数据点范围内，NLL非常小，但是在没有看过的数据点范围上，NLL就很大，这正是我们期待的结果。

<img src="/Users/lizhiwei/Documents/papers.active.unceratinty/image-20201021205342038.png" alt="image-20201021205342038" style="zoom:50%;" />

从具体数据上看，deep ensembles在NLL上去的最好的效果，部分RMSE的效果不是最好的，但是这可能是因为deep ensemble重点达到更低的NLL。



### classification

<img src="/Users/lizhiwei/Documents/papers.active.unceratinty/image-20201021205514474.png" alt="image-20201021205514474" style="zoom:50%;" />

用MNIST，SVHN数据集做实验，可以看出：

- ensemble+AT的效果基本是最好的
- ……



<img src="/Users/lizhiwei/Documents/papers.active.unceratinty/image-20201021205606193.png" alt="image-20201021205606193" style="zoom:50%;" />

当我们测试known class和unknown class的entropy时，我们会发现：

- ensemble的network越多，在测试unknown class时，效果更理想
- ensemble的network增多，不影响known class的entropy结果

<img src="/Users/lizhiwei/Documents/papers.active.unceratinty/image-20201021205806678.png" alt="image-20201021205806678" style="zoom:50%;" />

- 从数值上上也是类似的结论

- 在imageNet上测试，不随着ensemble net的数增加，效果逐渐变得理想

<img src="/Users/lizhiwei/Documents/papers.active.unceratinty/image-20201021205902216.png" alt="image-20201021205902216" style="zoom:50%;" />

从accuracy可以看出，MC dropout存在overconfident wrong prdiction问题（曲线不是递增的）

