# Deep Active Learning for Remote Sensing Object Detection

## 简介

本文介绍了一种新的基于不确定度的目标检测主动学习方法。本文关注遥感目标检测问题，我们不仅考虑了分类不确定度，还考虑了回归不确定度。

同时，本文还提出额外的加权系数，来解决遥感目标检测任务中的两大问题：类别不同和类别不均衡。（遥感图像非常大，需要剪切下来做检测。有的时候很多车聚集在一块区域，这样的区域物体很多）

## 未解决的问题

- 【conclusion】如何用分类和回归不确定度的weights来decrease imbalance from the dataset？

- 【conclusion】考虑回归不确定度，并且努力去选择那些有重要物体，但是没有outliers的图片，如何做到？
- 【regression uncertainty】用GMM计算bbox的概率密度，how？超参数-99，-10都是什么含义，为什么取这些值？



## 不确定度建模

### 分类不确定度

- class-imbalance问题

  - 遥感图像中有很多物体，比如车这种物体，非常小且很密集
  - 传统的uncertainty-based active learning倾向于挑选拥有更多物体的图片（结论来源？），这将导致有些类别的数量会越来越多，比如车，而其他比较少的类别则容易被忽略。

- 解决方案
  - 未解决class-imbalance的影响，我们设定一个类别权重：

  $$
  W_i^1 = \log_{10}^{n_i}
  $$

  ​		其中，$n_i$s是每个类别的样本数数量。

  - 在单张图像中的物体数量分布也是应该考虑的因素，所以每个类别平均数据/图也是需要考虑的因素，与之对应的权重是：

  $$
  x_i = \frac{n_i}{p_i} \\
  sum = \sum_{i=1}^cx_i \\
  W_i^2 = (sum+C)/(x_i+1)
  $$

  ​		其中，$p_i$指的是包含类别i的图像数量，作为分母$p_i\geq 1$ 

- 分类不确定度计算：

$$
U_c = \sum{W_i^1*W_i^2*(1-P)}
$$

​		其中P是<u>least confidence</u>（**最高conf？最低？**）

## 回归不确定度

对每个bbox，有长边$L = max(m,n)$和短边$S = min(m,n)$。

<u>本文使用这两个参数来计算bbox的概率分布密度（how？）</u>

当用有限的数据来训练网络时，我们应该关注主要物体（main object part）。因为就提升mAP而言，提升占大多数的物体的检测性能的效果比提升占少数的物体的检测性能更有用。

在有同样的分类不确定情况下，有着更大概率密度的物体，是我们需要选择的。

<u>回归不确定度正比于bbox的概率分布密度</u>



我们选择Gaussian Mixture Model(GMM)来分析bbox的概率分布密度函数。

首先，我们计算每个bbox的log概率L，<u>然后我们取L<-99的部分</u>(**为什么是-99？**)，最后我们计算回归不确定度$U_r$:
$$
U_r = 0.05 * (L_b + 10) + 0.5,\  L_b \geq -10 \\
U_r = 0.5 * \frac{L_b+100}{90}, \ L_b \leq -10\
$$

## WCR（weighted classification-regression）

$$
U_S = \sum(U_C*U_R)
$$

图片是active learning中的最小单元，我们需要计算一张图片的uncertainty。本文定义图片的uncertainty就是图片中所有物体uncertainty之和。每个物体的uncertainty，为分类uncertainty和回归uncertainty的乘积。



