# Uncertainty-based object deteciton acitve learning

[TOC]

## 摘要

本文主要介绍几篇(目前是2)基于uncertainty来做目标检测active learning的文章。

整体来看，目前做2D检测active learning的论文，如果是uncertainty-based，基本只用了检测的分类信息(objectness和class)，没有使用regression信息（bbox）。

第二篇用了bbox信息，但是很多细节没讲清楚。基于分类结果建模uncertainty的方法，可参考[uncertainty-based classification active learning](https://iwiki.woa.com/display/~changeliu/3.+uncertainty-based+classification+active+learning)的论文。

## 1. Active Learning for Deep Detection Neural Networks

> ICCV2019：https://arxiv.org/pdf/1911.09168.pdf
>
> code(tensorflow)：www.gitlab.com/haghdam/deep_active_learning 

### Active Learning 

![image-20200825113126210](https://tva1.sinaimg.cn/large/007S8ZIlly1gi6hw3c1v8j312e09un0o.jpg)

### 文章概述

- 本文目的：以减少模型的FP（误检）和FN（漏检）为目的，实现uncertainty-based active Learning

- 假设

  对一张图片X，$x_1$ 是它的patch，$x_2$是$x_1$平移$\epsilon$之后的结果，是$x_1$的neighbornood。**如果X在训练集中被模型”见“过足够多的次数，我们可以认为，对$x_2$和$x_1$，检测网络会预测非常类似的概率分布结果。**

- 推论

  如果$x_1$和$x_2$的后验概率$p(x_1|w)$、$p(x_2|w)$差别很大，即$x_1, x_2$中存在FP或者FN，那么$x_1, x_2$所属的图片X需要被挑出来标注并参与新模型训练。此时我们说，$x_1, x_2$的divergence很大，divergence可以被用来建模uncertainty。用$D(\Theta(x_1)||\Theta(x_2))$表示divergence，其中$\Theta()$是softmax函数。

  

  由以上假设，我们可得：对FP和FN样本(图片patch)，它和neighborhood的D会变得非常大。

### uncertainty modeling

- **输入**
  - 本文主要使用1阶段检测网络，类似SSD，网络输出5个resolution的feature map，来预测object
    - 输入size=WxH的图像，输出5个feature map（论文中叫probability matrix），被记为$\Theta^{k}$
  - 本文使用objectness检测结果（5个用于回归最后结果的feature map），来计算uncertainty。对行人检测任务来说，这些feature map就是一个二分类结果。
  - **<u>本文不使用bbox 回归分支</u>**
- **输出**
  - 我们希望输出一个的size=WxH的score matrix S，每个元素$s_{ij}$表示的是它和neighborhood的预测结果之间的divergence。这个divergence可以直接等价于uncertainty。

- **如何计算uncertainty score？**

1. **pixel-level probability**

   记$p_{ij}^k$是第k个probability matrix $\Theta^k$在（i,j）位置上的元素。对(m,n)位置，取半径为r，计算neighborhood概率分布：

$$
\hat{p}_{ij}^k = \frac{1}{(2r+1)^2}\sum_{i=m-r}^{m+r}\sum_{j=n-r}^{n+r}p_{ij}^k
$$
​		其中，**r是neighborhood的半径**

2. **pixel-level uncertainty score**

   entropy方程：
   $$
   \mathbb{H}(z) = -z\log{z} - (1-z)\log{(1-z)}
   $$
   

   对第k个proba matrix，（m,n）处，我们用下列公式计算uncertainty score：

$$
s_{mn}^k = \mathbb{H}(\hat{p}_{mn}^k) - \frac{1}{(2r+1)^2}\sum_{i=m-r}^{m+r}\sum_{j=n-r}^{n+r}\mathbb{H}(p_{ij}^k)
$$
​		这个公式的本质是：neighborhood内probability平均值的熵，减去neighborhood内probility的熵的平均值<u>（Mutual Information？？）</u>



​		在做对比实验时，我们也采取MCDropout方法。在我们的行人检测实验中，objectness层输出是2分类，计算uncertainty score：
$$
s_{mn}^k = \mathbb{H}(\hat{p}_{mn}^k) ,  \\
s_{mn}^k = \mathbb{H}(\hat{p}_{mn}^k)  - \frac{1}{T}\sum_{t=1}^{T}\mathbb{H}(p_{mn}^k|w,q)
$$
​		其中，q是dropout distribution。**我们计算同一模型（多个resolution feature map层）在局部区域的预测结果的divergence而MCdropout则是通过T次inference，计算多模型在同一位置预测结果的divergence。**

​	

**3. Aggregation** 

​	最后的在**S**上的score为5个proba matrix之和：
$$
s_{mn} = \sum_{k=1...K_{\Theta}}s_{mn}^k
$$

​	从公式（3）可以看出，**<u>本文定义uncertaintyscore为平均值的entropy减去entropy的平均值</u>**。

​	我们希望：

- 如果某处的预测结果和它周围很相近，$s_{mn}^k$将会接近0。否则，则会很大。

- 如果在该位置，各个resolution下的预测一致的话最后$s_{mn}$会接近于0。

  <img src="https://tva1.sinaimg.cn/large/007S8ZIlly1gi6hw3t3y8j30sa0fuju2.jpg" alt="image-20200825135219676" style="zoom:33%;" />

  我们最后希望有一个image-level的score，用于在active learning中对图像进行重要性排序，重要的图片，需要优先标注。如上图所示，我们**把图片分为互不重叠的$D_p$块区域**，定义image-level score 为z，则：

$$
z = \frac{1}{D_p}\sum_i{s_{max}^i}
$$

### 实验结果

TODO

### 一些疑问

- Mutual information作为uncertainty的物理含义（来源：分类uncertainty 建模论文，待继续研究）
- 把pixel-level聚合为image-level uncertainty时，直接除以像素数量，这意味着物体越多，uncertainty必然就越高，除以物体数量比较合适？（待验证）
- 

###



## 2. Deep Active Learning for Remote Sensing Object Detection

> https://arxiv.org/abs/2003.08793

### 简介

本文介绍了一种新的基于不确定度的目标检测主动学习方法。本文关注遥感目标检测问题，**不仅考虑了分类不确定度，还考虑了回归不确定度。**

同时，本文还提出额外的加权系数，来解决遥感目标检测任务中的两大问题：类别不同和类别不均衡。（遥感图像非常大，需要剪切下来做检测。有的时候很多车聚集在一块区域，这样的区域物体很多）



### 不确定度建模

#### 分类不确定度

- class-imbalance问题

  - 遥感图像中有很多物体，比如车这种物体，非常小且很密集
  - 传统的uncertainty-based active learning倾向于挑选拥有更多物体的图片（结论来源？），这将导致有些类别的数量会越来越多，比如车，而其他比较少的类别则容易被忽略。

- 解决方案
  - 未解决class-imbalance的影响，我们设定一个类别权重：

  $$
  W_i^1 = \log_{10}^{n_i}
  $$

  		其中，$n_i$s是每个类别的样本数数量。

  - 在单张图像中的物体数量分布也是应该考虑的因素，所以每个类别平均数据/图也是需要考虑的因素，与之对应的权重是：

  $$
  x_i = \frac{n_i}{p_i} \\
  sum = \sum_{i=1}^cx_i \\
  W_i^2 = (sum+C)/(x_i+1)
  $$

  		其中，$p_i$指的是包含类别i的图像数量，作为分母$p_i\geq 1$ 

- 分类不确定度计算：

$$
U_c = \sum{W_i^1*W_i^2*(1-P)}
$$

		其中P是<u>least confidence</u>（**最高conf？最低？**）

### 回归不确定度

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

### 分类+回归加权不确定度 WCR（weighted classification-regression）

$$
U_S = \sum(U_C*U_R)
$$

图片是active learning中的最小单元，我们需要计算一张图片的uncertainty。本文定义图片的uncertainty就是图片中所有物体uncertainty之和。每个物体的uncertainty，为分类uncertainty和回归uncertainty的乘积。



### 一些疑问

- 【conclusion】如何用分类和回归不确定度的weights来decrease imbalance from the dataset？

- 【conclusion】考虑回归不确定度，并且努力去选择那些有重要物体，但是没有outliers的图片，如何做到？

- 【regression uncertainty】用GMM计算bbox的概率密度，how？超参数-99，-10都是什么含义，为什么取这些值？

  

## 3. Deep active learning for object detection

> BMVC2018：http://bmvc2018.org/contents/papers/0287.pdf

TODO

## 4. Active Learning for Deep Object Detection

>  https://arxiv.org/pdf/1809.09875.pdf

TODO 