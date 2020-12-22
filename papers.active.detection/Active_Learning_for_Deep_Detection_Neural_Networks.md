# Active Learning for Deep Detection Neural Networks

问题：

- 用来给image打分的score function 是什么？

  - 使用什么输入？

    - 本文主要使用objectness检测结果，对行人检测任务来说，就是一个二分类层。
    - **<u>本文不使用bbox 回归分支</u>**

  - 如何计算？

    - 使用一个类似SSD的网络，输出5个resolution的feature map，来预测object

    - 输入size=WxH的图像，输出5个feature map（论文中叫probability matrix），被记为$\Theta^{k}$。

    - 我们希望计算一个的size=WxH的score matrix S，每个元素$s_{ij}$表示的是它和neighborhood的预测结果divergence。

    - 记$p_{ij}^k$是第k个probability matrix$\Theta^k$在（i,j）位置上的元素，我们的第一步就是计算（m，n）处的概率分布：
      $$
      \hat{p}_{ij}^k = \frac{1}{(2r+1)^2}\sum_{i=m-r}^{m+r}\sum_{j=n-r}^{n+r}p_{ij}^k
      $$
      其中，**r是neighborhood的半径**

    - 此时（m,n）处对第k个proba matrix的score为：
      $$
      s_{mn}^k = \mathbb{H}(\hat{p}_{mn}^k) - \frac{1}{(2r+1)^2}\sum_{i=m-r}^{m+r}\sum_{j=n-r}^{n+r}\mathbb{H}(p_{ij}^k)
      $$
      其中$\mathbb{H}$是entropy方程：
      $$
      \mathbb{H}(z) = -z\log{z} - (1-z)\log{(1-z)}
      $$
      
      在做对比实验时，我们也采取MCdropout方法，此时，对2类entropy，
         $s_{mn}^k = \mathbb{H}(\hat{p}_{mn}^k)$ , 
      
        $s_{mn}^k = \mathbb{H}(\hat{p}_{mn}^k)  - \frac{1}{T}\sum_{t=1}^{T}\mathbb{H}(p_{mn}^k|w,q)$
      
        其中，q是dropout distribution。与我们不同的是，我们计算局部区域的divergence，MCdropout则是通过T次inference，计算同一位置的divergence。
      
    - 最后的在**S**上的score为5个proba matrix之和：

$$
s_{mn} = \sum_{k=1...K_{\Theta}}s_{mn}^k
$$

- 从（3）可以看出，**<u>本文定义score为平均值的entropy减去entropy的平均值</u>**。事实证明，$s_{mn}^k$将会接近0，如果某处的预测结果和它周围很相近。否则，则会很大。
  
    - 最后$s_{mn}$会接近于0，如果在该位置，各个resolution下的预测一致的话
    
    - aggregating score

      <img src="https://tva1.sinaimg.cn/large/0081Kckwly1glwtcu3upsj30sa0fuju2.jpg" alt="image-20200825135219676" style="zoom:33%;" />

      我们最后希望有一个image-level的score，用于在active learning中对图像进行重要性排序，重要的图片，需要优先标注。如上图所示，我们**把图片分为互不重叠的$D_p$块区域**，定义image-level score 为z，则：
  $$
      z = \frac{1}{D_p}\sum_i{s_{max}^i}
  $$
  
- selecting  images
  
  - 对时序无关的图片集，按照score排序，挑出最高score的图标注
    
      - 对时序相关的图片集，比如video，则要考虑连续帧之间的相似性和冗余
      
        - 如果$t^{th}$帧被选中了，则$t^{th} - t^{th}+\Delta{t_1}$帧，在**当前activelearning迭代**中不会被选中
      
        - 如果$t^{th}$帧被选中了，则$t^{th} - t^{th}+\Delta{t_2}$帧，在**下一个active learning迭代**中不会被选中
      
          其中$\Delta{t_1} > \Delta{t_2}$

## 简介

![image-20200825113126210](https://tva1.sinaimg.cn/large/0081Kckwly1glwtcurcs4j312e09un0o.jpg)

- 假设

  - 对一张图片X，$x_1$ 是它的patch，$x_2$是$x_1$平移$\epsilon$之后的结果。如果X在训练集中被模型”见“过足够多的次数，我们可以认为，对$x_2$和$x_1$，检测网络会预测非常类似的概率分布结果
  - 也就是说，如果$x_2$和$x_1$和后验概率$p(x_1|w)$、$p(x_2|w)$会差别很大，导致divergence变大。用$D(\Theta(x_1)||\Theta(x_2))$表示他们的divergence，其中$\Theta()$是softmax函数
  - 我们假设：对FP和FN样本，它和它的neighborhood的D会变得非常大。

- 本文目的是：减少FP（误检）和FN（漏检）

  

## 关键概念



