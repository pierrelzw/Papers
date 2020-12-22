# Evaluating Merging Strategies for Sampling-based Uncertainty Techniques in Object Detection

## 关键概念

- spatial affinity measures (3)
- semantic affinity measures (2)
- clustering method(4)
- Affinity-clustering combination



## 简介

- 本文贡献
  - 通过多次sampling来建模模型认知不确定度的方法，主要用于分类、分割和回归问题。本文首次进行了目标检测问题的模型认知不确定度建模
  - 本文建立了一个用于评估目标检测（定位+分类）不确定度的metric方法和流程，并发现了对机器人应用最有效的metric方法
  - 本文通过对比相似度衡量方法（3种位置、2种分类）和聚类方法（clustering)的组合，提出一种最适合机器人应用的affinity-clustering组合
- spatial affinity measures (3)
  - IOU(Intersection Over Union)
  - PAC(Product Association Cost) ：在tracking领域中已经被证明其有效性
  - EAC(Exponential Association Cost) ： 在tracking领域中已经被证明其有效性
- semantic affinity measures (2)
  - Same winner class Label(SL)：
  - KL divergence(KL) ： 用来衡量几个sampling结果的softmax输出分布之间的相似度

- Clustering methods

  - BSAS(Basic Sequential Algorithmic Scheme)
    - 对每个detection，如果它和cluster的affinity小于某个阈值，则把它归入该cluster。
    - 组合：IOU & SL
    - 组合：IOU & KL
  - BSAS excl. (with intra-exclusivity) 
    - Intra-sample clustering -> intra-sample exclusive
    -  IOU & KL

  - Hungarian Method匈牙利算法
    - 从第一个sampling结果(detections)初始化一个数量为n的cluster
    - <u>计算m个检测结果和n个cluster之间的cost 矩阵（？？）</u>
    - <u>新的cluster哪些还没有被分配cluster的detections中产生</u>
  - Hierarchial Density-Based Spatial Clustering of Applications with Noise（HDBSCAN）
    - <u>Density-based?</u>
    - require 2D inputs：使用bbox centroids， top-left corner coordinates，distance(corners， image boudary)
    - 
    - 