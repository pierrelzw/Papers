# detr



## Loss

过去的detector大多用L1，L2 loss。最常用的是smooth_l1 loss，会因为bbox scale的不同导致结果差异，尽管这个差异相对比较小。所以我们使用了l1 loss和GIOU loss的加权平均作为最终的loss。由此，我们引入两个超参数。



