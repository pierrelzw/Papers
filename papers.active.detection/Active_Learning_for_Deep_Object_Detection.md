# Active Learning for Deep Object Detection

## 简介

本文提出了几种基于不确定度(uncertainty-based)的目标检测active learning方法。他们只需要检测网络中的分类结果即可。对特定的任务，objectness也很重要。针对imbalance问题，本文提出一种sample weighting方法来解决。

我们在Pascal VOC数据上实验发现，基于SUM建模uncertainty的效果最好，但是这种方式趋向于把那些只有一个物体的图片挑出来。这与实际应用不符，因为实际场景有很多物体会出现在图像里。

对一些特定的任务，OoD检测也很重要。

本文提供的主动学习方法，适应于几乎所有的目标检测算法，因为它只要求softmax分类结果(classification score distribution)。





