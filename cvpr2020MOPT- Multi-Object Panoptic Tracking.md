# MOPT

<img src="cvpr2020MOPT- Multi-Object Panoptic Tracking.assets/image-20210322114817422.png" alt="image-20210322114817422" style="zoom:50%;" />

## 摘要

本文提出panopticTrackNet archtecture, 同时输出semantic segmentation, instance segmentation以及instance tracking结果。

## 问题



## sematic segmentation head

- arch : 

- loss : the weighted per-pixel log-loss

  $$L_{semantic}(\Theta) = -\frac{1}{n}\sum\sum_{i,j}w_{ij}logp_{ij}(p_{ij}^*)$$

## instance segmentation head

- arch : Mask RCNN

  - RPN :  RPN网络用一个2-way的FPN feature作为输入，产出box proposal
  - ROI_Align则以每个proposal为输入，用两个分支分别输出box&classification和instance segmentation mask。
    - 第一个分支由下面的网络构成
      - 4个output filter为256的3x3的*separable conv*
      - 1个output filter=256的2x2 stride=2的transposed conv
      - 1个1x1output filter=N_thing的conv
      - 对每个class，这个branch产出28x28的mask logit
    - 第二个分支由下面的网络构成
      - 2个channel=1024的全连接网络
      - 一个输出为4xN_thing的网络层用于box regression
      - 一个输出为N_thing+1(background)的网络层用于classification

- loss :  

  $$L_{instance} = L_{os} + L_{op} + L_{cls} + L_{bbx} + L_{mask}$$

  os : object score

  op : proposal

  cls : classification

  bbx : bouding box

  mask : mask segmentation

## Instance Tracking Head

- arch

  为了更好地利用instance head的结果，包含ROIAlign Feature, predicted class, masks logits,

  - 用max-pooling把mask logits下采样2倍，使其resolution match ROIAlign Output
  - pooling之后得到256d的feature vector，然后分别连接两个128和32xN_thing长度的全连接层。
  - 对每个segmentation $s\in S$, 我们得到一个t frame内的association vector  $a_s^c$

- loss(the batch triplet loss with margin $\alpha$)

  $$L_{track} = \frac{1}{|S|} \sum_{s\in S} max(max_{e\in S}||a_s^{c, \phi} - a_e^{c,\phi}|| - min_{e\in S} ||a_s^{c, \phi} - a_e^{c,\hat\phi}|| + \alpha, 0)$$

  $\phi$ : trackID

  

