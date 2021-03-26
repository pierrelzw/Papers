# Dropout Sampling for Robust Object Detection in Open-Set Conditions

## Some concept

- 3 classes in openset object detection

  - Knows classes：即训练集中包含的类别
  - Known unknown classes：有些模型被训练成可以预测背景类，或者通过score threshold预测”其他类“
  - Unkown unkown classes ：没有在训练集出现过，但是被检测到的物体/类别

  举例来说，用常用的目标检测算法训练coco数据集得到一个模型，这个模型输出的就是Known classes。我们可以通过score阈值，把一些score比较低的预测直接归类为”unknown“，这就是Known unknown classes；当然也可以训练一个专门输出unknown classes的。前两种都是可能通过训练模型得到， 但是第三种，在模型预测阶段，没有办法知道。

  

## Intro



## Evaluation Metrics

- openset-error : 误检物体的数量。误检物体“和真值重叠度不够高、但是预测不是unknown“的物体。

  > 这类物体实际上是unknown object($IOU_{gt}<0.5$且类别预测为已知类别) ，但是没有被预测为‘unknown’。

  ` 这个指标可衡量模型对unknown-object的稳健性（误检），我们希望对所有已知类别物体的预测/观测，openset-error为0`

  

- precision：检测到了$IOU_{gt}>0.5$，并且分类对了

  ` 这个指标可衡量模型how well a detector classify unknown  and known objects`

  

- recall : 检测到了$IOU_{gt}>0.5$，但是分类错了

  `这个指标可衡量模型how well a detector classify known objects`

  

>所以这里只考虑分类？怪不得结论说，future work会考虑spatial uncertainty









