# CentripetalNet: Pursuing High-quality Keypoint Pairs for Object Detection



<img src="/Users/lizhiwei/Library/Application Support/typora-user-images/image-20200623114729270.png" alt="image-20200623114729270" style="zoom:50%;" />

Centripetal即向心的，和cornernet，Centernet（keypoints) 不同，CentripetalNet在角点预测的基础上，还预测了角点对中心的shift，这将极大地帮助CentripetalNet进行角点grouping (or matching) ，从而提高准确率。



## corner pooling from CornerNet

<img src="/Users/lizhiwei/Library/Application Support/typora-user-images/image-20200623115054070.png" alt="image-20200623115054070" style="zoom:40%;" />

<img src="/Users/lizhiwei/Library/Application Support/typora-user-images/image-20200623115116054.png" alt="image-20200623115116054" style="zoom:45%;" />

取feature map上x和y方向上的max value，分别生成新的feature maps，把两个方向的feature map相加，得到corner pooling结果。需要注意的是，这里取的不是一整行的max，而是取固定方向的max，比如topleft点的预测，取的就是往右看和往下看的max，rightbottom的预测，就是取往左看和往上看的max。

怎么实现呢？看了源码，就是用c++实现，然后python调用，但是，这本质一维pooling？不是。



## Corner matching

<img src="https://tva1.sinaimg.cn/large/007S8ZIlly1gg241o4m64j30yo0lcwj3.jpg" alt="image-20200623122410249" style="zoom:40%;" />

定义$R_{central}$ 为bbox真值中心点附近，大小为$bbox*u$的区域，corner+Centripetal落在这个区域，则匹配上.

## KeypointCenterNet 对cornerNet的改进

- Center Pooling

  <img src="/Users/lizhiwei/Library/Application Support/typora-user-images/image-20200623122223137.png" alt="image-20200623122223137" style="zoom:50%;" />

  这段不清晰，需要再仔细看看

- Cascade Corner Pooling

<img src="/Users/lizhiwei/Library/Application Support/typora-user-images/image-20200623122110441.png" alt="image-20200623122110441" style="zoom:50%;" />

