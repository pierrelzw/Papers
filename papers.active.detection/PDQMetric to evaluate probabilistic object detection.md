# PDQ:Metric to evaluate probabilistic object detection

## 摘要



## 解决问题(为什么？)

​	mAP的一些问题

- score阈值越低，mAP越大，而实际使用detector时，我们会设置一个相对较高的threshold，所有mAP计算时有一部分结果是我们完全不关注的

- 衡量accuracy时，没有对前景背景作区分。比如下面这种情况，如果预测结果是橙色框，iou也能到0.5（这一点我存疑，如果只检测到下半部分，pdq会不会很高？我们还是希望整体iou更高的。极端一点的例子，预测值是真值，iou=1，预测值top降低一些，pdq可能增大，而iou减小，更高iou阈值下的accuracy比pdq更具参考性？）

  <img src="https://tva1.sinaimg.cn/large/0081Kckwly1glgk9zoo26j30qw0aoq3y.jpg" alt="image-20200709104838875" style="zoom:50%;" />

## POD(probabilistic object detection)

POD VS conventional OD （for each known object in an Image）：

- conventional OD：class distribution && bbox(x1, y1, x2, y2)
- POD ： class distribution && bbox $\mathcal{B}= \{\mathcal{N}_0, \mathcal{N}_1\} = \{\mathcal{N}(\mu_0, \Sigma_0),\  \mathcal{N}(\mu_1, \Sigma_1)\}$  $\mu_i, \Sigma_i$ 分别代表bbox的top-left, right-bottom 角点均值和方差



## PDQ是什么

- PDQ将惩罚哪些输出低spatial uncertainty而实际上把forground和background互相预测错误的结果。
- PDQ没有超参数用来定义什么是sucess detection（e.g. IOU threshold)
- Ground Truth ： $\mathcal{G}_i^f = \{\hat{\mathcal{S}}_i^f, \hat{\mathcal{B}}_i^f, \hat{\mathcal{c}}_i^f\}$ = {分割标注，bbox标注，class 标注}
- Detection：$\mathcal{D}_j^f = \{P(x \in \mathcal{S}_j^f), \mathcal{S}_j^f), \mathcal{I}_j^f\} $
- PDQ可以用来验证上面提到的POD，或者是conventional OD （我们认为bbox内的score为$1-\epsilon$ , bbox外的score是$\epsilon$ (很小)。

## 如何使用PDQ

<img src="PDQMetric to evaluate probabilistic object detection.assets/image-20201208194556070.png" alt="image-20201208194556070" style="zoom:50%;" />

### spatial quality

$$
Q_s(\mathcal{G}_i^f, \mathcal{D}_j^f) = exp(-(L_{FG}(\mathcal{G}_i^f, \mathcal{D}_j^f) + L_{BG}(\mathcal{G}_i^f, \mathcal{D}_j^f)))
$$

$$
L_{FG}(\mathcal{G}_i^f, \mathcal{D}_j^f) = - \frac{1}{|\hat{\mathcal{S}}_i^f|} \sum_{x \in \mathcal{S}_i^f}\log{P(x \ \in \mathcal{S}_j^f)}
$$

$$
L_{BG}(\mathcal{G}_i^f, \mathcal{D}_j^f) = - \frac{1}{|\hat{\mathcal{S}}_i^f|} \sum_{x \in \mathcal{V}_{ij}^f}\log{1- P(x \ \in \mathcal{S}_j^f)}
$$

其中 $\mathcal{V}_{ij}^f = \{\mathcal{S}_j^f - \hat{\mathcal{B}}_i^f\} = \{\mathcal{S}_j^f \or  \hat{\mathcal{B}}_i^f\} -  \{\mathcal{S}_j^f \and  \hat{\mathcal{B}}_i^f\} $   指的是属于检测到但是不属于真值的部分， 如下图所示：

<img src="PDQMetric to evaluate probabilistic object detection.assets/image-20201208195815345.png" alt="image-20201208195815345" style="zoom:50%;" />

## label quality

$$
Q_L (\mathcal{G}_i^f, \mathcal{D}_j^f) = I_j^f(\hat{\mathcal{c}}_i^f)
$$

## pPDQ


$$
pPDQ(\mathcal{G}_i^f, \mathcal{D}_j^f) = \sqrt{Q_S * Q_L}
$$
以上，我们得到了对任意真值 $\mathcal{G}_i^f$ 和一个object detection结果$\mathcal{D}_j^f$， 该pair的pPDQ。通过对pPDQ排序即可得到最优的detection assignment。（与mAP计算类似，pPDQ同样可以保证1个detection，有且只有一个GT与其匹配）

有了pPDQ之后，我们可以通过Hungarian算法得到N帧图像TP :  $N_{TP}^f$ ,这里TP与计算AP时的TP不同，这里TP指的是能得到最优pPDQ match且pPDQ >0 的detection。如果一个det & GT的pPDQ == 0，说明漏检FN或者误检FP，他们的总数分别记为 $N_{FN}^f$和 $N_{FP}^f$  其中f为当前帧frame。

最后：
$$
PDQ(\mathcal{G}, \mathcal{D}) = \frac{1}{\sum_{f=1}^{N_F} N_{TP}^f + N_{FP}^f + N_{FN}^f} \sum_{f=1}^{N_F}\sum_{i=1}^{N_{TP}^f}q^f(i)
$$
其中$q^f(i)$是第f帧中，第i-th个被assign为TP的det和对应GT的pPDQ score。



### 怎么验证PDQ

- 实验
  - 对比了传统detector和proba-based detector[29]在不同指标下的结果
  - 对传统dectector，直接把预测框内所有像素的proba设为1-eps，框外的像素的proba为$eps=10^{-14}$
- 结论
  - pdq能够明显区分传统detector和proba-based detector
    - 我们发现yolov3 map高，但是pdq很差->yolo更加擅长理解物体是什么(what an object is)，但是在预测哪里有物体(where that object is)这件事上，yolo的结论一定可靠（？？？这个insight，有点难说服我，从这个角度看ssd很差，甚至更差）
    - 