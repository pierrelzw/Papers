# Uncertainty Estimation in DL(CV)
[TOC]

## 1. 为什么研究uncertainty？

> Know what you don't know

<img src="https://i.loli.net/2020/07/31/HtbKDj9i62JCMPA.png" alt="image-20200730223541402" style="zoom:30%;" />

训练好的神经网络（NN）模型，本质是一个拥有大量确定参数的函数，不管你给什么输入，它都能给你一个输出。这会导致两种我们不愿意看到的意外情况：

- 1）对明明错误的预测结果，模型输出的置信度却很高
- 2）对没有见过的输入(OoD，Out-of-ditribution)，比如给一个识别猫狗的模型输入一张桌子图片，模型一定会输出：”这是猫“ or “这是狗”。

所以，我们希望模型能输出uncertainty，告诉我们它对预测结果的确定程度。比如上面的例子中，我们希望对错误分类的样本、OoD样本，模型说”我不确定“——给出一个较高的uncertainty。以便我们可以拒绝这样的预测结果。

更进一步地，我们知道有很多问题都对Uncertainty有需求：

- 在自动驾驶系统中，我们不仅希望深度学习模型告诉我们前面有个人，还希望它告诉我们它对这个结果的信心是多少？

- 在使用AI进行辅助诊断时，医生希望AI告知结果时同时告诉他，它对这个结果有多确定，这样他就可以只关注AI不很确定(高uncertainty)的结果，从而减少工作量。

- **主动学习(Active Learning)中，我们想知道哪些数据应该被标注？**(high uncertainty data)

- OoD(Out-of-distribution) detection

- 深度学习可解释性

- 在统计学习中，我们经常会做显著性检测，其实就是uncertainty检验

- ……

## 2. 什么是uncertainty？

<img src="https://tva1.sinaimg.cn/large/007S8ZIlly1ghevtq0n46j30h9081my3.jpg" alt="img" style="zoom:60%;" />

参考NIPS2017年的论文 [What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? ](https://papers.nips.cc/paper/7141-what-uncertainties-do-we-need-in-bayesian-deep-learning-for-computer-vision.pdf) ，Gal阐述了两种uncertainty：Aleatoric uncertainty(i.e. data uncertainty) 和 Epistemic uncertainty(i.e. model uncertainty)，即随机不确定度(也称数据不确定度)，和认知不确定度(也称模型不确定度)。

比如下图的例子,（d）(e)分别是数据不确定度和模型不确定度，越亮代表uncertainty越高。

<img src="https://camo.githubusercontent.com/0900f774997bc2cccbac537a7c14ea568232256d/68747470733a2f2f747661312e73696e61696d672e636e2f6c617267652f30303753385a496c6c793167683937767732796e646a33313532306b61646d322e6a7067" alt="img" style="zoom:40%;" />

可以看出：

- aleatoric unceratinty主要出现在物体边缘和远处(d)。该uncertainty源于数据本身，主要是标注员对物体边缘标注的精度不够、远处物体成像质量较差导致。
>measure what you can't understand from data : 上图中标注不好的地方（边缘和远处）。我们无法从标注不够精确的数据中，学习出一个”可以预测精细的物体轮廓“的模型）

- Epistemic uncertainty主要出现在model预测不好的地方。比如最后一行，模型对人行道的分割结果较差(c)，所以Epistemic uncertainty比较高(e)
> measure what your model doesn't know ：上图中模型出现FP的地方

Epistemic uncertainty可以通过增加数据解决，比如下图：只有一个data point的时候，符合要求的模型有很多种可能，uncertainty很高。当数据点增加，模型逐渐确定，uncertainty减小。
<img src="https://i.loli.net/2020/07/31/qvC8ea2KASTyzkM.png" alt="image-20200731000751967" style="zoom:40%;" />
Aleatoric uncertainty 其实就是训练数据中的噪声，来源于数据收集/标注过程。这些噪声是随机的，而且是固定的。噪声越多，数据的不确定度越大。它可以被测量，但是无法通过增加数据减小。
- Heteroscedastic Aleatoric Uncertainty # TODO
- Homoscedastic Aleatoric Uncertainty # TODO

**Related work**

基于目前的调研，研究深度学习(DL) uncertainty就是研究Bayesian Neural Network(BNN)。根据最近2周(20200719-20200804)的调研，学界、业界对**DL尤其是CV uncertainty**的研究，相比classification、detection这些任务的研究，少得多。

- 从学术会议看，研究uncertain的AI顶会主要有NIPS, ICLR, ICML，机器人顶会ICRA和医疗图像处理会议MICCAI

- 从具体领域看，医疗图像处理领域，不确定度的研究比较多（论文数量多，可能和医疗图像已经应用于工业界有关），主要针对Classification和segmentation问题，机器人领域次之。

- 从具体问题看，Classification uncertainty的研究比较成熟，segmentation、regression(e.g. depth estimation)次之。如果看CV领域的uncertainty，基本上都能追溯到[Yarin Gal](http://www.cs.ox.ac.uk/people/yarin.gal/website/)和他在NIPS2017年发的论文 [What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? ](https://papers.nips.cc/paper/7141-what-uncertainties-do-we-need-in-bayesian-deep-learning-for-computer-vision.pdf) 。[Yarin Gal](http://www.cs.ox.ac.uk/people/yarin.gal/website/)曾经是剑桥的PHD，现在是牛津的助理教授，他的博士论文是[Uncertainty in Deep Learning](http://mlg.eng.cam.ac.uk/yarin/thesis/thesis.pdf)。

  研究Object Detection Uncertainty的比较少，目前看[QUT Centre for Robotics](https://research.qut.edu.au/qcr/) 的Niko Sünderhauf, Dimity Miller等人研究比较多，主要方向：[Bayesian Deep Learning and Uncertainty in Object Detection](https://nikosuenderhauf.github.io/projects/uncertainty/)，中了不少顶会。 他们提出了一种新的Metric [PDQ(Probabilitybased Detection Quality)](https://openaccess.thecvf.com/content_CVPRW_2019/papers/Uncertainty%20and%20Robustness%20in%20Deep%20Visual%20Learning/Miller_Benchmarking_Sampling-based_Probabilistic_Object_Detectors_CVPRW_2019_paper.pdf)，用于衡量uncertainty/probability。

- 开源代码和工具

  - Yohua Bengio的创业公司[ElementAI](https://www.elementai.com/about-us)开源了一个[Bayesian Active Learning框架Baal](https://github.com/ElementAI/baal)，支持Classification、Segmentation、regression任务的estimation，在uncertainty建模方面支持[MCDropOut](https://arxiv.org/abs/1506.02142)和[MCDropConnect](https://arxiv.org/pdf/1906.04569.pdf)
  - [Ali Harakeh](https://arxiv.org/search/cs?searchtype=author&query=Harakeh%2C+A)等人开源了[BayesOD: A Bayesian Approach for Uncertainty Estimation in Deep Object Detectors](https://arxiv.org/abs/1903.03838)的[代码](https://github.com/asharakeh/bayes-od-rc.git)，这篇论文介绍了检测uncertainty建模。具体来说#TODO

## 3. 怎么计算uncertainty?

接下来，我们来讲讲如何建模计算uncertainty，由于调研时间有限可能有疏漏，欢迎留言补充。

1. Epistemic uncertainty建模
   <img src="https://tva1.sinaimg.cn/large/007S8ZIlly1ghevtp5s25j30z60ckq4p.jpg" alt="image-20200804105920734" style="zoom:40%;" />
   [JunMa](https://www.zhihu.com/people/JunMa11)帮忙总结了几种对Epistemic uncertainty 建模的方式，如上图，也欢迎补充其他建模方法。。本文主要讲Monte-Carlo 和 Ensemble。
   对一个随机分布，不确定性建模的方法有很多，标准差、方差、风险值（VaR）和熵都是合适的度量。在深度学习中，建模不确定度需要用到Bayesian DeepLearning。从Bayesian的角度，深度学习训练的本质是求一个posterior distribution  $P(W|D)$，其中W是参数，D是数据。根据bayes theorem，我们有
   $$ P(W|D) = \frac{P(D|W)P(W)}{P(D)}$$
   但是这个公式没法用，因为P(D)理论上代表的是真实的数据分布 ，无法获取；~~P(W）在神经网络中也是不存在的，因为模型训练好以后，所有参数都是确定的数，而不是distribution，所以没法计算P(W)~~。于是我们想到bayes theorem的另一个公式：
   $$P(D) = \sum_i{P(D|W_i)P(W_i)}$$
   如果我们知道所有W，那么就可以计算P(D)了，但这也是不可能的。不过我们可以用蒙特卡洛法(Monte-Carlo)多次采样逼近：多次采样W计算$P_i(D)$，得到P(D)的近似分布，进而得到P(W|D)的估计。具体来说，有3种方式：

   - Ensembles：用类似bootstrap的方法，对数据集D，采样N次，用N次的结果分别训练模型，然后ensemble模型结果。这个方法的好处是接近真实的Monte-Carlo方法
   - MCDropout：在网络中加入Dropout层，在测试时也打开Dropout，让Dropout成为采样器。对采样N次的结果进行ensemble处理得到最后的uncertainty。这个方法的好处是不用做很多实验，节省成本，但是由于使用了Dropout，单次训练的时间会变长。
   - MCDropConnect：和加Dropout的思路差不多。不过这里不用加Dropout layer，而是通过随机drop connection，来达到随机采样的目的。

   从理论层面，MC-Dropout是variantianl inference(BNN的重要概念之一)的近似。具体来说#TODO

2. Aleatoric uncertainty建模
  <img src="https://tva1.sinaimg.cn/large/007S8ZIlly1ghevtqj025j30yy08u3zv.jpg" alt="image-20200804104534162" style="zoom:50%;" />
  <img src="https://tva1.sinaimg.cn/large/007S8ZIlly1ghevtr1s5yj30to0h2q5n.jpg" alt="image-20200804104444797" style="zoom:50%;" />
  [JunMa](https://www.zhihu.com/people/JunMa11)帮忙总结了三种Aleatoric Uncertainty的建模方法，这里介绍Probabilistic Deep Learning。从表格可以看出，其实就是在原始任务基础上，增加probability prediction，这个probability可用于measure uncertainty。
  比如分类任务原来只输出类别，现在还需要输出probability。为了准确表示uncertainty，这里的probability要求[calibrated probability](https://scikit-learn.org/stable/modules/calibration.html)，不能直接用用softmax输出的score。对目标检测任务也有[Probabilistic Object Detection](https://arxiv.org/abs/1811.10800)，这方面的研究工作有[Gaussian YOLOv3](https://arxiv.org/abs/1904.04620)、[Bayesian Object Detection](https://arxiv.org/abs/1903.03838)以及 #TODO

## 4. 总结
uncertainty estimation是深度学习在实际使用时非常重要的一环。因为我们不仅希望AI输出预测结果，还想知道AI对结果的确定度，综合两者才能更好地使用DL模型。

在DL领域，主要有两种不确定度，Aleatoric Uncertainty和Epistemic Uncertainty，前者可以认为是数据本身的噪声，也被称之为data uncertainty，后者主要源于模型认知能力，可通过增加训练数据解决。

为了建模计算Uncertainty，我们介绍了Monte-Carlo DropOut/DropConnect、Ensemble方法来建模Epistemic uncertainty，也介绍了Probabilistic DeepLearning用于计算(预测)Aleatoric Uncertainty。更多的建模方法有待进一步研究和比较，可参考文章提到的文章，也欢迎留言补充，谢谢。

## 5. 参考文献
[201909-医学图像分析中的Uncertainty学习小结](https://zhuanlan.zhihu.com/p/87955728) 推荐\*\*\*\*\*\*
[What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?](https://arxiv.org/pdf/1703.04977.pdf)推荐指数\****\*
[Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://papers.nips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles.pdf)推荐指数\****
[Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](http://proceedings.mlr.press/v48/gal16.pdf)
[机器之心：如何创造可信任的机器学习模型？先要理解不确定性](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650755237&idx=3&sn=55beb3edcef0bb4ded4b56e1379efbda&scene=0#wechat_redirect) 推荐指数\***
[Uncertainty Estimation in DL](https://zhuanlan.zhihu.com/p/82493716)推荐指数\***
[AI科技评论：学界 | 模型可解释性差？你考虑了各种不确定性了吗？](https://mp.weixin.qq.com/s?__biz=MzA5ODEzMjIyMA==&mid=2247496311&idx=3&sn=3e7f1df007926e6fba1124630046be76&source=41#wechat_redirect)
[CSDN:What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? 计算机视觉用于贝叶斯深度学习的不确定性](https://blog.csdn.net/weixin_39779106/article/details/78968982#1将异方差偶然不确定性和认知不确定性相结合)
[Uncertainties in Bayesian Deep Learning - kid丶的文章 - 知乎]( https://zhuanlan.zhihu.com/p/100998668)
[Homoscedastic regression贝叶斯神经网络建模两类不确定性——NIPS2017 - Dannis的文章 - 知乎](https://zhuanlan.zhihu.com/p/88654038)
[DL模型不确定性评价Can You Trust Your Model's Uncertainty? - JunMa的文章 - 知乎](https://zhuanlan.zhihu.com/p/69406639)
[YARIN GAL 个人网站，上述论文作者（剑桥、deepmind)](http://www.cs.ox.ac.uk/people/yarin.gal/website/blog_3d801aa532c1ce.html)
[【实验笔记】深度学习中的两种不确定性（上）](https://zhuanlan.zhihu.com/p/56986840)
[201910-Uncertainty in MICCAI 2019](https://zhuanlan.zhihu.com/p/87974770)