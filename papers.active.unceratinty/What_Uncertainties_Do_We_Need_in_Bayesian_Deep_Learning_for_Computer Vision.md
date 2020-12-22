# What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?

> Yarin Gal  University of Cambridge  yg279@cam.ac.uk
>
> Alex Kendall University of Cambridge agk34@cam.ac.uk

## 简介

计算机视觉认为认为可以大体分为两类：分类和回归。比如最简单的图像分类，属于分类问题；目标检测问题，属于分类+回归问题；语义分割问题，属于分类问题；深度估计问题，属于回归问题。

这些问题都有很多SOTA的模型，然而，很少有模型会输出其对结果的不确定度。回归问题一般只输出结果，不输出不确定度，分类问题一般输出一个normalized score vector，但是实际上score并不是能很好地代表不确定度。

在Bayesian Modeling中，我们把uncertainty分为两种：aleatoric uncertainty(又称 data uncertainty) 和 epistemic uncertainty(i.e. model uncertainty)。aleatoric uncertainty是来源于数据本身，原始数据质量（e.g. 图像清晰度）、标注质量（e.g. 标注精度）等因素影响，无法通过增加数据解决；epistemic uncertainty则可以通过增加数据解决。

![image-20200730191542498](https://tva1.sinaimg.cn/large/007S8ZIlly1gh97vw2yndj31520kadm2.jpg)

我们用实际案例来理解这两种不确定度，如上图所示：

- aleatoric unceratinty出现在物体边缘和远处，该uncertainty源于标注员对物体边缘标注的精度误差和远处较差的成像质量

- epidemic uncertainty出现在model预测不好的地方，往往出现在图像特征不够清晰，或模型没有见过的样本上。比如右下角的图，模型对人行道的分割结果较差，epistemic uncertainty比较高。

  

## 本文贡献

![image-20200730105539691](https://tva1.sinaimg.cn/large/007S8ZIlly1gh97vwr8tjj31bs0ek78n.jpg)

1. 明确定义了什么是aleatoric uncertainty和epistemic uncertainty，并提出了一种新的建模分类不确定度的方法
2. 通过bayesian modeling，在损失函数中考虑noisy data的影响，提升了模型性能1-3%。
3. 研究了aleatoric 和 epistemic uncertainty的trade-off，综合考虑模型的推理时间和性能

## Epistemic Uncertainty

在BNN中，我们假设模型参数 $W \sim \mathcal N(0,I)$ , 有deterministic NN不同，我们把BNN在参数的分布空间下得到的结果平均得到最后的结果（被称为marginalisation)。假设BNN的random output为 $f^W(x)$, 我们定义<u>Gaussian likelihood</u> 为$p(y|f^W(x))$

给一个数据集 $X = {x_1, x_2, ... x_N}, Y={y_1, y_2, ..., y_N}$, bayesian inference 就是希望计算posterior over weights : $p(W|X,Y)$

对regression任务，我们定义likelihood为Gaussian with mean，$p(y|f^W(x)) = \mathcal N(f^W(x), \sigma^2)$, 其中，$\sigma$是observation noise.

对classification任务，我们定义likelihood为 $p(y|f^W(x)) = Softmax(f^W(x))$

BNN中，计算最后结果的公式很容易给出：$p(W|X, Y)  = p(Y|X, W)P(W) / P(Y|X)$ ,但是无法计算（因为没法知道所有的weight W）。

于是很多工作通过近似的方法计算，其原理就是假设p(W|X, Y)和分布$q_{\theta}^*(W)$ 近似。

Dropout variational inference就是一种近似求解方法，这种方法要求在每个weight layer后加Dropout，然后在test阶段也使用dropout，并用MonteCarlo采样得到posterior P(W|X,Y)的近似。本质上，这种方法就是在找满足和posterior的KL divergence最小的 $q_{\theta}^*(W)$。

根据[An Introduction to Variational Methods for Graphical Models](https://people.eecs.berkeley.edu/~jordan/papers/variational-intro.pdf) 

> Dropout 被认为是variantional bayesian approximation，其中approximation distribution是两个有着很小variance的高斯分布，且其中一个分布的mean是0；

求解过程的最小化目标函数（loss）为：
$$
\mathcal{L}(\theta, p) = -\frac{1}{N}\sum_{i=1}^{N}{\log{p(y_i|f^{\widehat{W_i}}(x_i))}} + \frac{1-p}{2N}||\theta||^2
$$


其中，N是数据点数，p是dropout probability， $\widehat{W}_i \sim q_{\theta}^*(W)$，其中，$\theta$是simple distribution's parameters to be optim

对regression任务，如果假设我们的回归目标服Gaussian，则上式中的 $\log p$ 满足
$$
-\log{p(y_i|f^{\widehat{W_i}}(x_i))} \propto \frac{1}{2\sigma^2}||y_i-f^{\widehat{W_i}}(x_i)||^2 + \frac{1}{2}\log{\sigma^2}
$$
补充，Gaussian probability density 
$$
g(x) = \frac{1}{\sigma\sqrt{2\pi}}exp(-\frac{||x-\mu||^2}{2\sigma^2})
$$


- **classification**
  $$
  p(x) = \frac{1}{1+e^{-x}} = \frac{e^x}{e^x+1}
  $$
  
  
  
  Epidemic Uncertainty approximation：
  
$$
  p(y=c|x,X,Y)\approx\frac{1}{T}\sum_{t=1}^{T}Softmax(f^{\widehat{W}_i}(x))
$$




  The uncertainty of probability vector **p** : 
$$
  H(p) = -\sum_{c=1}^{C}p_c\log(p_c)
$$

- **Regression**
  
  
  
  Epidemic Uncertainty = predictive variance :
$$
  Var(y)\approx\sigma^2+\frac{1}{T}\sum_{t=1}^{T}f^{\widehat{W_t}}(x)^Tf^{\widehat{W_t}}(x_t) - {E(y)^T}{E(y)}
$$

  


  approximationn predictive mean: 
$$
  E(y) = \frac{1}{T}\sum_{t=1}^{T}f^{\widehat{W}_t}(x)
$$
  $\sigma^2$ : the amount of noise inherent in the data 

  

##  Aleatoric Uncertainty

对回归任务，有两种随机不确定度，一种是同质的，一种是异质的。

- Homoscedastic regression

  > Homoscedastic regression assumes constant observation noise σ for every input point x.

  对任何环境下的输入，都具有的随机不确定性（或者说噪声$\sigma$），与input data无关。比如相机本身的噪声（ISP），比如激光雷达淋雨后导致的质量降低。

- Heteroscedastic c regression

  与输入数据有关的随机不确定性。比如，在深度估计任务中，一张图像是颜色一样的一面墙，另一张是包含vanishing lines的图像，前者的uncertainty应该比后者高。
  
  > 在非BNN网络中， observation noise被作为模型的weight decay然后被忽略
  
  可是，如果说噪声 是data-dependent的，那么我们可以用网络学习它：
  $$
  \mathcal L_{NN}(\theta) = \frac{1}{N}\sum_i \frac{1}{2\sigma(x_i)^{2}}||y_i-f(x_i) ||^2 + \frac{1}{2}\log \sigma(x_i)^2
  $$
  这个loss将被添加一个weight decay参数$\lambda$. 需要提醒的是，这里variational inference没有考虑所有可能的参数$\theta$，我们用的是MAP（Maximum a posteriori estimation）——只计算一种可能的$\theta$.
  
  > 这种方法不能获得model uncertainty，因为epistemic uncertainty是模型property不是数据的。

## Combine Heteroscedastic Aleatoric & Epistemic Uncertainty

> 我们看到在regression任务中，aleatoric uncertainty可以被解释为learned loss attenuation——能够使模型对noisy data更robust。我们把heteroscedastic regression中的方法推广到classification任务中。

我们假设用gaussianc likehood来建模aleatoric uncertainty，则此时我们希望最小化的目标函数为：
$$
\mathcal L_{BNN}(\theta) = \frac{1}{D}\sum_i \frac{1}{2}\hat\sigma_i^{-2}||y_i-\hat y_i||^2 + \frac{1}{2}\log \hat\sigma_i^2
$$
其中，$y_i$是一个数据点对应的输出，一个数据点可以是一个pixel，也可以是一张图片（此时D=1)。

这个loss 包含两个部分：前半部分是预测值和GT的残差再乘以variance， 后半部分是一个regularization项。

- 可以防止网络趋向于把所有datapoint的variance学得很大。
- 太大uncertainty的datapoint（noise？）会被惩罚，网络会趋向于忽略这样的数据
- 同时，网络会抑制，低uncertainty但是高residual的样本，因为这样第一项会很大

这使得网络对noise data更加robust，因为（网络收敛时？）高uncertainty的样本，其对loss的贡献会更小。



在实际网络的时候，我们让网络去学(estimate)$s_i = \log \sigma_i^2$，因为log会降低要学习的量的scale，让整个学习过程更加稳定。此时loss 函数为：
$$
\mathcal L_{BNN}(\theta) = \frac{1}{D}\sum_i \frac{1}{2}exp(-s_i)||y_i-\hat y_i||^2 + \frac{1}{2}s_i
$$
https://github.com/asharakeh/pod_compare/blob/7784b0b45fc6e6ee17e87132c75fa9583004e57f/src/probabilistic_modeling/probabilistic_retinanet.py#L298

```python
loss_box_reg = 0.5 * torch.exp(-pred_bbox_cov) * smooth_l1_loss(
  pred_anchor_deltas,
  gt_anchors_deltas,
  beta=self.smooth_l1_beta)

loss_covariance_regularize = 0.5 * pred_bbox_cov
loss_box_reg += loss_covariance_regularize

loss_box_reg = torch.sum(
  loss_box_reg) / max(1, self.loss_normalizer)
```

## Heteroscedastic Aleatoric Uncertainty in Cls task

Heteroscedastic classification NN

> tenchnically classification has input-dependent uncertainty  

对一个分类网路NN，它对每个pixel i预测了一个vector $f_i $，把$f_i$输入softmax则得到vector $p_i$. 我们改变model, 假设$f_i$服从Gaussian distribution：
$$
\hat x_i | W \sim \mathcal N(f_i^W, (\sigma_i^W)^2)
$$

$$
\hat p_i = Softmax(\hat x_i)
$$

其中，$f_i^W, \sigma_i^W$ 是包含权重为W的网络的输出.

我们expected log likelihood：
$$
log E_{\mathcal N(\hat x_i; f_i^W, (\sigma_i^W)^2)}[\hat p_i, c]
$$
其中，c是pixel i的observed class。

我们想integrate 这个gaussian distribution，但是没法直接计算，于是我们用数值计算方法：通过MonteCarlo integration近似。需要说明的是，我们只从最后logits开始采样，所以这会非常快。我们重写上面的函数，并得到下面的numerically-stable stochastic loss：
$$
\hat x_{i,t} = f_i^W + \sigma_i^W\epsilon_t,  \epsilon_t \sim \mathcal N(0,1)
$$

$$
\mathcal L_x = \sum_i \log \frac{1}{T}\sum_t \exp(\hat x_{i,t,c} - log\sum_{c'}\exp\hat x_{i,t,c'} )
$$

其中，$x_{i,t,c'}$是第 $logits = x_{i,t}$ 的第$c'$个元素。

这个目标函数也可以被解释为learning loss attenuation（为什么我们希望它小呢？包括之前在regression任务中，为什么他们合理呢？）



## 如何计算variance？ network estimation

不管是分类还是回归，都用一层网络层预测variance（实际上是$\log \sigma^2$）

https://github.com/asharakeh/pod_compare/blob/7784b0b45fc6e6ee17e87132c75fa9583004e57f/src/probabilistic_modeling/probabilistic_retinanet.py#L457

```python
# Create subnet for classification variance estimation.
if self.compute_cls_var:
    self.cls_var = nn.Conv2d(
        in_channels,
        num_anchors *
        num_classes,
        kernel_size=3,
        stride=1,
        padding=1)

    for layer in self.cls_var.modules():
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
            torch.nn.init.constant_(layer.bias, -10.0)

# Create subnet for bounding box covariance estimation.
if self.compute_bbox_cov:
    self.bbox_cov = nn.Conv2d(
          in_channels,
          num_anchors * self.bbox_cov_dims,
          kernel_size=3,
          stride=1,
          padding=1)
    for layer in self.bbox_cov.modules():
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.normal_(layer.weight, mean=0, std=0.0001)
            torch.nn.init.constant_(layer.bias, 0)
```

然后在loss计算的时候：

- 如果计算了cls_var，则基于原始分类logit（mean）和var重新建模logits，并做T次采样得到最终结果，
- 如果计算了bbox_var（or bbox_cov)，则输出该cov作为uncertainty

https://github.com/asharakeh/pod_compare/blob/7784b0b45fc6e6ee17e87132c75fa9583004e57f/src/probabilistic_modeling/probabilistic_retinanet.py#L228

```python
if self.compute_cls_var:
  # Compute classification variance according to:
  # "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?", NIPS 2017
  if self.cls_var_loss == 'loss_attenuation':
    num_samples = self.cls_var_num_samples
    # Compute standard deviation(the neteork estimate log\simga^2)
    pred_class_logits_var = torch.sqrt(torch.exp(
      pred_class_logits_var[valid_mask]))

    pred_class_logits = pred_class_logits[valid_mask]

    # Produce normal samples using logits as the mean and the standard deviation computed above
    # Scales with GPU memory. 12 GB ---> 3 Samples per anchor for
    # COCO dataset.
    univariate_normal_dists = distributions.normal.Normal(
      pred_class_logits, scale=pred_class_logits_var)

    pred_class_stochastic_logits = univariate_normal_dists.rsample(
      (num_samples,))
    pred_class_stochastic_logits = pred_class_stochastic_logits.view(
      (pred_class_stochastic_logits.shape[1] * num_samples, pred_class_stochastic_logits.shape[2], -1))
    pred_class_stochastic_logits = pred_class_stochastic_logits.squeeze(
      2)

    # Produce copies of the target classes to match the number of
    # stochastic samples.
    gt_classes_target = torch.unsqueeze(gt_classes_target, 0)
    gt_classes_target = torch.repeat_interleave(
      gt_classes_target, num_samples, dim=0).view(
      (gt_classes_target.shape[1] * num_samples, gt_classes_target.shape[2], -1))
    gt_classes_target = gt_classes_target.squeeze(2)

    # Produce copies of the target classes to form the stochastic
    # focal loss.
    loss_cls = sigmoid_focal_loss_jit(
      pred_class_stochastic_logits,
      gt_classes_target,
      alpha=self.focal_loss_alpha,
      gamma=self.focal_loss_gamma,
      reduction="sum",
    ) / (num_samples * max(1, self.loss_normalizer))
    else:
      raise ValueError(
        'Invalid classification loss name {}.'.format(
          self.bbox_cov_loss))
else:
  # Standard loss computation in case one wants to use this code
  # without any probabilistic inference.
  loss_cls = sigmoid_focal_loss_jit(
    pred_class_logits[valid_mask],
    gt_classes_target,
    alpha=self.focal_loss_alpha,
    gamma=self.focal_loss_gamma,
    reduction="sum",
  ) / max(1, self.loss_normalizer)

```





##  TODO

- 结合fata和model uncertainty的方法
- 实验



## Reference

[机器之心：如何创造可信任的机器学习模型？先要理解不确定性](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650755237&idx=3&sn=55beb3edcef0bb4ded4b56e1379efbda&scene=0#wechat_redirect) 推荐指数\****

[AI科技评论：学界 | 模型可解释性差？你考虑了各种不确定性了吗？](https://mp.weixin.qq.com/s?__biz=MzA5ODEzMjIyMA==&mid=2247496311&idx=3&sn=3e7f1df007926e6fba1124630046be76&source=41#wechat_redirect)

[CSDN:What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? 计算机视觉用于贝叶斯深度学习的不确定性]([https://blog.csdn.net/weixin_39779106/article/details/78968982#1%E5%B0%86%E5%BC%82%E6%96%B9%E5%B7%AE%E5%81%B6%E7%84%B6%E4%B8%8D%E7%A1%AE%E5%AE%9A%E6%80%A7%E5%92%8C%E8%AE%A4%E7%9F%A5%E4%B8%8D%E7%A1%AE%E5%AE%9A%E6%80%A7%E7%9B%B8%E7%BB%93%E5%90%88](https://blog.csdn.net/weixin_39779106/article/details/78968982#1将异方差偶然不确定性和认知不确定性相结合)

[Uncertainties in Bayesian Deep Learning - kid丶的文章 - 知乎]( https://zhuanlan.zhihu.com/p/100998668)

[Homoscedastic regression贝叶斯神经网络建模两类不确定性——NIPS2017 - Dannis的文章 - 知乎](https://zhuanlan.zhihu.com/p/88654038)

