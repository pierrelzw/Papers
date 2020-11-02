# Large-Scale Visual Active Learning with Deep Probabilistic Ensembles

## 摘要

”标注那些真正重要的数据“ ，在dnn任务中是一个挑战。基于uncertainty estimation的active learning是一种解决方案。

本文介绍DPEs，一种可拓展的regularized ensemble方法，可近似BNN。我们通过实验分析：对分类任务用CIFAR10/100，ImageNet数据集，对分割任务用BDD100k数据集，发现我们的DPEs方法在相对较少数据标注情况下，达到不错的性能，在有限标注预算情况下，提高performance baseline



## Deep ensembles

### Variantinal Inference

在BNN中，我们把网络参数w作为latent variable，它来自prior distribution p(w). 这些权重和我们的observation x之间可通过likelihood p(x|w)联系起来。我们希望计算posterior distribution p(w|x)
$$
p(w|x) = \frac{p(w)p(x|w)}{p(x)} = \frac{p(w)p(x|w)}{\int p(x|w)p(w)dw}
$$
这个式子没法解，因为我们无法获得所有的p(w)。

于是我们想到，我们可以生成一系列的distribution $q^*$来作为latent variable。

我们的目的，是希望每个$q^*(w)$和真实posterior（即求解目标）$p(w|x)$之间的KL divergence最小：
$$
q^*(w) = argmin_{q(w)\in D}KL(q(w)||p(w|x))
$$

$$
= argmin_{q(w) \in D}\mathbb{E}([log\frac{q(w)}{p(w|x)}])
$$



为了让这个函数可解，我们减去一个和weight无关的const，由此我们得到新的目标函数:
$$
-ELBO = \mathbb{E}([log\frac{q(w)}{p(w|x)}]) - \log{p(x)} \\
= \mathbb{E[\log{q(w)}]} - \mathbb{E}[\log{p(w|x)}] - \log{p(x)}
$$

$$
-ELBO = \mathbb{E}[\log{q(w)}] - \mathbb{E}[log\frac{p(w)p(x|w)}{p(x)}]- \log{p(x)} 
\\
\ \ \ \ \ \ \ \ \ \ \ \ \ \ \  
= \mathbb{E}[\log{q(w)}] - \mathbb{E}[\log{p(w)}] - \mathbb{E}[\log{p(x|w)}] + \mathbb{E}[\log{p(x)}] - \log{p(x)}
\\
= \mathbb{E}[\log{q(w)}] - \mathbb{E}[\log{p(w)}] - \mathbb{E}[\log{p(x|w)}]
\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \\\
= \mathbb{E}[log\frac{q(w)}{p(w)}] - \mathbb{E}[\log{p(x|w)}] 
\ \ \ \ \ \ \ \  \ \ \ \ \ \ \ \  \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \\
= KL(q(w)||p(w)) - \mathbb{E}[\log{p(x|w)}]
\ \ \ \ \ \ \  \ \ \ \ \ \ \ \  \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \
$$

上式中第一项是$q(w)$ 和$p(w)$KL散度，第二项是NLL(negative log  likelihood )期望



**Co-adapation**(网络不同feature以未知、复杂的方式相互关联) 可以使deterministic net（BNN不是deterministic的）的训练更容易，但是也影响了它的泛化性。

传统的regularization方法，比如dropout可以一定程度上提升泛化性，但是这其中就有training容易程度和泛化性之间的平衡问题。*（L1，L2正则化呢？）*

一个deterniministic ensemble可以exploit co-adaptation，因为每个网络都是各自优化的，这使得他们更加容易训练（优化）。



一般来说，BNN会阻止其explot co-adaptation，因为对所有采样自BNN的所有net，我们希望最小化NLL。理论上这将达到很好的泛化性，但是这也导致BNN基本无法训练。

本文提出一种regularization来实用ensemble的optimization simplicity。



传统的regularization方法常用L1,L2 regularization。（具体怎么用来着？？？）

给定从ensemble中得到的参数set，我们由此可以得到结果set，我们用公式5中的KL divergence作该结果set的regularization penalty $\Omega$。



在我们的任务中，我们选择Gaussian作为q(w)和p(w)的先验。


$$
KL(q||p) = \frac{1}{2}(\log{\frac{\sigma_q^2}{\sigma_p^2}} + \frac{\sigma_p^2 + (\mu_q-\mu_p)^2}{\sigma_q^2}-1)
$$
我们用KaimingHe的方法，初始化我们的参数作为prior。对BN参数，我们固定$\sigma_p^2=0.01$，$\mu_p=1$（weight)，$\mu_p=0$（bias）

对convRELU网络（n_i input channels，n_o output channels），我们设置$\mu_p=0, \sigma_p^2=\frac{2}{n_owh}$,

全连接层可以认为是conv的特例。

对网络层l，我们有  
$$
\Omega^l = \sum_{i=1}^{n_in_owh}(\log{\sigma_i^2} + \frac{2}{n_owh\sigma_i^2} + \frac{\mu_i^2}{\sigma_i^2})
$$
其中，$\mu_i$ 和$\sigma_i$是从ensemble members中取的均值和方差。上式中，第一项防止方差过大，这使得ensemble members之间的差异不至于太大，第二项惩罚方差小于prior的情况，第三项使得实际参数的mean和prior接近，当方差比较小的时候。

所以，新的目标函数
$$
\Theta^* = argmin_{\Theta}\sum_{i=1}^{N}\sum_{e=1}^{E}H(y^i, M_e(x^i, \Theta_e)) + \beta\Omega(\Theta)
$$
其中$\{x^i, y^i\}_{i=1}^{N}$是训练数据，E是ensemble数量。$\Theta_e$是模型$M_e$的参数。我们通过公式（7）获得惩罚项$\Omega$。$\beta$是一个常数，用来限制惩罚项的scale。通过对每个model的loss进行求和，我们得到ELBO中的NLL项的近似。









