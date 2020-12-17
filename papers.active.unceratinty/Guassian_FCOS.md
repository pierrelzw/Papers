## Contributions

1. 首次在anchor-free 检测器上实现bbox uncertainty estimation. 
2. 设计了新的loss function, 在power likelihood的启发下，通过对loss进行IoU加权，来解决misspecification problem
3. 通过在COCO数据集实验发现，uncertainty estimation对box预测不准有改善

## Questions

- misspecification problem
- Uncertainty calibration with NMS
- power likelihood



## Gaussian FCOS

Centerness可以作为间接的localization uncertainty，因为它被用来过滤bbox，但是只有中心点的uncertainty还不够。

假设center对四个方向的offset是独立的。那么有
$$
P
$$

- Power likelihood

  之前的很多工作把box offset和GT box认为是gaussian 和 dirac，分别对应negative likelihood loss和KL-divergence loss。这等价于
  $$
  \mathcal{L}  = -\frac{1}{N}\sum P_D(x)·\log{P_\Theta(x)}
  $$
  其中，$P_D,P_{\Theta}$分别是Dirac Delta function和gaussian probability density function

  

- misspecification problem : Dirac Distribution不属于gaussian family。

  根据Assign a value to a power likelihood，power likelihodd 可以解决这个问题。我们使用IOU作为power，因为更高的IOU收到的影响更大（什么影响？）

  
  $$
  \mathcal L_u = \frac{\lambda}{N_{pos}}\sum_i\sum_k IoU_i·\log P_{\Theta}(B_{i,k}^g|\mu_k, \sigma_k^2)
  $$

  $$
  \mathcal L_u = \frac{\lambda}{N_{pos}}\sum_i IoU_i·[\sum_k \{\frac{(B_{i,k}^g-\mu_k)^2}{2\sigma_k^2} + \frac{1}{2}\log{\sigma_k^2\} } + 2\log2\pi])
  $$

- uncertainty calibration

  传统的NMS只考虑class score。我们通过class score * (1 - uncertainty)用于NMS。一些可视化的效果证明，我们的方法有助于减少FP。

