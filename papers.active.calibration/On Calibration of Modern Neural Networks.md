# On Calibration of Modern Neural Networks

## 问题

1. 为什么NN会miscalibrated?
2. 如何取消或者弱化这种情况？
3. ECE实际使用？



## 重要概念

- perfect calibration

  给N=100个预测结果，如果每个预测结果的conf=0.8，perfect calibration意味着接近N*=0.8=80个结果会是正确的（N越大，正确预测结果数，越接近0.8N）:
  $$
  \mathbb{P}(\hat{Y}=Y|\hat{P}=p)=p, \forall{p} \in [0,1]
  $$

- Reliability Diagram

  <img src="/Users/lizhiwei/Documents/papers.active.calibration/image-20201102140046542.png" alt="image-20201102140046542" style="zoom:30%;" />

上图中，bottom graph是 accuracy基于confidence的函数。Gap是实际acc和expected acc的差异。

把conf分为M个区间，定义$B_m$是预测结果conf落入第m个区间的sample ind的集合：
$$
acc(B_m) = \frac{1}{|B_m|}\sum_{i\in B_m}\mathbb{1}(\hat{y}_i = y_i)
$$

$$
conf（B_m) = \frac{1}{|B_m|}\sum_{i\in B_m}\hat{p}_i
$$

从 $acc(B_m)$ 和 $conf(B_m)$ 的差异，可以看出多少sample是well calibrated，多少不是

- Expected Calibrantion Error（ECE）

  calibration的统计结果，可以通过ECE给出：
  $$
  \mathbb{E}_{\hat{P}}[|\mathbb{P}(\hat{Y}=Y|\hat{P}=p) - p|]
  $$

  $$
  ECE = \sum_{m=1}^{M}\frac{|B_m|}{n}|acc(B_m) - conf(B_m)|
  $$

  <u>其中n是样本数？</u>

