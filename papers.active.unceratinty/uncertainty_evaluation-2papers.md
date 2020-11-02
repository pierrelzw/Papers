# uncertainty_evaluation-2papers

## why?

推荐两篇关于不确定度(uncertainty estimation)效果评估的文章。从两篇文章中，可以了解：

1. 常用的uncertainty estimation方法
2. 如何评估uncertainty结果的优劣
3. 那种uncertainty estimation方法效果最好？



## Can You Trust Your Model’s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift

1. 常用的uncertainty estimation方法

   - (Vanilla) Maximum softmax probability (Hendrycks & Gimpel, 2017)

     去softmax输出的

   -  (Temp Scaling) Post-hoc calibration by temperature scaling using a validation set (Guo et al., 2017)

   -  (Dropout) Monte-Carlo Dropout (Gal & Ghahramani, 2016; Srivastava et al., 2015) with rate p

     在每个网络层后添加dropout层（droprate=p），正常训练。test时开启dropout，进行多次inference采样预测结果，对多次结果进行平均得到最后输出（Monte-Carlo）

   -  (Ensembles) Ensembles of M networks trained independently on the entire dataset using random
     initialization (Lakshminarayanan et al., 2017) (we set M = 10 in experiments below)

     用M个网络同时训练所有数据，每个网络用随机初始化

   -  (SVI) Stochastic Variational Bayesian Inference for deep learning (Blundell et al., 2015; Graves,
     2011; Louizos & Welling, 2017, 2016; Wen et al., 2018). We refer to Appendix A.6 for details of
     our SVI implementation.

   -  (LL) Approx. Bayesian inference for the parameters of the last layer only (Riquelme et al., 2018)
     – (LL SVI) Mean field stochastic variational inference on the last layer only
     – (LL Dropout) Dropout only on the activations before the last layer

2. 如何评估uncertainty结果的优劣

   - Negative Log-Likelihood (NLL）
   - Brier Score  (Brier, 1950) 
   - Expected Calibration Error (ECE)

3. 那种uncertainty estimation方法效果最好？

结论：Ensembles最好

