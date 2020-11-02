# Adam Leśnikowski: Data-Driven Dataset Creation

## dataset

- 分割：100k train/10k test
- 检测：150k train/10k test



## experiments

- 分割

<img src="/Users/lizhiwei/Library/Application Support/typora-user-images/image-20201023113314615.png" alt="image-20201023113314615" style="zoom:50%;" />

十次实验的结果

<img src="/Users/lizhiwei/Library/Application Support/typora-user-images/image-20201023113449084.png" alt="image-20201023113449084" style="zoom:50%;" />

active learning选择的图片



BNN太难训练，实际上 ，少数Bayesian+大多数deterministic layer的组合，也可以达到不错的uncertainty estimation效果。

训练的时候就一次，但是test时，会有bayesian+deterministic的组合。



BDD 、 C400、C410



detection : 1-10 USD/img

segmentation：1k-10k USD/img（medical，expert）