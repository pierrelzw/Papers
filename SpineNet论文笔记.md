## 链接 
[SpineNet: Learning Scale-Permuted Backbone for Recognition and Localization](https://arxiv.org/abs/1912.05027)
## 摘要
图像分类任务通常采用逐步减小分辨率的网络结构，但是这种结果并不适合同时需要识别和定位的任务。为减小分辨率损失问题，分类任务中采用encoder-decoder结构，但是该结构并不高效，尤其在需要产生强多尺度的特征时。我们将介绍一种通过NAS搜索得到的网络——SpineNet，它具有scale-permute intermediate 特征和cross-scale连接。相比ResNet-FPN，spineNet具有6%的AP增益。SpineNet结构也可以迁移到分类网络，在iNaturalist fine-grained dataset 比赛中，在top1 accuracy上比ResNet高出6%。

## 核心要点
1. 尺度交换网络结构&和resNet的对比：
**如何保证不同层之间的concate，connection本身包含conv或者deconv么？** 
![scale-permute](https://img-blog.csdnimg.cn/20200205171150877.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3poaXdlaTJjb2Rlcg==,size_16,color_FFFFFF,t_70)
2. Cross-Scale connection
![cross-scale-fusion](https://img-blog.csdnimg.cn/20200207164001653.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3poaXdlaTJjb2Rlcg==,size_16,color_FFFFFF,t_70)
- upsampling用最近邻interp
- downsampling用stride=2的3x3conv
- fusion则element-wise addition
- 为了保持resampling的低计算复杂度，使用一个channel调控系数$\alpha$

4.  从ResNet到SpineNet ![ResNet2SpineNet](https://img-blog.csdnimg.cn/20200205173901996.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3poaXdlaTJjb2Rlcg==,size_16,color_FFFFFF,t_70)
通过交换ResNet的特征建立scale-permute Net。(a)是基于ResNet的FPN网络；(b)使用7个ResNet block + 10 sclae-permute block ; (c) 全部是scale-permute结构; (d)基于scale-permute思想搜索得到的网络。以上网络随着scale-permute的应用，AP逐步增加。

## 实验结果
1. 轻量检测网络
![on-device](https://img-blog.csdnimg.cn/20200205174204681.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3poaXdlaTJjb2Rlcg==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200205174247686.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3poaXdlaTJjb2Rlcg==,size_16,color_FFFFFF,t_70)
与MBlock结合，
CenterNet ResNet-18：AP=28.1，142FPS@TITAN-V
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200205175703543.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3poaXdlaTJjb2Rlcg==,size_16,color_FFFFFF,t_70)

