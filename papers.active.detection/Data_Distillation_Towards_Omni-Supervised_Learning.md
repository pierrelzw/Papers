# Data Distillation: Towards Omni-Supervised Learning

<img src="https://tva1.sinaimg.cn/large/0081Kckwly1glwte9y3coj30og0qmq60.jpg" alt="image-20200824121353651" style="zoom:33%;" />

## 关键概念

- omni-supervised learning ：一种特殊的semi-supervised learning方法

- data distillation： 对未标注图片分别用不同的预处理方法处理，然后用一个模型推理并聚合推理结果，以产生未标注图片的标注

- model distillation：



- knowledge distillation：本文旨在从未标注数据中获取提升模型能力的东西，我们称之为knowledge。用data distillation生成未标注数据的标注后，我们有了新的数据集。我们用模型来进行训练。这个模型被称之为student model，训练原始labeled data的模型被称之为teacher model。学生模型和老师模型可以一样也可以不同。但是在训练策略上我们有2个限制：1）对每个batch，我们保证一定有原始标注数据的生成标注数据 2）我们会训练更加久来保证模型能够更好地fit更大的数据集（student model用的）



- data transformation：本文主要用了两种数据变换方法：scaling和horizontal-flliping
  - Scaling: 把图片resize使得，图片的短边等于[400, 500, 600, ……, 1200]中的的一个，
- Ensembling：在聚合模型结果的时候，我们可以选择对每个阶段、每个head的输出结果进行 multi-transform inference，比如mask-rcnn。但是为了简化，本文只对kepoint detection的keypoint head进行 multi-transform inference，聚合方式采用简单的average，然后取max得到keypoint
- selecting predictions：ensembling得到的predictions可能有FP（也会有FN吧？），所以我们通过detection score来选择哪些是我们希望留下的label。我们发现，当score_threshold使得每个未标注数据中的instance数和已标注数据中的instance数差不多一样时，效果最好。

