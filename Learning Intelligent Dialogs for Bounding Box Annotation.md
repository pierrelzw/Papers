#Learning Intelligent Dialogs for Bounding Box Annotation



<img src="/Users/lizhiwei/Library/Application Support/typora-user-images/image-20201019115204953.png" alt="image-20201019115204953" style="zoom:50%;" />

## 摘要

这篇文章主要提出一种智能的bbox标注加速方法：通过训练一个模型（agent）来自动选择一些的actions，以辅助人类标注。具体来说，主要包两种action：1.人工检验，即标注员检验一个由检测器detector产生的bbox 2. 手动画bbox。如上图所示，左图检测器生成2两个框，该情况适合人工检验（人工检验后删除不对的bbox、微调即可），右图产生了很多bbox，大多数bbox并不准确，这种情况适合人工手动画bbox。

这篇文章探索了两种模型来加速box标注。一种输出prob，代表该bbox通过人工检验（positively verified）无需修改的概率，另一种模型基于强化学习，通过trial-and-error交互实验RL训练。



## 问题建模

œ