

## KeyConcept

1. **Scoring function**

- goal ： 计算一个image-level score以表示informativeness

- assumption：对每个classe，detector输出一个2D confidence/probability map

- **Entropy**

  $$H(p_c) = p_c\log{p_c} + (1 - p_c)\log(1 - p_c)$$

  其中，$p_c$是位置p，类别c的probability

- **Mutual Information（MI）**

  make use of E models to measure disagreement

  1. 计算平均probability

     $\bar{p_c} = \frac{1}{|E|}\sum_{e\in E}p_c^{(e)}$

  2. 计算MI

     $$MI(p_c) = H(\bar{p_c}) - \frac{1}{|E|}\sum_{e\in E}H(p_c^{(e)})$$

     MI会倾向于把模型 disagreement比较高的图片挑出来

- **Gradient of the output layer(Grad)**

  

- **Bounding-boxes with confidence（Det-Ent）**

  假设模型不仅输出bbox，还有一个关联的probability——bbox的uncertainty。这使我们可以计算每个bbox的entropy。进而我们可以aggregation计算image-level的分数。

  

2. **Weighted Average Precision（wMAP）**

3. **Score Aggregation**

   我们会用max或者avg来聚合计算得到最终的image-level score。特别的，对Det-Ent，我们取sum而不是avg。*（也会bias到有很多物体的图片上，另外，哪些box会参与到计算当中呢？）*

   $$s = \max\limits_{c \in C}\max\limits_{p}I(p_c)$$

   取max，很容易被outliers影响，取avg则比较倾向于选择那些有很多objects的图片（因为一个高uncertainty的图片的score，可能低于有很多低uncertainty objects的图片）

4. **Sampling strategies**

   核心 ： 从unlabeled dataset $\mathcal{X}_u$ 里选择N个sample。

   问题：只根据informativeness筛选数据的问题是，很容易挑选类似的图片，比如视频连续帧，这在自动驾驶领域是非常常见的。因此，我们还需要考虑数据的diversity。

   diversity：

   1）对unlabeled images，我们计算他们的embedding vector

   2）使用euclidiean distance、cosine similarity计算similarity matrix D

   基于这matrix，我们考虑3种筛选策略：】

   - **k-means++ (KMPP)**

     1. 随机初始化第一个中心$c$，把它加入$C$

     2. 对每个数据点 $x \in \mathcal{X} \backslash C$，计算它到最近的中心点的距离 $d_{min}(x) = min_{c \in C} d(x, c)$

     3. 加入一个新的中心点 $c_i$到C当中：根据distance和uncertainty score的probability distribution 

        $$ p(x) = s(x)d_{min}(x) / \sum_{x \in \mathcal{X}}s(x)d_{min}(x)$$

        离最近中心点最远的，uncertain的图片，最有可能被选为下一个centroid。

        *（从数学角度，为什么选这样一个函数呢？）*

   - **Core-set(CS)**

     从字面意思就可以理解：coreset是一个可以最好表征大数据集分布的小数据集。最早在Active Learning for Convolutional Neural Networks: A Core-Set Approach,被提出来。

     我们贪心算法逐步构建core-set，每次迭代，centroid $c_i$满足

     $$ c_i = arg\max\limits_{x \in \mathcal{X}}\min\limits_{c \in \mathcal{C}}s(x)d(x,c)$$ 

     离最近的中心点距离最大的数据点，会优先被筛选出来。和kmeans++一样，这里用score对distance进行加权

     *（core-set和kmeans++的区别是？迭代终止条件分别是）*

   - **Sparse Modeling(OMP)**

     参考 Uncertainty-based active learning via sparse modeling for image classification

     这种方法，旨在把uncertainty和diversity一起考虑。用一个linear combination把两个聚合为一个score function

     我们的目标，是从data pool里选出N的sample。这N个sample所能包含的information越接近整个data pool越好。

     $$\tilde{x} = min$$

## Experiment

<img src="Scalable Active Learning for Object Detection.assets/image-20210219160921173.png" alt="image-20210219160921173" style="zoom:50%;" />

实验设置：

- 847k train + 33k test。5 classes ：car, pedestrian, bicycle, traffic sign and traffic light.

- SSD网络，ensemble of 6 models，初始100k，随后每次200k数据

1. 使用5种不同scoring function + 不同的aggregation function

   *（score都是基于confidence map计算得到的）*

   可以发现，Entropy+Max，MI + Max的效果不错。Det-ent + sum aggregation的效果最好。但是它bias到有很多物体的图片上了， 所以标注成本依然很高。综合来看，MI似乎在trade-off上做得最好。

   这里MC-Droput的核心，及时test阶段依然保留dropout

2. 比较不同的Data Sampling方法

   - topN ： the most uncertain
   - bottom-N：the least uncertain
   - top-N/2 - bottom-N/2 ：combination of both
   - *top-third：combination between most uncertain and slightly easier samples*

   从table2可以看出，当困难样本越多，最后的效果越好。可以看出，topN方法的效果最好。所以**后续的实验，我们都用topN sampling。**

3. 进一步比较考虑**diversity** 的data sampling 方法

   用两种不同的embedding 方法：**DN 和 VGG**。

   - 前者用检测网络一样的backbone
   - 后者用的是VGG网络

   我们在backbone的最后一层卷积层做global averga pooling，分别得到160D和512D的embedding vector。

   同时， 我们还比较了euclidian和cosiine metric来衡量embeddings之间的距离。

   从 table III可以看出，在sampling的时候考虑diversity分别优于topN sampling 0.95%和1.5%。

4. active learning vs random

   在筛选数据的时候，考虑每次从全量数据池中筛选。这样会筛选重复数据，但是这其实是在增加困难样本的同时，变相减少了标注。（但是实际使用的时候呢？）

   从table IV可以看出，active learning一直优于random。同时，每次从全量数据而不是unlabeled dataset里选择方式，比其他方法都好。在第三次迭代中，active learning仅挑选42%的数据，就获得了比全量数据还好的效果。

   *这里用的是哪个方法？* ***max-MI + topN sampling**

<img src="Scalable Active Learning for Object Detection.assets/image-20210221145036720.png" alt="image-20210221145036720" style="zoom:50%;" />

5. 一个night-time  image A/B test

假设一个很大的unlabeled dataset。给一个模型，我们的目的是提高夜晚场景VRU（vulnerable road user， e.g. pedestrian，bicycle）的检测效果。

我们有一个850k的数据集，这里包含很少的夜晚场景且包含行人/自行车的图片。

<img src="Scalable Active Learning for Object Detection.assets/image-20210221151501317.png" alt="image-20210221151501317" style="zoom:30%;" />

我们通过人工和active learning分别选择一部分数据（19k），以应对night-time场景。

**Evaluation：**我们用cross validation验证（三次）。从新标注的19k数据里，我们把数据按照train/test=90/10分三次。

**Training：** 新训练集 = 原训练集 + 挑选数据集trainset；训练后测试分别在“新测试集”和“新测试+老测试集”上进行。

从TableV可以看出，手动挑选、al挑选数据并加入训练集，都提升了行人和自行车的的性能。

对行人类别，AL挑选的数据提升wMAP3.3%，而manual 挑选数据提升wMAP1.1%*（数据不符，应该是0.74%）*。对自行车，AL挑选的数据提升wMAP5.9%，而manual 挑选数据提升wMAP1.4%。

这都说明AL挑选出来的数据能够明显提升夜间场景的性能（正是我们的目的）

为了移除由筛选方式造成的测试集不同。我们同时又在manual testset上测试，依然发现AL的效果更好*（只是没有那么明显了，自行车3.2% vs 2.3%，行人1.28% vs 1.35%）*

我们发现两种方式挑选出来的数据，标注成本是差不多的（差异5%）

但是，我们发现，AL挑出来的数据包含更多的objects（12%），另外，我们发现AL挑选出的图片包含更多的行人和自行车，而比较少其他类别比如car。

*（所以用 **MaxMI + topN**就达到这么好的效果了？）*

总结：



本文提出一种scalable的active learning方法。核心就是用一个image-level scoring function来衡量每张new unlabeled image的informativeness，然后把这些最具有informativeness的图片挑出来标注、加入训练集。

1. 本文比较了不同scoring function + sampling function

2. 提升了night-time 场景下行人、自行车的检测性能。

