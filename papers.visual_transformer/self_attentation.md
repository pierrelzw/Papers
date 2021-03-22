# self_attention

疑问：

- q, k attention已经实现取部分分支的的结果了，为什么还要v？有点类似skip-connection？

- multi-head的一个head的区别？

## 为什么要 self_attation

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gnqfq3weubj315e0rqdj6.jpg" alt="image-20210217090425326" style="zoom:50%;" />

过去一想到sequence问题，就想用RNN来解决（seq2seq）。但是RNN不利于并行，因为sequence内容有前后依赖关系。于是，有人提出用CNN来替代RNN。不过这样的话，也有问题，CNN能处理的长度有限，于是想到用堆叠CNN的方法来处理长sequence（堆叠之后感受野增加了）。

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gnqfq1tdojj313f0u0q7x.jpg" alt="image-20210217090831275" style="zoom:50%;" />

但是一味地堆叠CNN的方法是有局限的。因为这意味着需要随着sequence长度变长不停地增加CNN堆叠数，这样的条件不利于模型设计。

## 什么是self_attation

所以有了一个新想法：self_attention：

> 是一种新的layer 
>
> 和RNN一样可以处理sequence问题
>
> 支持并行化，每个输出$b_1 - b_4$都是基于整个输入sequence计算得到的，且可以并行输出
>
> scalability比较好，能够处理比较长的序列

<img src="https://tva1.sinaimg.cn/large/008eGmZEly1gnqfq2u6ncj313u0is0vt.jpg" alt="image-20210217091245065" style="zoom:50%;" />



## self-atttention具体是怎么做的？

<img src="https://pic3.zhimg.com/80/v2-8537e0996a586b7c37d5e345b6c4402a_1440w.jpg" alt="img" style="zoom:50%;" />

<img src="https://pic2.zhimg.com/80/v2-f7b03e1979c6ccd1dab4b579654c8cd5_1440w.jpg" alt="img" style="zoom:50%;" />

假设input为$x_1 - x_4$, 对输入乘上一个矩阵W得到embedding $a_1 - a_4$。然后这4个向量进入self-attention。

> 1. 给embedding乘上三个向量Q，K，V。以$a_1$为例，乘完以后得到$q_1, k_1, v_1$
>
> 2. 拿每个query q去对每个key k做attention $\alpha_{i,j} = q_i·k_j/\sqrt{d}$，其中d是Q和K的dimension；实际操作是把Q和K做scaled inner product（Mat_4x1 * Mat_1x4 -> Mat_4x4 ），得到$\alpha$
>
> 3. 对$\alpha$取softmax，得到两个向量有多近的衡量$\hat\alpha$（attention就是看两个向量有多近）
>
> 4. 用$\hat\alpha$乘向量V得到向量B；如果希望输出$b_i$使用整个sequence的信息，那么只要学习相应的$\hat\alpha_{i,j} != 0$, 如果要考虑local information，就学习出相应的$\hat\alpha_{i,j} = 0$ （这一步有点像skip connection）

用矩阵表示计算：

<img src="https://pic3.zhimg.com/v2-6cc342a83d25ac76b767b5bbf27d9d6e_r.jpg" alt="preview" style="zoom:50%;" />![img](https://pic2.zhimg.com/80/v2-52a5e6b928dc44db73f85001b2d1133d_1440w.jpg)

<img src="https://pic3.zhimg.com/v2-6cc342a83d25ac76b767b5bbf27d9d6e_r.jpg" alt="preview" style="zoom:50%;" />![img](https://pic2.zhimg.com/80/v2-52a5e6b928dc44db73f85001b2d1133d_1440w.jpg)

<img src="https://pic4.zhimg.com/80/v2-1b7d30f098f02488c48c3601f8e13033_1440w.jpg" alt="img" style="zoom:50%;" />

![preview](https://pic2.zhimg.com/v2-8628bf2c2bb9a7ee2c4a0fb870ab32b9_r.jpg)

### multi-head self-attention

考虑多个head的情况，每个head和之前的过程一样。以2head为例，分别有两个(Q、K、V）向量组合对输入embedding进行乘积处理。进而得到两个output $B$

<img src="https://pic1.zhimg.com/80/v2-688516477ad57f01a4abe5fd1a36e510_1440w.jpg" alt="img" style="zoom:50%;" />

<img src="https://pic3.zhimg.com/80/v2-b0891e9352874c9eee469372b85ecbe2_1440w.jpg" alt="img" style="zoom:50%;" />

然后，我们把不同head的输出进行concat，再用一个transform matrix把concat向量转化为和输入一样长度的final output。

<img src="https://pic1.zhimg.com/80/v2-df5d332304c2fd217705f210edd18bf4_1440w.jpg" alt="img" style="zoom:33%;" />

<img src="https://pic2.zhimg.com/80/v2-f784c73ae6eb34a00108b64e3db394fd_1440w.jpg" alt="img" style="zoom:50%;" />

这里有一组multi-head的结果，可看出，绿色head(一组q、k、v)更关注global的信息，而红色的head(一组q、k、v)更关注local的信息。

<img src="https://pic3.zhimg.com/v2-6b6c906cfca399506d324cac3292b04a_r.jpg" alt="preview" style="zoom:50%;" />

### positional encoding

现在的self-attention已经可以在并行的前提下，处理长序列输入了。但是它还没有位置信息：输入一个单词向量，对输出向量的每个元素而言，“近处”和“远处”的效果是一样的。

没有表示位置的信息(No position information in self attention)，所以你输入“A打了B”或者“B打了A”，self-attention是无法区分的。

原文paper作者的做法：给每个embedding+一个one-hot vector表示位置信息。



<img src="https://pic3.zhimg.com/80/v2-b8886621fc841085300f5bb21de26f0e_1440w.jpg" alt="img" style="zoom:50%;" />

<img src="https://pic4.zhimg.com/80/v2-7814595d02ef37cb762b3ef998fae267_1440w.jpg" alt="img" style="zoom:50%;" />

具体做法：给每个位置规定一个表示位置信息的向量 $e^i$ （*向量？*）, 让他和$a^i$相加后再输入self-attention进行处理。

**为什么是 ![[公式]](https://www.zhihu.com/equation?tex=e%5Ei) 与 ![[公式]](https://www.zhihu.com/equation?tex=a%5Ei) 相加？为什么不是concatenate？加起来以后，原来表示位置的资讯不就混到 ![[公式]](https://www.zhihu.com/equation?tex=a%5Ei) 里面去了吗？不就很难被找到了吗？**

实际上，如果我们给输入x append一个one-hot向量p，那么我们有：

 <img src="https://tva1.sinaimg.cn/large/008eGmZEly1gnqfq2cj2nj30nu03m74h.jpg" alt="image-20210217113755843" style="zoom:50%;" />

所以给$a^i$直接加$e^i$，实际上等价于给输入添加一个one-hot编码，然后再做embedding。

Tranformer中除了单词的embedding，还需要使用位置embedding表示单词出现在句子中的位置，这对NLP任务非常重要。

位置embedding用PE表示，PE的维度与输入embedding的维度一样。PE可以通过训练得到，也可以通过某种公式计算得到。在Transformer设计中采用了后者：

$$PE_{(pos, 2i)} = sin(pos / 10000 ^{2i/d_{model}})$$

$$PE_{(pos, 2i+1)} = cos(pos / 10000 ^{2i/d_{model}})$$

其中，pos表示token在sequence中的位置，例如第一个token“我”的pos=0。

当pos=1时：

$$PE(1) = [sin(1/10000^{0/512}), cos(1/10000^{0/512}),sin(1/10000^{2/512}), cos(1/10000^{2/512}), ...]$$

其中$d_{model} = 512$, 底数是10000，*为什么是10000？*

*$d_{model}$ ???*

这个式子的好处：

- 每个位置有一个唯一的PE

- 是PE能够适应比训练即里面所有句子更长的句子，假设训练集里面最长的句子是有20个单词，突然来了一个长度为21的句子，则使用公式可以计算出第21为的Embedding。

- 可以让模型更容易地计算出相对位置，对于固定的长度的间距k，任意位置的$PE_{pos+k}$都可以被$PE_{pos}]$的线性函数表示，因为三角函数特性：

  $$cos(pos + k) = cos(pos)cos(k) - sin(pos)sin(k)$$

  $$sin(pos + k) = sin(pos)cos(k) + cos(pos)sin(k)$$

## 如何使用self_attation

在seq2seq问题中，如何使用self-attention？把Encoder-Decoder中的RNN用self-attention替换掉。

<img src="https://pic4.zhimg.com/80/v2-287ebca58558012f9459f3f1d5bc3827_1440w.jpg" alt="img" style="zoom:50%;" />

## Transformer代码解析

<img src="https://pic4.zhimg.com/80/v2-1719966a223d98ad48f98c2e4d71add7_1440w.jpg" alt="img" style="zoom:50%;" />

为什么是layer normalization？