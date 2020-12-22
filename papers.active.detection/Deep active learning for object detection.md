

- White-box and Black-box active learning
  - Black-box : 不管网络结构，直接用bbox分类的score构建query function(uncertainty)
    - max-max-entropy(max of max entropy of box in all classes)
    - max-entropy (max of sum entropy per classes)
    - Sum-entropy (sum entropy of all classes)
  - White-box：知道网络结构什么样，会利用网络结构信息，比如中间feature map层构建query function(uncertainty)
    - 
- 

- 本文使用的active learning算法伪代码(**Margin**)：

```
对每张U中的图i
	用SSD检测i，得到每一类的bboxes $B_c$ 

​	对每一类c

​		对每个box b in B_c

​			如果conf<0.1 or >0.9 continue

​			找到产生这个bbox的feature map层s(source)

​			a_l = {}

​			对s上的所有bbox

​				如果bbox和b的overlap >j 则把该bbox加入A(辅助的bboxes)

​				a_l = a_l \and argmax_{a \in A} prob(a) 去所有和b重叠>j中prob最大的bbox加入a_l

​			second_max = max_{a \in a_l} prob(a) ：取a_l中prob最大的bbox的prob，赋值给second_max（？？？2nd物理含义不理解。因为b被选出来，意味着它的conf最高，剩下的top自然就是2nd了）

​			margin[b] = prob(b) - second_max 

​		margin[c] = 
```



