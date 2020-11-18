# CenterNet : Objects as Points 笔记

- 网络特点
  - 在更大的feature map上做预测（4 VS 16）
  - 对heatmap每个位置，直接预测C+4的vector：C(num_classes), O(offset of center), S(size)
  - 把gt按照高斯分布，沿着hwc三个维度，洒在gt center point周围，用focal loss计算损失。对于GT重叠的位置，取大的为准
- 优点
  - 速度快 
    - res18@147fps@28.1%AP
    - DLA34@52fps@37.4%AP
    - Hourglass101@1.4fps@45.1%AP 
  - 无需NMS
  - 网络结构可拓展至3d bbox预测、人体关键点预测等问题
- 确定
  - 对大物体（超过1/3图像面积，大公交车）的框检测不好
  - 对中心重叠物体检测效果不太好
- 模型训练采用标准的监督学习，推理仅仅是单个前向传播网络，不存在NMS这类后处理。
- inference
  - 取heatmap上每个类别的前100峰值点：将heatmap上所有响应点与其连接的8个临近点相比较，如果响应点的响应>=临近点的值，则保留。
  - 令$ \hat{P_c} $是检测到的c类别的中心点的集合，$\hat{P}=\{\hat{x_i}, \hat{y_i}\}_{i=1}^n$。每个关键点坐标以整型坐标$（\hat{x_i}, \hat{y_i}）$的形式给出。$\hat{Y}_{x_iy_ic}$为inference conf。产生如下bbox：$$( \hat{x_i} + \delta\hat{x_i} - \hat{w_i}/2, \ \hat{y_i} + \delta\hat{y_i} - \hat{h_i}/2,\\ \hat{x_i} + \delta\hat{x_i} + \hat{w_i}/2, \ \hat{y_i} + \delta\hat{y_i} + \hat{h_i/2}) $$ 



## gaussian

```python
class CTDetDataset(data.Dataset):
  def __getitem__(self, index):
    img_id = self.images[index]
    file_name = self.coco_loadImgs(ids=[img_id])[0]['file_name']
    img_path = os.path.join(self.img_dir, filename)
    ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    anns = self.coco.loadAnns(ids=ann_ids)
    num_objs = min(len(anns), self.max_objs)
```

