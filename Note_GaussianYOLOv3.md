# Note_Gaussian YOLOv3
[TOC]
## why ？
YOLOv3的输出包含3部分：objectness info, category info, bbox。objectness和category都是通过softmax/sigmoid classification得到的，有对应score可以用来建模uncertainty(值得指出的是，classification score不能直接用来作为uncertainty，需要calibration或者用其他方法，比如熵来建模)，但是bbox是regression得到的，没有对应的score来反应reliability of bbox。

## what？
### YOLOv3 bbox输出回顾
- 下图中蓝色框是预测结果，虚线框为anchor box，实线网格是feature map
- 最终的bbox输出为$b_x, b_y, b_w, b_h$,  而模型实际回归的是$t_x, t_y, t_w, t_h$， 计算loss时，也会把gt_box映射到$t_x^{gt}, t_y^{gt}, t_w^{gt}, t_h^{gt}$
- $(c_x, c_y )$是当前grid左上角角点的坐标，$\sigma(t_x), \sigma(t_y)$是prior box中心点相对于左上角角点的坐标
- $p_w, p_h$ 是prior box的宽和高

<img src="https://tva1.sinaimg.cn/large/007S8ZIlly1ghg6e9cdi2j30o80nk77n.jpg" alt="image-20200805193456177" style="zoom:40%;" />


### Gaussian modeling

<img src="https://tva1.sinaimg.cn/large/007S8ZIlly1ghg6e9y3thj30no0eydhy.jpg" alt="image-20200805153518552" style="zoom:50%;" />

在bbox regression中，总共需要回归4个变量($t_x, t_y, t_w, t_h$)。我们可以这4个变量的gaussian model来计算每个bbox的uncertainty。具体来说，输入x，对每个变量y，我们有
$$
p(y|x) = N(y;\mu(x), \sum(x))
$$
其中$\mu(x)$和$\sum(x)$是均值和方差。

为了输出bbox的uncertainty，我们对每个bbox变量进行gaussian modeling，即模型输出bbox时不直接输出$t_x, t_y, t_w, t_h$，而是输出$\hat{\mu}_{t_x}, \hat{\sum}_{t_x}, \hat{\mu}_{t_y}, \hat{\sum}_{t_y}, \hat{\mu}_{t_w}, \hat{\sum}_{t_w},  \hat{\mu}_{t_h}, \hat{\sum}_{t_h}$。 考虑YOLOv3的输出层，我们用sigmoid处理$t_x, t_y, t_w, t_h$

$$
\mu_{t_x} = \sigma(\hat{\mu}_{t_x}), \mu_{t_y} = \sigma(\hat{\mu}_{t_y}), \mu_{t_w} = \hat{\mu}_{t_w}, \mu_{t_h} = \hat{\mu}_{t_h}
$$

$$
{\sum}_{t_x}= \sigma(\hat{\sum}_{t_x}), 
{\sum}_{t_y}= \sigma(\hat{\sum}_{t_y}) \\
{\sum}_{t_w}= \sigma(\hat{\sum}_{t_w}),
{\sum}_{t_h}= \sigma(\hat{\sum}_{t_h}),
$$

$$
\sigma(x) = \frac{1}{1+exp^{(-x)}}
$$

其中均值就是最后bbox的坐标，方差就是bbox的uncertainty。需要注意的是，$t_x, t_y$必须是bbox的中心，所以我们用sigmoid function来归一化（与YOLOv3一致），对$t_x, t_y, t_w, t_h$的variance我们也用sigmoid处理（归一化到0-1之间方便建模uncertainty，如何计算loss？）。但是由于$t_w, t_h$是通过prior box + offset得到的，我们不用sigmoid处理他们（因为他们可正可负）。

<img src="https://tva1.sinaimg.cn/large/007S8ZIlly1ghg6e91bn6j31ay0kaaed.jpg" alt="image-20200805160928497" style="zoom:50%;" />

计算量

yolov3：99x10^9  Flops, gaussian yolov3 : 99.04x10^9  Flops 增加了4%的计算量



## how?

### Reconstruction of loss function

对objectness和category，YOLOv3使用cross_entroy loss。对bbox，YOLOv3使用sum square loss。

在Gaussian YOLOv3中，因为bbox的坐标已经用gaussian modeling，loss也应该改为negtive log likelihood（NLL） loss：
$$
L_x = -\sum_{i=1}^W\sum_{j=1}^H\sum_{k=1}^K\gamma_{ijk}\log(
N(x_{ijk}^G|\mu_{t_x}(x_{ijk}), {\sum}_{t_x}(x_{ijk}))
+\epsilon)
$$
$L_x$是x的NLL loss，对$y,w,h$同样可以计算$L_y,L_w,L_h$。其中，$N(x_{ijk}^G|\mu_{t_x}(x_{ijk}), {\sum}_{t_x}(x_{ijk}))$是高斯概率密度函数，W,H是用于生成回归box的feature map的宽和高，K是anchors数量。更进一步地，$\mu_{t_x}(x_{ijk})$就是最后bbox中心点横坐标$t_x$，是在$(i,j)$这个位置的第k个ancho被offset修正后得到的。${\sum}_{t_x}(x_{ijk})$也是网络输出值，代表$t_x$这个坐标的uncertainty。



### Code

#### model output

```python
class YOLOLayer(nn.Module):
    """
    detection layer corresponding to yolo_layer.c of darknet
    """
    def __init__(
        self, config_model, layer_no, in_ch, ignore_thre=0.7):
        """
        Args:
            config_model (dict) : model configuration.
                ANCHORS (list of tuples) :
                ANCH_MASK:  (list of int list): index indicating the anchors to be
                    used in YOLO layers. One of the mask group is picked from the list.
                N_CLASSES (int): number of classes
                GAUSSIAN (bool): predict uncertainty for each of xywh coordinates in Gaussian YOLOv3 way.
                    For Gaussian YOLOv3, see https://arxiv.org/abs/1904.04620
            layer_no (int): YOLO layer number - one from (0, 1, 2).
            in_ch (int): number of input channels.
            ignore_thre (float): threshold of IoU above which objectness training is ignored.
        """

        super(YOLOLayer, self).__init__()
        strides = [32, 16, 8] # fixed
        self.anchors = config_model['ANCHORS']
        self.anch_mask = config_model['ANCH_MASK'][layer_no]
        self.n_anchors = len(self.anch_mask)
        self.n_classes = config_model['N_CLASSES']
        self.gaussian = config_model['GAUSSIAN']
        self.ignore_thre = ignore_thre
        self.stride = strides[layer_no]
        all_anchors_grid = [(w / self.stride, h / self.stride)
                                 for w, h in self.anchors]
        self.masked_anchors = [all_anchors_grid[i]
                               for i in self.anch_mask]
        self.ref_anchors = np.zeros((len(all_anchors_grid), 4))
        self.ref_anchors[:, 2:] = np.array(all_anchors_grid)
        self.ref_anchors = torch.FloatTensor(self.ref_anchors)

        channels_per_anchor = 5 + self.n_classes  # 5: x, y, w, h, objectness
        if self.gaussian:
            print('Gaussian YOLOv3')
            channels_per_anchor += 4  # 4: xywh uncertainties 
        self.conv = nn.Conv2d(in_channels=in_ch,
                              out_channels=channels_per_anchor * self.n_anchors,
                              kernel_size=1, stride=1, padding=0)
```

- YOLOv3基于strides = [32, 16, 8]的三个feature map生成anchor box，**对每个anchor新增4个channel用于分别回归xywh uncertainties**

```python
class YOLOLayer(nn.Module):
  	# ………………
		def forward(self, xin, labels=None):
        """
        In this
        Args:
            xin (torch.Tensor): input feature map whose size is :math:`(N, C, H, W)`, \
                where N, C, H, W denote batchsize, channel width, height, width respectively.
            labels (torch.Tensor): label data whose size is :math:`(N, K, 5)`. \
                N and K denote batchsize and number of labels.
                Each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
        Returns:
            loss (torch.Tensor): total loss - the target of backprop.
            loss_xy (torch.Tensor): x, y loss - calculated by binary cross entropy (BCE) \
                with boxsize-dependent weights.
            loss_wh (torch.Tensor): w, h loss - calculated by l2 without size averaging and \
                with boxsize-dependent weights.
            loss_obj (torch.Tensor): objectness loss - calculated by BCE.
            loss_cls (torch.Tensor): classification loss - calculated by BCE for each class.
            loss_l2 (torch.Tensor): total l2 loss - only for logging.
        """
        output = self.conv(xin)

        batchsize = output.shape[0]
        fsize = output.shape[2]
        n_ch = 5 + self.n_classes  # channels per anchor w/o xywh unceartainties
        dtype = torch.cuda.FloatTensor if xin.is_cuda else torch.FloatTensor

        output = output.view(batchsize, self.n_anchors, -1, fsize, fsize)
        output = output.permute(0, 1, 3, 4, 2)  # shape: [batch, anchor, grid_y, grid_x, channels_per_anchor]

        if self.gaussian:
            # logistic activation for sigma of xywh
            sigma_xywh = output[..., -4:]  # shape: [batch, anchor, grid_y, grid_x, 4(= xywh uncertainties)]
            sigma_xywh = torch.sigmoid(sigma_xywh)

            output = output[..., :-4]
        # output shape: [batch, anchor, grid_y, grid_x, n_class + 5(= x, y, w, h, objectness)]

        # logistic activation for xy, obj, cls
        output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(
            output[..., np.r_[:2, 4:n_ch]])

        # calculate pred - xywh obj cls

        x_shift = dtype(np.broadcast_to(
            np.arange(fsize, dtype=np.float32), output.shape[:4]))
        y_shift = dtype(np.broadcast_to(
            np.arange(fsize, dtype=np.float32).reshape(fsize, 1), output.shape[:4]))

        masked_anchors = np.array(self.masked_anchors)

        w_anchors = dtype(np.broadcast_to(np.reshape(
            masked_anchors[:, 0], (1, self.n_anchors, 1, 1)), output.shape[:4]))
        h_anchors = dtype(np.broadcast_to(np.reshape(
            masked_anchors[:, 1], (1, self.n_anchors, 1, 1)), output.shape[:4]))

```



#### GT encode

 TODO

#### Loss

```python
class YOLOLayer(nn.Module):
  	# ………………
		def forward(self, xin, labels=None):
  			# …………………
        # loss calculation
        output[..., 4] *= obj_mask
        output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask

        target[..., 4] *= obj_mask
        target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask

        loss_obj = F.binary_cross_entropy(output[..., 4], target[..., 4], reduction='sum')
        loss_cls = F.binary_cross_entropy(output[..., 5:], target[..., 5:], reduction='sum')

        if self.gaussian:
            loss_xy = - torch.log(
                self._gaussian_dist_pdf(output[..., :2], target[..., :2], sigma_xywh[..., :2]) + 1e-9) / 2.0
            loss_wh = - torch.log(
                self._gaussian_dist_pdf(output[..., 2:4], target[..., 2:4], sigma_xywh[..., 2:4]) + 1e-9) / 2.0
        else:
            loss_xy = F.binary_cross_entropy(output[..., :2], target[..., :2], reduction='none')
            loss_wh = F.mse_loss(output[..., 2:4], target[..., 2:4], reduction='none') / 2.0
        loss_xy = (loss_xy * tgt_scale).sum()
        loss_wh = (loss_wh * tgt_scale).sum()

        loss = loss_xy + loss_wh + loss_obj + loss_cls

    def _gaussian_dist_pdf(self, val, mean, var):
        return torch.exp(- (val - mean) ** 2.0 / var / 2.0) / torch.sqrt(2.0 * np.pi * var)
```

在回归bbox($t_x, t_y, t_w, t_h$)时，YOLOv3使用MSE loss。在Gaussian YOLOv3中，由于输出在x,y,w,h基础上新增了他们的uncertainty，loss也相应改变，这里使用NLL loss。参考前面的公式，其中$N(x_{ijk}^G|\mu_{t_x}(x_{ijk}), {\sum}_{t_x}(x_{ijk}))$是高斯分布的概率密度函数，对应代码参考上面最后的`_gaussian_dist_pdf`