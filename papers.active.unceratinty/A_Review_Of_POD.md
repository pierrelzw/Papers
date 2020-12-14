- Uncertainty estimation

  - 
  - MCDropOut

  $$
  p(y|x, \mathcal{D}) \approx \frac{1}{T}\sum_{t=1}^{T}p(y|x, W_t)
  $$

  其中，T为进行多次dropout 前向计算的次数

  - Deep ensemble
    $$
    p(y|x, \mathcal{D}) \approx \frac{1}{M}\sum_{m=1}^{M}p(y|x, W_m)
    $$
    其中，M是不同的模型weights，由同样的模型不同初始化得到

  - 分析cls uncertainty常用的是softmax输出

  - 分析regression uncertainty则常用gaussian distribution 或者是Gaussian Mixture Model

  - `Optimizing a probability network using Direct Modeling can be view as a maximum likelihood problem, where W are optimized to maximize the observation likelihood of training data.`

    对训练数据（x,y），negative log likelihood : $L(x, W) = -\log(p(y|x, W))$，这其实就是CrossEntropyLoss。

    对Regression问题来说

    
    $$
    L(x,W) = \frac{(y-\hat{\mu}(x, W))^2}{2\hat\sigma^2(x,W)} + \frac{log{\sigma^2(x, W)}}{2}
    $$
    $L(x,W)$ 可以认为是标准的$L_2\ Loss(i.e. (y-\hat{\mu}(x, W))^2)$ 通过  $\hat\sigma^2(x,W)$加权处理然后又被$\log{\hat\sigma^2(x,W)}$ 进行regularization。

- Uncertainty Evaluation

  - Shannon Entropy

  - Mutual Information: Entropy - E（Entropy）

    - 第二项来自MCDropout或者是DeepEnsembles

  - Calibration Plot

  - Negative Log Likelihood
    $$
    NLL = \sum_{n=1}^{N_{test}}\log(p(y_n|x_n, \mathcal{D}))
    $$
    

    其中，$ y_n$是GT，NLL越低则模型预测分布对test GT的的拟合越好。

    - NLL间接表示了Uncertainty Calibration的效果

  - Brier Score
    $$
    BS = \frac{1}{N_{test}}\sum_{n=1}^{N_{test}}\sum_{c=1}{C}(\hat{s}_{n,c} - y_{n,c})^2
    $$
     其中 $\hat{s}_{n,c}$ pre

  - Error Curve

    - 如果把预测结果按照Uncertainty（NLL，ECE， MI，Shannon Entropy）排序，那么如果逐步删除那些Uncertainty最高的sample，剩下数据的average error应该逐步降低。
    - 以被删除sample的占比为横轴，avg error(e.g.  cross entropy error for CS, MSE for regression)为纵轴，画图

  - Total Variance 

    - 用来衡量regression task中概率分布的dispersion，通过算covariance matrix的trace(对所有对角矩阵上的元素求和)
    - TV只能衡量单个variable的variance，但是不计算不同regression 变量之间的correlation

  上述uncertainty分析方法，都是在面对domain shift（train\test distribution不同）得到的。真实的问题，往往是一个open-set问题，比如自动驾驶。我们的感知系统总会遇到一些在数据集中没有见到过的物体。

  我们从过往的工作中发现两点：

  - Uncertainty 和模型性能存在trade-off，也就是说一个Uncertainty分析比较好的模型，其准确率、mAP可能会比较低
  - 并没有Uncertainty哪种方法就一定好于另一种。这个领域仍是一个比较新的领域，尤其是在存在domain shift、需要一个大模型来解决问题的情况下。

- POD（probabilistic object detection）

  - 传统的OD算法，一般只输出bbox 但是没有Uncertainty，并输出softmax分类结果。也就是说算法的输出是deterministic的，算法只输出看到了什么，但是不输出它对结果有多确定。
  - 我们使用一阶段网络，但是其实一阶段二阶段的本质一样。（无非是分类、回归问题）
  - 我们先用一个比较好的detect net提取特征，然后在head出添加MCDropOut或者ensemble来进行epitemic uncertainty建模。这也是大多数人工作的做法。
  - 而对 bbox的uncertainty（aleatoric），通过添加额外的输出，用NN直接预测来实现。
  - 

- POD with Epistemic Uncertainty  

- POD with Aleatoric Uncertainty 

- POD with Aleatoric & Epistemic Uncertainty 

  - Direct modeling + MCDropOut/Deep ensembles

    - 假设bbox Gaussian Distributed

    - 用GMM计算bbox mean、var
      $$
      \hat\mu(x) = \frac{1}{T}\sum_{t=1}^{T}\hat\mu(x, W_t)
      $$

      $$
      \hat\sigma^2(x) = \hat\sigma_e^2(x) + \hat\sigma_a^2(x)
      $$

      $$
      \hat\sigma_e^2(x) = \frac{1}{T}\sum_{t=1}^{T}(\hat\mu(x, W_t)) ^2- (\hat\mu(x))^2
      $$

      $$
      \hat\sigma_a^2(x) = \frac{1}{T}\sum_{t=1}^{T}\hat\sigma^2(x,W_t)
      $$

      其中，$\hat\mu(x, W_t) = f(x, W_t)$，$\sigma^2(x, W_t)$是 bbox或者classification logit vector的mean和var; 

      $\hat\sigma_e^2(x)$则是epistemic uncertainty，用T次输出的sample variance建模

      $\hat\sigma_a^2(x)$则是aleatoric uncertainty，用T次预测variance（$\hat\sigma^2(x,wW_t)$）的平均值建模

      需要注意的是，

## code



```python
class ProbabilisticRetinaNetHead(RetinaNetHead):
    """
    The head used in ProbabilisticRetinaNet for object class probability estimation, box regression, box covariance estimation.
    It has three subnets for the three tasks, with a common structure but separate parameters.
    """

    def __init__(self,
                 cfg,
                 use_dropout,
                 dropout_rate,
                 compute_cls_var,
                 compute_bbox_cov,
                 bbox_cov_dims,
                 input_shape: List[ShapeSpec]):
        super().__init__(cfg, input_shape)

        # Extract config information
        # fmt: off
        in_channels = input_shape[0].channels
        num_classes = cfg.MODEL.RETINANET.NUM_CLASSES
        num_convs = cfg.MODEL.RETINANET.NUM_CONVS
        prior_prob = cfg.MODEL.RETINANET.PRIOR_PROB
        num_anchors = build_anchor_generator(cfg, input_shape).num_cell_anchors
        # fmt: on
        assert (
            len(set(num_anchors)) == 1
        ), "Using different number of anchors between levels is not currently supported!"
        num_anchors = num_anchors[0]

        self.compute_cls_var = compute_cls_var
        self.compute_bbox_cov = compute_bbox_cov
        self.bbox_cov_dims = bbox_cov_dims

        # For consistency all configs are grabbed from original RetinaNet
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate

        # 原始retinaNet head : cls_subnet + bbox_subnet
        cls_subnet = []
        bbox_subnet = []
          
        for _ in range(num_convs):
            cls_subnet.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1))
            cls_subnet.append(nn.ReLU())

            bbox_subnet.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1))
            bbox_subnet.append(nn.ReLU())

            if self.use_dropout:
                cls_subnet.append(nn.Dropout(p=self.dropout_rate))
                bbox_subnet.append(nn.Dropout(p=self.dropout_rate))

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        
        # 原始retinaNet head : cls_subnet + bbox_subnet
        self.cls_score = nn.Conv2d(
            in_channels,
            num_anchors * num_classes,
            kernel_size=3,
            stride=1,
            padding=1)
        self.bbox_pred = nn.Conv2d(
            in_channels,
            num_anchors * 4,
            kernel_size=3,
            stride=1,
            padding=1)

        for modules in [
                self.cls_subnet,
                self.bbox_subnet,
                self.cls_score,
                self.bbox_pred]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

        # Create subnet for classification variance estimation.
        if self.compute_cls_var:
            self.cls_var = nn.Conv2d(
                in_channels,
                num_anchors * num_classes,
                kernel_size=3,
                stride=1,
                padding=1)

            for layer in self.cls_var.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, -10.0)

        # Create subnet for bounding box covariance estimation.
        if self.compute_bbox_cov:
            self.bbox_cov = nn.Conv2d(
                in_channels,
                num_anchors * self.bbox_cov_dims,
                kernel_size=3,
                stride=1,
                padding=1)

            for layer in self.bbox_cov.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.0001)
                    torch.nn.init.constant_(layer.bias, 0)

```

```python
    def losses(
            self,
            anchors,
            gt_classes,
            gt_boxes,
            pred_class_logits,
            pred_anchor_deltas,
            pred_class_logits_var=None,
            pred_bbox_cov=None):
        """
        Args:
            For `gt_classes` and `gt_anchors_deltas` parameters, see
                :meth:`RetinaNet.get_ground_truth`.
            Their shapes are (N, R) and (N, R, 4), respectively, where R is
            the total number of anchors across levels, i.e. sum(Hi x Wi x A)
            For `pred_class_logits`, `pred_anchor_deltas`, `pred_class_logits_var` and `pred_bbox_cov`, see
                :meth:`RetinaNetHead.forward`.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a scalar tensor
                storing the loss. Used during training only. The dict keys are:
                "loss_cls" and "loss_box_reg"
        """
        num_images = len(gt_classes)
        gt_labels = torch.stack(gt_classes)  # (N, R)
        anchors = type(anchors[0]).cat(anchors).tensor  # (R, 4)
        gt_anchor_deltas = [self.box2box_transform.get_deltas(anchors, k) for k in gt_boxes]
        gt_anchor_deltas = torch.stack(gt_anchor_deltas)  # (N, R, 4)

        valid_mask = gt_labels >= 0
        pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)
        num_pos_anchors = pos_mask.sum().item()
        get_event_storage().put_scalar("num_pos_anchors", num_pos_anchors / num_images)
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
            1 - self.loss_normalizer_momentum
        ) * max(num_pos_anchors, 1)

        # classification and regression loss

        # Shapes:
        # (N x R, K) for class_logits and class_logits_var.
        # (N x R, 4), (N x R x 10) for pred_anchor_deltas and pred_class_bbox_cov respectively.

        # Transform per-feature layer lists to a single tensor
        pred_class_logits = cat(pred_class_logits, dim=1)
        pred_anchor_deltas = cat(pred_anchor_deltas, dim=1)

        if pred_class_logits_var is not None:
            pred_class_logits_var = cat(
                pred_class_logits_var, dim=1)

        if pred_bbox_cov is not None:
            pred_bbox_cov = cat(
                pred_bbox_cov, dim=1)

        gt_classes_target = torch.nn.functional.one_hot(gt_labels[valid_mask], num_classes=self.num_classes + 1)[
                           :, :-1
                           ].to(pred_class_logits[0].dtype)  # no loss for the last (background) class

        # Classification losses
        if self.compute_cls_var:
            # Compute classification variance according to:
            # "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?", NIPS 2017
            if self.cls_var_loss == 'loss_attenuation':
                num_samples = self.cls_var_num_samples
                # Compute standard deviation
                pred_class_logits_var = torch.sqrt(torch.exp(
                    pred_class_logits_var[valid_mask]))

                pred_class_logits = pred_class_logts[valid_mask]

                # Produce normal samples using logits as the mean and the standard deviation computed above
                # Scales with GPU memory. 12 GB ---> 3 Samples per anchor for
                # COCO dataset.
                univariate_normal_dists = distributions.normal.Normal(
                    pred_class_logits, scale=pred_class_logits_var)

                pred_class_stochastic_logits = univariate_normal_dists.rsample(
                    (num_samples,))
                pred_class_stochastic_logits = pred_class_stochastic_logits.view(
                    (pred_class_stochastic_logits.shape[1] * num_samples, pred_class_stochastic_logits.shape[2], -1))
                pred_class_stochastic_logits = pred_class_stochastic_logits.squeeze(
                    2)

                # Produce copies of the target classes to match the number of
                # stochastic samples.
                gt_classes_target = torch.unsqueeze(gt_classes_target, 0)
                gt_classes_target = torch.repeat_interleave(
                    gt_classes_target, num_samples, dim=0).view(
                    (gt_classes_target.shape[1] * num_samples, gt_classes_target.shape[2], -1))
                gt_classes_target = gt_classes_target.squeeze(2)

                # Produce copies of the target classes to form the stochastic
                # focal loss.
                loss_cls = sigmoid_focal_loss_jit(
                    pred_class_stochastic_logits,
                    gt_classes_target,
                    alpha=self.focal_loss_alpha,
                    gamma=self.focal_loss_gamma,
                    reduction="sum",
                ) / (num_samples * max(1, self.loss_normalizer))
            else:
                raise ValueError(
                    'Invalid classification loss name {}.'.format(
                        self.bbox_cov_loss))
        else:
            # Standard loss computation in case one wants to use this code
            # without any probabilistic inference.
            loss_cls = sigmoid_focal_loss_jit(
                pred_class_logits[valid_mask],
                gt_classes_target,
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction="sum",
            ) / max(1, self.loss_normalizer)

        # Compute Regression Loss
        pred_anchor_deltas = pred_anchor_deltas[pos_mask]
        gt_anchors_deltas = gt_anchor_deltas[pos_mask]
        if self.compute_bbox_cov:
            if self.bbox_cov_loss == 'negative_log_likelihood':
                if self.bbox_cov_type == 'diagonal':
                    # Compute regression variance according to:
                    # "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?", NIPS 2017

                    # This is the log of the variance. We have to clamp it else negative
                    # log likelihood goes to infinity.
                    pred_bbox_cov = torch.clamp(
                        pred_bbox_cov[pos_mask], -7.0, 7.0)

                    loss_box_reg = 0.5 * torch.exp(-pred_bbox_cov) * smooth_l1_loss(
                        pred_anchor_deltas,
                        gt_anchors_deltas,
                        beta=self.smooth_l1_beta)

                    loss_covariance_regularize = 0.5 * pred_bbox_cov
                    loss_box_reg += loss_covariance_regularize

                    loss_box_reg = torch.sum(
                        loss_box_reg) / max(1, self.loss_normalizer)
            else:
                raise ValueError(
                    'Invalid regression loss name {}.'.format(
                        self.bbox_cov_loss))

            # Perform loss annealing.
            standard_regression_loss = smooth_l1_loss(
                pred_anchor_deltas,
                gt_anchors_deltas,
                beta=self.smooth_l1_beta,
                reduction="sum",
            ) / max(1, self.loss_normalizer)
            probabilistic_loss_weight = min(1.0, self.current_step/self.annealing_step)
            probabilistic_loss_weight = (100**probabilistic_loss_weight-1.0)/(100.0-1.0)
            loss_box_reg = (1.0 - probabilistic_loss_weight)*standard_regression_loss + probabilistic_loss_weight*loss_box_reg
        else:
            # Standard regression loss in case no variance is needed to be
            # estimated
            loss_box_reg = smooth_l1_loss(
                pred_anchor_deltas,
                gt_anchors_deltas,
                beta=self.smooth_l1_beta,
                reduction="sum",
            ) / max(1, self.loss_normalizer)

        return {"loss_cls": loss_cls, "loss_box_reg": loss_box_reg}
```

