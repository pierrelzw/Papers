# CenterNet loss完全理解

- Focal Loss中，如何定义难易样本？score接近1就是easy，接近0就是hard？
- 人体关键点检测中，loss如何定义，RegWeiightLoss是什么？



## 关键点检测

```python
loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
			 opt.off_weight * off_loss + opt.hp_weight * hp_loss + \
		   opt.hm_hp_weight * hm_hp_loss + opt.off_weight * hp_offset_loss
```

总共包含6个loss：**hm_loss, wh_loss, off_loss, hp_loss, hm_hp_loss, hp_offset_loss**

先看看multi_pose_loss 定义

```python
class MultiPoseLoss(torch.nn.Module):
  def __init__(self, opt):
    super(MultiPoseLoss, self).__init__()
    self.crit = FocalLoss()
    self.crit_hm_hp = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
    self.crit_kp = RegWeightedL1Loss() if not opt.dense_hp else \
                   torch.nn.L1Loss(reduction='sum')
    self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
                    RegLoss() if opt.reg_loss == 'sl1' else None
```

这里定义了几种计算loss的criterion，实际使用的loss有FocalLoss(), RegWeightedL1Loss(), RegL1Loss()

```python
class RegL1Loss(nn.Module):
  def __init__(self):
    super(RegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class RegWeightedL1Loss(nn.Module):
  def __init__(self):
    super(RegWeightedL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _tranpose_and_gather_feat(output, ind)
    mask = mask.float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss
```

`L1Loss()` 比较好理解，
$$
L(x, y) = \{1_1, 1_2, ... , 1_N\}^T, 1_n = |x_n - y_n| \\

L(x, y) = mean(L(x, y)),\ if\ reduction = 'mean' \\ 
L(x, y) = sum(L(x, y)),\ \ if\ reduction = 'sum'\ \ \ \\
$$
`RegWeightedL1Loss()` 和`L1Loss()` 的区别在mask上，？？？？

```python
class FocalLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(FocalLoss, self).__init__()
    self.neg_loss = _neg_loss

  def forward(self, out, target):
    return self.neg_loss(out, target)

def _neg_loss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0

  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss
```

- hm_loss : `hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks`

  其中hm就是以物体（e.g.）、人体bbox中心点为中心draw_gaussian后得到的heatmap

- wh_loss : 

  wh_weight默认值？

  ```python
  if opt.wh_weight > 0:
    wh_loss += self.crit_reg(output['wh'], batch['reg_mask'], batch['ind'], batch['wh']) / opt.num_stacks
  ```

  其中`batch['reg_mask']`,`batch['ind']`都是size为`(batch_size, num_objs)`的Tensor。

  ind是原图中正样本的$index \in \{1, 2 ,..., h*w\}​ : ind[k] = ct_int[1] * output_res + ct_int[0]` 其实就是把h*w resize为1dim时的index。

  <u>regmask只在正样本处为1，其他地方为0。（但是是在num_objs维度？？？</u>

- off_loss

  ```python
  if opt.reg_offset and opt.off_weight > 0:
    off_loss += self.crit_reg(output['reg'], batch['reg_mask'], batch['ind'], batch['reg']) / opt.num_stacks
  ```

  reg_offset、wh_weight默认值？

- hp_loss : `hp_loss += self.crit_kp(output['hps'], batch['hps_mask'], batch['ind'], batch['hps']) / opt.num_stacks`

  hps_mask，(num_objs, num_joints*2) :  关键点的地方为1，其他地方为0 `kps_mask[k, j * 2: j * 2 + 2] = 1`

  hps，(num_objs, num_joints*2) :  各个关键点到中心点的距离  `kps[k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int`

- hm_hp_loss，：

  ```python
  if opt.hm_hp and opt.hm_hp_weight > 0:
    hm_hp_loss += self.crit_hm_hp(output['hm_hp'], batch['hm_hp']) / opt.num_stacks
  ```

  `draw_gaussian(hm_hp[j], pt_int, hp_radius)` hm_hp就是在每个keypoint周围draw_gaussian

- hp_offset_loss

  ```python
  if opt.reg_hp_offset and opt.off_weight > 0:
    hp_offset_loss += self.crit_reg(
      output['hp_offset'], batch['hp_mask'], batch['hp_ind'], batch['hp_offset']) / opt.num_stacks
  ```

  `hp_offset[k * num_joints + j] = pts[j, :2] - pt_int`

  hp_offset 和检测时的reg一样，是keypoint在下采样后在恢复原图时和原位置的offset

  hp_ind为每个keypoint所在位置index, (max_objs * num_joints ) : `hp_ind[k * num_joints + j] = pt_int[1] * output_res + pt_int[0]`

  

