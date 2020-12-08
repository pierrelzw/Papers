# 如何快速理解GAN

[TOC]

## 前言

从我第一次听说GAN，到我敲下这篇文章，开始入门，至少过去了2年。原因很多，但是经过最近反思，一个主要原因就是“恐惧”——对未知的恐惧。因为一开始就对GAN一知半解，加上我在工作中几乎不需要GAN，所以也就没有下定决心搞明白它。这间接导致我听到、看到GAN相关的论文、文章时，我不仅没有去解决我的“一知半解”，反而因为潜意识的恐惧，经常很快略过这些内容。结果，我对GAN的基本理解延后了2年。所以如果你对GAN感兴趣，我这里提供一个简单的入门教程。这是个代码教程，所以建议马上跟着敲代码做实验，不是明天不是收藏之后以后看，就是现在。

## 如何

这篇文章是关于how的，所以直接上答案：

1. 找一个简单的pytorch GAN教程（本文就是）
2. 简单浏览代码，优先回答以下问题：
   - 看看代码的最终目的是什么？
   - 每个大的代码模块在干什么？（这个阶段先不用深究看不懂的代码） 

3. copy代码，跑一下看看效果。虽然细节代码不需要看，但是这一步其实要求你能看懂80%的代码。能回答以下问题：

   - 代码输入是什么？
   - 使用了什么数据？
   - 预期结果是什么？
   - 如何显示结果？

   这些都是要明白的。所以建议优先寻找pytorch官网的tutorial，其次是知乎、google上的blog文章。

4. 对着copy了能跑的代码，新建一个文件（or工程，如果代码很复杂的话），一步一步跟着敲下来。如果代码简单的话，大概1h左右，你就能从代码层面理解GAN的基本原理了。你会发现，最开始的GAN，从代码上看，非常简单易懂。

下面是我使用的代码（参考了知乎文章），代码经过测试，请放心使用。

```python

import torch
from torchvision import transforms, datasets
from torch import nn, optim
from torch.utils.data import DataLoader, sampler
from torch.autograd import Variable
import matplotlib.gridspec as gridspec

import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
import os
now_timestr = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_dir = './results_{}'.format(now_timestr)

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# hyper params
batch_size = 100
NOISE_DIM = 100
NUM_TRAIN = 50000
NUM_VAL = 5000

def deprocess_img(x):
    return (x+1.0)/2.0

class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset
    Args:
      num_samples : #of desired datapoints
      start : offset where we should start slecting from
    """

    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))  # TODO

    def __len__(self):
        return self.num_samples

def show_images(images, show=False, save_path=None):
    images = np.reshape(images, [images.shape[0], -1])
    print(images.shape)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg, sqrtimg]))

    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path)

data_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
train_set = datasets.MNIST('./mnist', train=True, download=True, transform=data_tf)
train_data = DataLoader(train_set, batch_size=batch_size, sampler=ChunkSampler(NUM_TRAIN, 0)) # TODO ?no shuffle here

val_set = datasets.MNIST('./mnist', train=True, download=True, transform=data_tf)
val_data = DataLoader(val_set, batch_size=batch_size, sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))

imgs =  deprocess_img(train_data.__iter__().next()[0].view(batch_size, 784)).numpy().squeeze()
show_images(imgs)

# Discriminator
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = nn.Sequential(                     # b, 1, 28, 28
            nn.Conv2d(1, 32, kernel_size=5, padding=2), # b, 32, 28, 28
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, 2)                          # b, 32, 14, 14
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2), # b, 64, 14, 14
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, 2)                           # b, 64, 7, 7
        )

        self.fc = nn.Sequential(
            nn.Linear(64*7*7, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid() # output val \in (0, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class generator(nn.Module):
    def __init__(self, noise_dim=NOISE_DIM):
        super(generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(noise_dim, 1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 7*7*128),
            nn.ReLU(True),
            nn.BatchNorm1d(7*7*128)
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # b, 64, 14, 14
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, 4, 2, padding=1), # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 128, 7, 7)
        x = self.conv(x)
        return x

bce_loss = nn.BCEWithLogitsLoss()

def discriminator_loss(logits_real, logits_fake):
    size = logits_real.size(0)
    true_labels = Variable(torch.ones(size, 1).float().cuda())
    fake_labels = Variable(torch.zeros(size, 1).float().cuda())
    loss = bce_loss(logits_real, true_labels) + bce_loss(logits_fake, fake_labels)
    return loss

def generator_loss(logits_fake):
    size = logits_fake.size(0)
    fake_labels = Variable(torch.ones(size, 1).float().cuda())
    loss = bce_loss(logits_fake, fake_labels)
    return loss

def get_optimizer(net):
    optimizer = optim.Adam(net.parameters(), lr=3e-4, betas=(0.5, 0.999))
    return optimizer

def train_gan(D_net, G_net, D_optimizer, G_optimizer,
              show_every=250, noise_size=100, num_epochs=10):
    iter_count = 0
    for epoch in range(num_epochs):
        for x,_ in train_data:
            bs = x.size(0)

            # discriminator
            real_data = Variable(x).cuda()
            logits_real = D_net(real_data)

            # sample noise from torch.randn()
            sample_noise = (torch.randn(bs, noise_size) - 0.5) / 0.5 #[-1, 1]
            g_fake_seed = Variable(sample_noise).cuda()

            # generate fake image
            fake_images = G_net(g_fake_seed)

            # send it to discriminator
            logits_fake = D_net(fake_images)

            # compute discriminator_loss
            discriminator_error = discriminator_loss(logits_real, logits_fake)

            # optim discriminator
            D_optimizer.zero_grad()
            discriminator_error.backward()
            D_optimizer.step()


            # generate fake data
            sample_noise = (torch.randn(bs, noise_size) - 0.5) / 0.5
            g_fake_seed = Variable(sample_noise).cuda()

            fake_images = G_net(g_fake_seed)
            logits_fake = D_net(fake_images)

            # compute generator loss
            generator_error = generator_loss(logits_fake)

            # optimize generator
            G_optimizer.zero_grad()
            generator_error.backward()
            G_optimizer.step()

            if (iter_count % show_every == 0):
                print(f"epoch: {epoch}, Iter: {iter_count}, D:{discriminator_error:.4}, G:{generator_error:.4}")
                imgs_numpy = deprocess_img(fake_images.data.cpu().numpy())
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                save_path = '{}/{}_{}_D{:.4}_G{:.4}.jpg'.format(save_dir, epoch, iter_count,
                                                                discriminator_error.data,
                                                                generator_error.data)
                print(save_path)

                show_images(imgs_numpy, show=False, save_path=save_path)
                print("iter_count: {}".format(iter_count))
            iter_count += 1

# start training
D_net = discriminator().cuda()
G_net = generator().cuda()

D_optimizer = get_optimizer(D_net)
G_optimizer = get_optimizer(G_net)

train_gan(D_net, G_net, D_optimizer, G_optimizer, num_epochs=50)

```

## 效果图

（待补充）

## 后话

当然，看完这篇文章，你也按照上面的代码敲完，你也只是理解了最简单版本的GAN。GAN在这几年有了巨大进展，比如要求配对的条件GAN，不要求配对的cycleGAN等等。如果你有学习生成对抗网络的需求，还是要老老实实去看这些GAN变体的论文和代码吧

![image-20201120194259177](https://tva1.sinaimg.cn/large/0081Kckwly1gky9kq39joj31sw0s2n4w.jpg)





## 参考

[PyTorch 学习笔记（十）：初识生成对抗网络（GANs）](https://zhuanlan.zhihu.com/p/68098661)

[DCGAN TUTORIAL](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)