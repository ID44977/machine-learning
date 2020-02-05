from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.core.pylabtools import figsize
from IPython.display import HTML

# 设置随机种子
manualSeed = 999
#manualSeed = random.randint(1, 10000) # 想获取新结果时使用
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

dataroot = "c:/ml/celeba" #数据集的根目录
workers = 0 #载数据的线程数量
batch_size = 128 #训练过程batch大小
image_size = 64 #训练图片大小，所有图片均需要缩放到这个尺寸
nc = 3 #通道数量，通常彩色图就是rgb三个值
nz = 100 #产生网络输入向量的大小
ngf = 64 #产生网络特征层的大小
ndf = 64 #判别网络的特征层的大小
num_epochs = 5 #训练数据集迭代次数
lr = 0.0002 #学习率
beta1 = 0.5 #Adam最优化方法中的超参 beta1
ngpu = 1 #可用的gpu数量（0为cpu模式）


# 创建数据集（包含各种初始化）
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# 创建数据载入器 DataLoader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# 设置训练需要的处理器
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
'''
####展示一些训练数据####
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64],padding=2, normalize=True).cpu(),(1,2,0)))
'''
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# 产生网络代码
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 输入向量z，通过第一个反卷积
            # 将100的向量z输入，输出channel设置为(ngf*8)，经过如下操作
            # class torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
            # 后得到(ngf*8) x 4 x 4，即长宽为4，channel为ngf*8的特征层
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),  # 这里的ConvTranspose2d类似于deconv，前面第 章介绍过原理
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # 继续对特征层进行反卷积，得到长宽为8，channel为ngf*4的特征层  (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # 继续对特征层进行反卷积，得到长宽为16，channel为ngf*2的特征层  (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # 继续对特征层进行反卷积，得到长宽为32，channel为ngf的特征层  (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # 继续对特征层进行反卷积，得到长宽为64，channel为nc的特征层  (nc) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


################################################################
#### 将产生网络实例化 ####
# 创建生成器
netG = Generator(ngpu).to(device)

# 处理多gpu情况
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

# 应用weights_init函数对随机初始化进行重置，改为服从mean=0, stdev=0.2的正态分布的初始化
netG.apply(weights_init)

print(netG)


# 判别网络代码
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # 输入为一张宽高为64，channel为nc的一张图片，得到宽高为32，channel为ndf的一张图片  (ndf) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 经过第2次卷积 得到宽高为16，channel为ndf*2的一张图片 (ndf*2) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),  # 使用大尺度的步长来代替下采样（pooling），这样可以更好地学习降采样的方法
            nn.LeakyReLU(0.2, inplace=True),
            # 经过第3次卷积 得到宽高为8，channel为ndf*4的一张图片  (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 经过第4次卷积 得到宽高为4，channel为ndf*8的一张图片  (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 经过第5次卷积并过sigmoid层，得最终一个概率输出值
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # 最终通过Sigmoid激活函数输出该张图片是真实图片的概率
        )

    def forward(self, input):
        return self.main(input)


#### 将判别网络实例化 ####
# 创建判别器
netD = Discriminator(ngpu).to(device)

# 处理多gpu情况
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

# 应用weights_init函数对随机初始化进行重置，改为服从mean=0, stdev=0.2的正态分布的初始化
netD.apply(weights_init)

print(netD)

#初始化二元交叉熵损失函数
criterion = nn.BCELoss()

#创建一个batch大小的向量z，及产生网络的输入数据
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

#定义训练过程的真图片/假图片的标签
real_label = 1
fake_label = 0

#为产生网络和判别网络设置Adam优化器
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

#训练过程：主循环
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
for epoch in range(num_epochs):  # 训练集迭代的次数
    for i, data in enumerate(dataloader, 0):  # 循环每个dataloader中的batch

        ############################
        # (1) 更新判别网络：最大化 log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## 用全部都是真图片的batch训练
        netD.zero_grad()
        # 格式化batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
        # 将带有正样本的batch，输入到判别网络 中进行前向计算，得到结果放到变量output中
        output = netD(real_cpu).view(-1)
        # 计算loss
        errD_real = criterion(output, label)
        # 计算梯度
        errD_real.backward()
        D_x = output.mean().item()

        ## 用全部都是假图片的batch训练
        # 先产生网络的输入向量
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # 通过产生网络生成假的样本图片
        fake = netG(noise)
        label.fill_(fake_label)
        # 将生成的全部假图片输入到判别网络中进行前向计算，得到结果放到变量output中
        output = netD(fake.detach()).view(-1)
        # 在假图片batch中计算刚刚判别网络的loss
        errD_fake = criterion(output, label)
        # 计算该batch的梯度
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # 将真图片与假图片的误差加和
        errD = errD_real + errD_fake
        # 更新判别网络D
        optimizerD.step()

        ############################
        # (2) 更新产生网络： 最大化 log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # 产生网络的标签是真实的图片
        # 由于刚刚更新了判别网络，这里让假数据再过一遍判别网络，用来计算产生网络的loss并回传
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        # 更新产生网络G
        optimizerG.step()

        # 打印训练状态
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # 保存loss，用于后续画图
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # 保留产生网络生成的图片，后续用来看生成的图片效果
        if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

####查看网络Loss的变化####
plt.figure(figsize(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

####对比真实图片和产生的假图片####
real_batch = next(iter(dataloader))

#画真实的图片
plt.figure(figsize(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

#画出产生网络最后一个迭代产生的图片
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()

