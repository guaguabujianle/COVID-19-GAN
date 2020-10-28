# %%
import os
import math
import time
import numpy as np
import itertools
from PIL import Image
import matplotlib.pyplot as plt

import torch.optim as optim
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision import models

from networks import define_D, define_G, set_requires_grad, Unet, WarpResNet18
from dataloader import *
from config.config_dict import Config
from log.train_logger import TrainLogger
from optimizer import RangerV2
from utils import load_model_dict, cycle, AverageMeter, save_model_dict, label_smooth

# %%
config = Config()
args = config.get_config()
logger = TrainLogger(args)
logger.info(__file__)
result_dir = logger.get_result_dir()
model_dir = logger.get_result_dir()

os.environ["CUDA_VISIBLE_DEVICES"] = args.get("device")
save_model = args.get("save_model")
opt = args.get("optimizer")
model_name = args.get("model")
lr = args.get("lr")

# %%
# download in:
# https://www.kaggle.com/plameneduardo/sarscov2-ctscan-dataset
# Please cite:
# Angelov, Plamen, and Eduardo Almeida Soares. "EXPLAINABLE-BY-DESIGN APPROACH FOR COVID-19 CLASSIFICATION VIA CT-SCAN." medRxiv (2020).Soares, Eduardo, Angelov, Plamen, Biaso, Sarah, Higa Froes, Michele, and Kanda Abe, Daniel. "SARS-CoV-2 CT-scan dataset: A large dataset of real patients CT scans for SARS-CoV-2 identification." medRxiv (2020). doi: https://doi.org/10.1101/2020.04.24.20078584.
abnormal_dir = r'D:\data\COVID-19\COVID'
normal_dir = r'D:\data\COVID-19\non-COVID'

abnormal_path = glob(os.path.join(abnormal_dir, "*"))
normal_path = glob(os.path.join(normal_dir, "*"))
train_path = abnormal_path + normal_path
visual_path = train_path.copy()

print("train length:", len(train_path))
print("visual length:", len(visual_path))
print("normal length:", len(normal_path))

# %%
# config transform
train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomResizedCrop((256, 256), scale=(0.9, 1.0)), transforms.ToTensor()])
visual_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
train_set = SingleDataset(train_path, train_transform)
visual_set = SingleDataset(visual_path, visual_transform)
normal_set = SingleDataset(normal_path, transform=train_transform)
train_loader = DataLoader(train_set, num_workers=0, batch_size=8, shuffle=True)
visual_loader = DataLoader(visual_set, num_workers=0, batch_size=8, shuffle=True)
normal_loader = DataLoader(normal_set, num_workers=0, batch_size=8, shuffle=True)
iter_normal_loader = iter(cycle(normal_loader))

# %%
# config network, loss function and optimizer
input_nc = 1
output_nc = 1
ngf = 96
ndf = 96

netG = define_G(input_nc, output_nc, ngf, model_name, norm='instance').cuda()
netD = define_D(input_nc, ndf, 'patch', norm='instance').cuda()

netF_path = r'D:\model\covid-19\validating, epoch%3A40, loss%3A0.0503, acc%3A0.9819.pt'
netF = models.resnet18(pretrained=True)
num_features = netF.fc.in_features
netF.fc = nn.Linear(num_features, 2, bias=False)
netF.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
load_model_dict(netF, ckpt=netF_path)
netF = netF.cuda()
set_requires_grad([netF], False)
netF_feat = WarpResNet18(netF)

criterion_gan = nn.MSELoss()
criterion_cons = nn.L1Loss()
criterion_cls = nn.CrossEntropyLoss()

if opt == 'Adam':
    optimizer_G = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(netD.parameters(), lr=lr * 2, betas=(0.5, 0.999))
elif opt == 'Ranger':
    optimizer_G = RangerV2(netG.parameters(), lr=2e-4)
    optimizer_D = RangerV2(netD.parameters(), lr=4e-4)
else:
    raise Exception(f'{opt} is not supported currently!')

# visualization
def visualization(test_loader, netG, epoch, save_dir, max_example=4):
    real_list = []
    fake_list = []
    diff_list = []

    # load data
    data = next(iter(test_loader))
    image = data
    image = image.cuda()
 
    with torch.no_grad():
        fake_normal = netG(image)

    if image.size(0) > max_example:
        image = image[:max_example]
        fake_normal = fake_normal[:max_example]

    diff_image = F.relu(image - fake_normal)

    real_list.append(image.cpu())
    fake_list.append(fake_normal.cpu())
    diff_list.append(diff_image.cpu())

    real_cat = torch.cat(real_list, dim=0)
    fake_cat = torch.cat(fake_list, dim=0)
    diff_cat = torch.cat(diff_list, dim=0)

    real_normal = vutils.make_grid(real_cat, normalize=True).numpy().transpose((1, 2, 0))
    fake_normal = vutils.make_grid(fake_cat, normalize=True).numpy().transpose((1, 2, 0))
    diff = vutils.make_grid(diff_cat, normalize=True).numpy().transpose((1, 2, 0))

    result = np.concatenate((real_normal, fake_normal, diff), axis=0)

    path = os.path.join(save_dir, f"{epoch}.png")
    plt.imsave(path, result)

def accuracy(pred, label):
    pred = torch.argmax(pred.detach(), dim=1)
    acc = torch.sum(pred == label).float() / len(label)

    return acc

# %%
# training parameters setting
epochs = 400
steps_per_epoch = 100
num_iter = math.ceil((epochs * steps_per_epoch) / len(train_loader))

global_step = 0
global_epoch = 0

running_loss_GD = AverageMeter()
running_loss_D = AverageMeter()
running_loss_cons = AverageMeter()
running_loss_feat = AverageMeter()

lambda_cons = 3.0
# lambda_cons = 4.0
real = torch.tensor([1.])
fake = torch.tensor([0.])

netG.train()
netD.train()
netF.eval()

# start training
for i in range(num_iter):
    for data in train_loader:
        global_step += 1

        # load data
        image = data
        normal_image = next(iter_normal_loader)
        image = image.cuda()
        normal_image = normal_image.cuda()

        #------------------#
        # generator -> discriminator
        #------------------#

        ####################
        # train generator
        ####################
        # discriminator require no gradients when optimizing generator
        set_requires_grad([netD], requires_grad=False)
        fake_normal = netG(image)
        pred_fake = netD(fake_normal)
        label = real.expand_as(pred_fake).cuda()
        loss_GD = criterion_gan(pred_fake, label)
        loss_cons = criterion_cons(fake_normal, image) * lambda_cons
        loss_L1 = loss_GD + loss_cons
        optimizer_G.zero_grad()   
        loss_L1.backward()
        optimizer_G.step()

        running_loss_GD.update(loss_GD.item(), image.size(0))

        ####################
        # train discriminator
        ####################

        # set discriminator required gradients
        set_requires_grad([netD], requires_grad=True)

        # train on real data
        pred_real = netD(normal_image)
        label = real.expand_as(pred_real).cuda()
        # only smooth positive label, also called one sized smooth
        label = label_smooth(label)
        loss_D_real = criterion_gan(pred_real, label)

        pred_fake = netD(fake_normal.detach())
        label = fake.expand_as(pred_fake).cuda()
        # label = label_smooth(label)
        loss_D_fake = criterion_gan(pred_fake, label)

        loss_D = (loss_D_real + loss_D_fake) * 0.5
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        running_loss_D.update(loss_D.item(), image.size(0))
        running_loss_cons.update(loss_cons.item(), image.size(0))

        #------------------#
        # generator -> features
        #------------------#
        fake_normal = netG(image)
        netF_feat.clear_feature_buffer()
        f = netF(fake_normal)
        fake_features = netF_feat.get_features()
        
        netF_feat.clear_feature_buffer()
        f = netF(image)
        real_features = netF_feat.get_features()

        feat0= criterion_cons(fake_features[0], real_features[0]) * 3.
        feat1 = criterion_cons(fake_features[1], real_features[1]) * 2.5
        feat2 = criterion_cons(fake_features[2], real_features[2]) * 2.
        feat3 = criterion_cons(fake_features[3], real_features[3]) * 1.5
        feat4 = criterion_cons(fake_features[4], real_features[4])  * 1.

        loss_feat = (feat0 + feat1 + feat2 + feat3 + feat4)

        optimizer_G.zero_grad()
        loss_feat.backward()
        optimizer_G.step()

        running_loss_feat.update(loss_feat, image.size(0))

        if global_step % steps_per_epoch == 0:
            epoch_loss_GD = running_loss_GD.get_average()
            epoch_loss_D = running_loss_D.get_average()
            epoch_loss_cons = running_loss_cons.get_average()
            epoch_loss_feat = running_loss_feat.get_average()

            running_loss_GD.reset()
            running_loss_D.reset()
            running_loss_cons.reset()
            running_loss_feat.reset()

            msg = "epoch- %d, loss_GD- %.4f, loss_cons- %.4f, loss_feat- %.4f, loss_D- %.4f" % (global_epoch, epoch_loss_GD, epoch_loss_cons, epoch_loss_feat, epoch_loss_D)

            logger.info(msg)

            visualization(visual_loader, netG, global_epoch, result_dir, max_example=8)

            global_epoch += 1

            if save_model:
                save_model_dict(netG, logger.get_model_dir(), msg + '_G')
                save_model_dict(netD, logger.get_model_dir(), msg + '_D')
                # save_model_dict(netG, logger.get_model_dir(), msg + '_G')
                # save_model_dict(netD, logger.get_model_dir(), msg + '_D')
   

# %%




