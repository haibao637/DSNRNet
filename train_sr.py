import torch
from datasets import SEDataSet,SRDataSet
from torch.utils.data.dataloader import DataLoader
from model import *
import torch.nn.functional as F
import numpy as np
import math
from loss import *
# import cv2
# import matplotlib.pyplot as plt
import visdom
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
vis = visdom.Visdom(env="senet")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def train():
    dataset = SRDataSet("/home/yanjianfeng/data/teco_data/")
    logdir= "/home/yanjianfeng/data/srnet"
    if os.path.exists(logdir) == False:
        os.makedirs(logdir)
    print(len(dataset))
    dataloader = DataLoader(dataset,batch_size=4,shuffle=True,drop_last=False,num_workers=4)
    device=torch.device("cuda")
    model = SRNet(1).cuda()
    output_pad = torch.nn.ReplicationPad2d(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, betas=(0.9, 0.95))
    loadckpt= "{}/srnet_sample_model_{:0>6}.ckpt".format("/home/haibao637/data/", 4900)
    # state_dict = torch.load(loadckpt)
    # model.load_state_dict(state_dict['model'], strict=False)
    # optimizer.load_state_dict(state_dict['optimizer'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.9)
    for epoch in range(0,20):
        lens = len(dataloader)
        for step,[lr,hr] in enumerate(dataloader):
            hr= hr.cuda()#b,v,c,h,w
            lr = lr.cuda()
            batch_size,channel,_,height,width = lr.shape
            # lr = lr.permute(0,2,1,3,4) # b, c ,v,h,w
            # print(imgs.shape)
            # img_down = F.avg_pool2d(img,kernel_size=3,stride=2,padding=1)
            optimizer.zero_grad()

            sup = model(lr)#b,1,h,w
            # print(sup.shape,imgs[:,:1].shape)
            # print(imgs.shape)
            img = hr[:,:,1]
            # print(sup.shape,img.shape)
            sups = hr+0.0 # b, c,v,h,w
            sups[:,:,1] = sup
            loss_1 = loss_sr(sups,hr)
            loss_2 = loss_se(sup,img)
            loss_3 = F.l1_loss(sup,img)
            loss =   loss_1 +  loss_2 + 0.05*loss_3
            loss.backward()
            optimizer.step()
            print("epoch ",epoch,"step %d/%d(%02f) : loss_sr %02f,loss_se %02f,loss_base %02f"%(step,lens,step/lens,loss_1.item(),loss_2.item(),loss_3.item()))
            if (step)%20 == 0:
                psnr_out = psnr(sup,img)
                # imgs_up   = F.interpolate(imgs_down,scale_factor=2.0,mode='bicubic',align_corners=True)#b,c,h/2,w/2
                # psnr_cubic = psnr(imgs_up,img_strip)
                # print(img.shape,sup.shape)
                vis.images(torch.cat([img[:2], sup[:2]], 2) * 255.0, win="sr_rnn")
                vis.line(X=np.column_stack([step+lens*epoch]),Y=np.column_stack([loss.item()]),
                win="sd_rnn_loss",update='append',opts=dict(showlegend=True,legend=["sd_rnn_loss"]))
                vis.line(X=np.column_stack([step+lens*epoch]),Y=np.column_stack([lr_scheduler.get_lr()]),
                win="sd_rnn_lr",update='append',opts=dict(showlegend=True,legend=["sd_rnn_lr"]))
                vis.line(X=np.column_stack([step+lens*epoch]),Y=np.column_stack([psnr_out.item()]),
                win="sd_rnn_psnr",update='append',opts=dict(showlegend=True,legend=["sd_rnn_psnr"]))
            if (step)%1000 == 0:
                torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                logdir+"/snet_model_%02f_%.2f.ckpt"%(epoch+1,step/lens))
        lr_scheduler.step()
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()},
            "{}/snet_model_{:0>6}.ckpt".format(logdir, epoch+1))
train()