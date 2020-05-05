import torch
from datasets import SEDataSet,SRDataSet
from torch.utils.data.dataloader import DataLoader
from model import *
import torch.nn.functional as F
import numpy as np
import math
from loss import *
from tensorboardX import SummaryWriter
from utils import *
# import cv2
# import matplotlib.pyplot as plt
import visdom
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
vis = visdom.Visdom(env="senet_8.0")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

writer = SummaryWriter("SRNet/10/2")
def train():
    dataset = SRDataSet("/home/yanjianfeng/data/teco_data/lr/HR/")
    val_dataset = SRDataSet("/home/yanjianfeng/data/Vid4/GT/")
    logdir= "/home/yanjianfeng/data/srnet_10.0"
    if os.path.exists(logdir) == False:
        os.makedirs(logdir)
    print(len(dataset))
    dataloader = DataLoader(dataset,batch_size=24,shuffle=True,drop_last=False,num_workers=4)
    val_dataloader = DataLoader(val_dataset,batch_size=4,shuffle=True,drop_last=False,num_workers=1)
    device=torch.device("cuda")
    model = SRNet().cuda()

    # writer.add_graph(model,torch.rand(1,3,3,64,64).cuda())
    # model = model.cuda()

    output_pad = torch.nn.ReplicationPad2d(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-4, betas=(0.9, 0.99))
    # loadckpt= "{}/snet_model_{:0>6}.ckpt".format("/home/yanjianfeng/data/srnet_8.0/", 6)
    # state_dict = torch.load(loadckpt)
    # model.load_state_dict(state_dict['model'])
    model = nn.DataParallel(model)
    # optimizer.load_state_dict(state_dict['optimizer'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.99)
    # for _  in range(20):
    #     lr_scheduler.step()
    for epoch in range(0,2000):
        lens = len(dataloader)
        model.train()
        for step,[lr,hr] in enumerate(dataloader):
            hr= hr.cuda()#b,v,c,h,w
            lr = lr.cuda()
            batch_size,channel,_,height,width = lr.shape
            # lr = lr.permute(0,2,1,3,4) # b, c ,v,h,w
            # print(imgs.shape)
            # img_down = F.avg_pool2d(img,kernel_size=3,stride=2,padding=1)
            optimizer.zero_grad()

            base,sup = model(lr)#b,1,h,w
            # print(sup.shape,imgs[:,:1].shape)
            # print(imgs.shape)
            hr = hr.permute(0,2,1,3,4)
            img = hr[:,:,hr.shape[2]//2]

            # print(sup.shape,img.shape)
            sups = hr+0.0 # b, c,v,h,w
            sups[:,:,hr.shape[2]//2] = sup
            loss_1 = loss_sr(sups,hr)
            loss_2 = loss_se(sup,img)
            loss_3 = CharbonnierLoss(sup,img)
            loss =     loss_3
            # loss = CharbonnierLoss(sup,img)
            loss.backward()

            optimizer.step()
            print("epoch ",epoch,"step %d/%d(%02f) : loss_sr %02f,loss_se %02f,loss_base %02f"%(step,lens,step/lens,loss_1.item(),loss_2.item(),loss.item()))
            if (step)%20 == 0:
                psnr_out = psnr(sup,img)
                # imgs_up   = F.interpolate(imgs_down,scale_factor=2.0,mode='bicubic',align_corners=True)#b,c,h/2,w/2
                # psnr_cubic = psnr(imgs_up,img_strip)
                # print(img.shape,sup.shape)
                # vis.images(torch.cat([img[:2], sup[:2]], 2).clamp(0,1.0) * 255.0, win="sr_rnn")
                # vis.line(X=np.column_stack([step+lens*epoch]),Y=np.column_stack([loss.item()]),
                # win="sd_rnn_loss",update='append',opts=dict(showlegend=True,legend=["sd_rnn_loss"]))
                # vis.line(X=np.column_stack([step+lens*epoch]),Y=np.column_stack([lr_scheduler.get_lr()]),
                # win="sd_rnn_lr",update='append',opts=dict(showlegend=True,legend=["sd_rnn_lr"]))
                # vis.line(X=np.column_stack([step+lens*epoch]),Y=np.column_stack([psnr_out.item()]),
                # win="sd_rnn_psnr",update='append',opts=dict(showlegend=True,legend=["sd_rnn_psnr"]))
                total_step = step+lens*epoch
                save_images(writer, 'train', {"hr":img,"sr":sup,"base":base,"detail":sup-base}, total_step)
                writer.add_scalar("train/loss_se",loss_1,total_step)
                writer.add_scalar("train/loss_sr",loss_2,total_step)
                writer.add_scalar("train/loss_ct",loss_3,total_step)
                writer.add_scalar("train/psnr",psnr_out,total_step)
                writer.add_scalar("train/base_psnr",psnr(base,img),total_step)
            # if (step)%1000 == 0:
            #     torch.save({
            #     'model': model.state_dict(),
            #     'optimizer': optimizer.state_dict()},
            #     logdir+"/snet_model_%02f_%.2f.ckpt"%(epoch+1,step/lens))
        psnrs = []
        model.eval()
        with torch.no_grad():
            val_lens = len(val_dataloader)
            for step,[lr,hr] in enumerate(val_dataloader):
                hr= hr.cuda()#b,v,c,h,w
                lr = lr.cuda()
                batch_size,channel,_,height,width = lr.shape
                # lr = lr.permute(0,2,1,3,4) # b, c ,v,h,w
                # print(imgs.shape)
                # img_down = F.avg_pool2d(img,kernel_size=3,stride=2,padding=1)
                base,sup = model(lr)#b,1,h,w
                hr = hr.permute(0,2,1,3,4)
                img = hr[:,:,hr.shape[2]//2]
                psnrs.append(psnr(sup,img).item())
                loss_3 = CharbonnierLoss(sup,img)
                print("val epoch ",epoch,"step %d/%d(%02f) : loss_base %02f"%(step,val_lens,step/val_lens,loss.item()))
                save_images(writer, 'val', {"hr":img,"sr":sup,"base":base,"detail":sup-base}, step+val_lens*epoch)

            writer.add_scalar("val/psnr",sum(psnrs)/len(psnrs),epoch)

        lr_scheduler.step()
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()},
            "{}/snet_model_{:0>6}_.ckpt".format(logdir, epoch+1))
    writer.close()
train()