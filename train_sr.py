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
# vis = visdom.Visdom(env="senet_8.0")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

writer = SummaryWriter("SRNet/23/1/")
def train():
    dataset = SRDataSet("//home/yanjianfeng/data/vimeo90K/vimeo_septuplet/lr/HR/","//home/yanjianfeng/data/vimeo90K/vimeo_septuplet/lr/mLRs4/")
    # val_dataset = SRDataSet("/home/yanjianfeng/data/teco_data/val/gt/","/home/yanjianfeng/data/teco_data/val/mLRs4/")
    val_dataset = SRDataSet("/home/yanjianfeng/data/Vid4/GT/","/home/yanjianfeng/data/Vid4/lr/LRs4/",'val')
    # dataset = SRDataSet("/home/yanjianfeng/data/Vid4/GT/","/home/yanjianfeng/data/Vid4/lr/LRs4/")
    logdir= "/home/yanjianfeng/data/srnet_23.0"
    if os.path.exists(logdir) == False:
        os.makedirs(logdir)
    print(len(dataset))
    dataloader = DataLoader(dataset,batch_size=32,shuffle=True,drop_last=True,num_workers=2)
    val_dataloader = DataLoader(val_dataset,batch_size=1,shuffle=True,drop_last=True,num_workers=1)
    device=torch.device("cuda")
    model = SRNet().cuda()

    # writer.add_graph(model,torch.rand(1,3,3,64,64).cuda())
    model = model.cuda()

    # output_pad = torch.nn.ReplicationPad2d(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.99))
    loadckpt= "{}/snet_model_{:06d}_step.ckpt".format("/home/yanjianfeng/data/srnet_21.0/", 95000)
    # loadckpt= "{}/snet_model_{:06d}_.ckpt".format("/home/yanjianfeng/data/srnet_15.0/", 9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.95)
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'],False)
    optimizer.load_state_dict(state_dict['optimizer'])
    lr_scheduler.load_state_dict(state_dict['scheduler'])

    model = nn.DataParallel(model)
    # for _  in range(20):
    #     lr_scheduler.step()
    total_step = 95000
    for epoch in range(0,2000):
        lens = len(dataloader)
        model.train()
        for step,[lr,gt] in enumerate(dataloader):
            gt= gt.cuda()#b,c,h,w
            lr = lr.cuda()
            total_step+=1
            batch_size,channel,_,height,width = lr.shape
            # lr = lr.permute(0,2,1,3,4) # b, c ,v,h,w
            # print(gts.shape)
            # gt_down = F.avg_pool2d(gt,kernel_size=3,stride=2,padding=1)
            optimizer.zero_grad()

            base,sup = model(lr)#b,1,h,w
            # print(sup.shape,gts[:,:1].shape)
            # print(gts.shape)
            # gt = gt.permute(0,2,1,3,4) # b, c,v,h,w
            # gt = gt[:,:,gt.shape[2]//2]

            # print(sup.shape,gt.shape)
            # sups = gt+0.0 # b, c,v,h,w
            # sups[:,:,gt.shape[2]//2] = sup
            # loss_1 = loss_sr(sups,gt)
            loss_1 = CharbonnierLoss(base,gt)
            loss_2 = loss_se(sup,gt)
            loss_3 = CharbonnierLoss(sup,gt)

            loss =    loss_3 + loss_2
            # loss = CharbonnierLoss(sup,gt)
            loss.backward()
            optimizer.step()
            # total_step = step+lens*epoch
            print("epoch ",epoch,"step %d/%d(%02f) : loss_se %02f,loss_base %02f"%(step,lens,step/lens,loss_2.item(),loss_3.item()))
            if (total_step)%20 == 0:
                psnr_out = psnr(sup,gt)
                # gts_up   = F.interpolate(gts_down,scale_factor=2.0,mode='bicubic',align_corners=True)#b,c,h/2,w/2
                # psnr_cubic = psnr(gts_up,gt_strip)
                # print(gt.shape,sup.shape)
                # vis.images(torch.cat([gt[:2], sup[:2]], 2).clamp(0,1.0) * 255.0, win="sr_rnn")
                # vis.line(X=np.column_stack([step+lens*epoch]),Y=np.column_stack([loss.item()]),
                # win="sd_rnn_loss",update='append',opts=dict(showlegend=True,legend=["sd_rnn_loss"]))
                # vis.line(X=np.column_stack([step+lens*epoch]),Y=np.column_stack([lr_scheduler.get_lr()]),
                # win="sd_rnn_lr",update='append',opts=dict(showlegend=True,legend=["sd_rnn_lr"]))
                # vis.line(X=np.column_stack([step+lens*epoch]),Y=np.column_stack([psnr_out.item()]),
                # win="sd_rnn_psnr",update='append',opts=dict(showlegend=True,legend=["sd_rnn_psnr"]))

                save_images(writer, 'train', {"gt":gt,"sr":sup,"cubic":base,"sr-cubic":sup-base,"sr-gt":sup-gt}, total_step)
                writer.add_scalar("train/loss_base",loss_1,total_step)
                writer.add_scalar("train/loss_se",loss_2,total_step)
                writer.add_scalar("train/loss_ct",loss_3,total_step)
                writer.add_scalar("train/psnr",psnr_out,total_step)
                writer.add_scalar("train/psnr_base",psnr(base,gt),total_step)
                writer.add_scalar("train/lr",lr_scheduler.get_lr(),total_step)
            # if (step)%1000 == 0:
            #     torch.save({
            #     'model': model.state_dict(),
            #     'optimizer': optimizer.state_dict()},
            #     logdir+"/snet_model_%02f_%.2f.ckpt"%(epoch+1,step/lens))
            if (total_step%5000) == 0:
                psnrs = []
                # psnrs_1 = []
                model.eval()
                with torch.no_grad():
                    val_lens = len(val_dataloader)
                    for val_step,[lr,gt] in enumerate(val_dataloader):
                        optimizer.zero_grad()
                        gt= gt.cuda()#b,v,c,h,w
                        lr = lr.cuda()
                        batch_size,channel,_,height,width = lr.shape
                        # lr = lr.permute(0,2,1,3,4) # b, c ,v,h,w
                        # print(gts.shape)
                        # gt_down = F.avg_pool2d(gt,kernel_size=3,stride=2,padding=1)
                        base,sup = model(lr)#b,1,h,w
                        psnrs.append(psnr(sup,gt).item())
                        # psnrs_1.append(psnr(sup,gt).item())
                        loss = CharbonnierLoss(sup,gt)
                        print("val epoch ",epoch,"step %d/%d(%02f) : loss_base %02f"%(val_step,val_lens,val_step/val_lens,loss.item()))
                        #
                        if (val_step)%10 == 0:
                            writer.add_scalar("val/loss_ct",loss,int(val_step+val_lens*(total_step/5000)))
                            save_images(writer, 'val', {"gt":gt,"sr":sup,"cubic":base,"sr-cubic":sup-base,"sr-gt":sup-gt}, int(val_step+val_lens*(total_step/5000)))
                    torch.save({
                        'model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': lr_scheduler.state_dict()},

                        logdir+"/snet_model_%06d_step.ckpt"%(total_step))

                    writer.add_scalar("val/psnr",sum(psnrs)/len(psnrs),total_step)
                    # writer.add_scalar("val/psnr",sum(psnrs_1)/len(psnrs_1),total_step)

        lr_scheduler.step()
        # torch.save({
        #     'model': model.module.state_dict(),
        #     'optimizer': optimizer.state_dict()},
        #     "{}/snet_model_{:0>6}_.ckpt".format(logdir, epoch+1))
    writer.close()

def test(total_step=0):
    psnrs = []
    # psnrs_1 = []
    model.eval()
    with torch.no_grad():
        val_lens = len(val_dataloader)
        for val_step,[lr,gt] in enumerate(val_dataloader):
            optimizer.zero_grad()
            gt= gt.cuda()#b,v,c,h,w
            lr = lr.cuda()
            batch_size,channel,_,height,width = lr.shape
            # lr = lr.permute(0,2,1,3,4) # b, c ,v,h,w
            # print(gts.shape)
            # gt_down = F.avg_pool2d(gt,kernel_size=3,stride=2,padding=1)
            base,sup = model(lr)#b,1,h,w
            psnrs.append(psnr(sup,gt).item())
            # psnrs_1.append(psnr(sup,gt).item())
            loss = CharbonnierLoss(sup,gt)
            print("step %d/%d(%02f) : loss_base %02f"%(val_step,val_lens,val_step/val_lens,loss.item()))
            #
            if (val_step)%10 == 0:
                writer.add_scalar("val/loss_ct",loss,int(val_step+val_lens*(total_step/5000)))
                save_images(writer, 'val', {"gt":gt,"sr":sup,"cubic":base,"sr-cubic":sup-base,"sr-gt":sup-gt}, int(val_step+val_lens*(total_step/5000)))
        writer.add_scalar("val/psnr",sum(psnrs)/len(psnrs),total_step)
        writer.add_scalar("val/psnr",sum(psnrs)/len(psnrs),total_step)
        # print(sum(psnrs)/len(psnrs))
# test(150000)
train()