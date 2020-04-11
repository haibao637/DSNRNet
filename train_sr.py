import torch
from datasets import SEDataSet
from torch.utils.data.dataloader import DataLoader
from model import SENet,SRNet
import torch.nn.functional as F
import numpy as np
import math
# import cv2
# import matplotlib.pyplot as plt
import visdom
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
vis = visdom.Visdom(env="senet")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def train():
    dataset = SEDataSet("/home/yanjianfeng/data/hive_top/download/images/")
    logdir= "/home/yanjianfeng/data/srnet_model_grad_Scharr_x_y_gen_1.5_noise_level_0.01_3_layer_16_upsample"
    if os.path.exists(logdir) == False:
        os.makedirs(logdir)
    print(len(dataset))
    dataloader = DataLoader(dataset,batch_size=64,shuffle=True,drop_last=False,num_workers=4)
    device=torch.device("cuda")
    model = SRNet(1).cuda()
    loss0 = torch.nn.L1Loss()
    # sobelx=torch.Tensor([-1,-2,-1,0,0,0,1,2,1])
    # sobelx.requires_grad = False
    # sobely=torch.Tensor([-1,0,1,-2,0,2,-1,0,1])
    # sobely.requires_grad = False
    # sobelx = sobelx.type(torch.float32).reshape(1,1,3,3).cuda()
    # sobely = sobely.type(torch.float32).reshape(1,1,3, 3).cuda()


    scharrx=torch.Tensor([-3,0,3,-10,0,10,-3,0,3])
    scharrx.requires_grad = False
    scharry=torch.Tensor([-3,-10,-3,0,0,0,3,10,3])
    scharry.requires_grad = False
    scharrx = scharrx.type(torch.float32).reshape(1,1,3,3).cuda()
    scharry = scharry.type(torch.float32).reshape(1,1,3, 3).cuda()


    laplacian=torch.ones([1,1,3,3],dtype=torch.float32).cuda()
    gaussian = torch.Tensor([1,2,1,2,4,2,1,2,1]).type(torch.float32).reshape(1,1,3,3)/16.0
    gaussian = gaussian.cuda()
    laplacian[:,:,1,1]=8
    scale=2.0
    output_pad = torch.nn.ReplicationPad2d(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    loadckpt= "{}/srnet_sample_model_{:0>6}.ckpt".format("/home/haibao637/data/", 4900)
    # state_dict = torch.load(loadckpt)
    # model.load_state_dict(state_dict['model'], strict=False)
    # optimizer.load_state_dict(state_dict['optimizer'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.9)
    for epoch in range(0,20):
        lens = len(dataloader)
        for step,img in enumerate(dataloader):
            img = img.cuda()
            # img_down = F.avg_pool2d(img,kernel_size=3,stride=2,padding=1)
            optimizer.zero_grad()
            img_down = F.interpolate(img,scale_factor=0.5,mode="bilinear")
            detail = model(img_down)

            img_up = F.interpolate(img_down,scale_factor=2.0,mode="bilinear")
            img_up = F.conv2d(output_pad(img_up),gaussian,stride=1,padding=0)
            # coarse = torch.clamp(coarse,-1.0,1.0)
            # img_up = F.interpolate(img,scale_factor=2.0,mode="bilinear")

            # detail = img-coarse
            # detail = img*(1+detail)
            detail_loss = torch.mean((torch.abs(detail*img)).type(torch.float32))


            # img_patch = torch.nn.functional.unfold(img, kernel_size=(3, 3), padding=1)
            # img_m = torch.mean(img_patch, dim=1, keepdim=True)
            # gt_var = scale * torch.mean((img_patch - img_m) ** 2, dim=1, keepdim=True).reshape(detail.shape)
            # # mask = (gt_var>1e-2).type(torch.float32)
            # gt_var = torch.clamp(gt_var*250,0.1,1.0)
            # gt_var = (gt_var - gt_var.min()) / (gt_var.max() - gt_var.min() + 1e-6)
            # detail = detail
            output_down = img_up*(1+detail)
            # output_down = img+detail

            output_down_blur = F.avg_pool2d(output_pad(output_down),kernel_size=3,padding=0,stride=1)
            # output_down_blur = F.conv2d(output_pad(output_down),gaussian,stride=1,padding=0)
            # valid_count = torch.sum(mask)+1e-6
            # print(valid_count/mask.size())
            # output_down_loss0 = torch.sum(((output_down-img)**2)*mask)/valid_count
            output_down_loss0 = loss0(output_down_blur,img)

            output_down_grad_x = (F.conv2d(output_down, scharrx, stride=1, padding=1))
            output_down_grad_y = (F.conv2d(output_down, scharry, stride=1, padding=1))
            gt_down_grad_x = (F.conv2d(img, scharrx, padding=1))
            gt_down_grad_y = (F.conv2d(img, scharry, padding=1))

            gt_base = img - gt_down_grad_x-gt_down_grad_y

            # print("grad",img.min(),img.max(),gt_down_grad_x.min(),gt_down_grad_x.max())
            # grad range (-4,4)
            gt_down_grad_x = gt_down_grad_x/32.0
            gt_down_grad_y = gt_down_grad_y/32.0
            # mask = gt_down_grad_x>0
            # gt_down_grad_x[mask] = torch.sqrt(gt_down_grad_x[mask])
            # gt_down_grad_x[mask==False] = -torch.sqrt(-gt_down_grad_x[mask==False])
            # mask = gt_down_grad_y>0
            # gt_down_grad_y[mask] = torch.sqrt(gt_down_grad_y[mask])
            # gt_down_grad_y[mask==False] = -torch.sqrt(-gt_down_grad_y[mask==False])

            # base_gt = img - (gt_down_grad_x+gt_down_grad_y)/2.0

            # gt_down_grad_x = torch.sin(gt_down_grad_x*math.pi/2)
            # gt_down_grad_y = torch.sin(gt_down_grad_y*math.pi/2)
            # base = output_down - (gt_down_grad_x+gt_down_grad_y)/2.0
            # base_loss = loss0(base,base_gt)
            gen=1.0/1.5
            mask = gt_down_grad_x >0
            th=0.01


            gt_down_grad_x[mask]-=th
            gt_down_grad_x[mask]=torch.where(gt_down_grad_x<0,torch.sin(gt_down_grad_x/th*math.pi/2.0)*th,((gt_down_grad_x)/(1.0-th))**gen*(1.0-th))[mask]
            gt_down_grad_x[mask]+=th

            mask = mask ==False
            gt_down_grad_x[mask]+=th
            gt_down_grad_x[mask]= torch.where(gt_down_grad_x>0,torch.sin(gt_down_grad_x/th*math.pi/2.0)*th,-(((-gt_down_grad_x)/(1.0-th))**gen)*(1.0-th))[mask]
            gt_down_grad_x[mask]-=th

            mask = gt_down_grad_y > 0


            gt_down_grad_y[mask]-=th
            gt_down_grad_y[mask]=torch.where(gt_down_grad_y<0,torch.sin(gt_down_grad_y/th*math.pi/2.0)*th,((gt_down_grad_y)/(1.0-th))**gen*(1.0-th))[mask]
            gt_down_grad_y[mask]+=th

            mask = mask ==False
            gt_down_grad_y[mask]+=th
            gt_down_grad_y[mask]= torch.where(gt_down_grad_y>0,torch.sin(gt_down_grad_y/th*math.pi/2.0)*th,-(((-gt_down_grad_y)/(1.0-th))**gen)*(1.0-th))[mask]
            gt_down_grad_y[mask]-=th



            # mask = gt_down_grad_y>0
            # gt_down_grad_y[mask]= torch.sin(((gt_down_grad_y[mask])**(gen)-0.5)*math.pi)*16.0+16.0

            # gt_down_grad_y[mask==False]=torch.sin(((-((-gt_down_grad_y[mask==False])**(gen))+0.5)*math.pi))*16.0-16.0

            # gt_down_grad_x = torch.sin(gt_down_grad_x*math.pi/2.0)
            # gt_down_grad_y = torch.sin(gt_down_grad_y*math.pi/2.0)

            # gt_down_grad_x/=32.0
            # gt_down_grad_y/=32.0
            output_down_grad_x/=32.0
            output_down_grad_y/=32.0


            # grad_down_xy = output_down_grad_x*output_down_grad_y
            # gt_grad_down_xy = gt_down_grad_x*gt_down_grad_y
            grad_down_x_loss = loss0(output_down_grad_x, gt_down_grad_x)
            grad_down_y_loss = loss0(output_down_grad_y, gt_down_grad_y)
            # grad_down_xy_loss = loss0(grad_down_xy,gt_grad_down_xy)

            base = output_down-output_down_grad_x-output_down_grad_y

            base_loss = loss0(base,gt_base)
            # output = img_up+detail
            # coarse_up = img_up
            # detail_up = img_up - coarse_up
            # detail_up = F.interpolate(detail,scale_factor=2.0,mode='bilinear')


            # output = coarse_up+detail_up

            # coarse_up_down = F.avg_pool2d(output_pad(output),kernel_size=3,padding=0,stride=2)
            # coarse_up_loss = loss0(coarse_up_down,img)
            # output_blur = F.avg_pool2d(output_pad(output),kernel_size=3,padding=0,stride=2)
            # output_blur = F.conv2d(output_pad(output),gaussian,stride=2,padding=0)
            # output_loss0 = loss0(output_blur, img)

            # super loss
            # output_blur = F.conv2d(output_pad(output),gaussian,stride=2,padding=0)

            # output_grad_x = torch.sign(F.conv2d(output,scharrx,stride=1,padding=1))
            # output_grad_y = torch.sign(F.conv2d(output,scharry,stride=1,padding=1))
            # print(output_grad_x.shape)

            # gt_grad_x = F.interpolate(gt_down_grad_x,scale_factor=2.0,mode="bilinear")
            # gt_grad_y = F.interpolate(gt_down_grad_y, scale_factor=2.0, mode="bilinear")




            # output_patch = torch.nn.functional.unfold(output_down,kernel_size=(3,3),padding=1)
            # output_m = torch.mean(output_patch,dim=1,keepdim=True)
            # output_var = torch.mean((output_patch-output_m)**2,dim=1,keepdim=True).reshape(output_down.shape)


            # var_loss = loss0(output_var,gt_var)
            # output_lp = F.conv2d(output,laplacian,padding=1)
            # gt_lp = F.conv2d(img,laplacian,padding=1)*scale
            # lp_loss = loss0(output_var,gt_var)
            # mask = gt_var >1e-6
            # print(np.count_nonzero(mask.cpu().numpy())*1.0/mask.cpu().numpy().size)
            # gt_var +=1e-6
            # s  = output_var/gt_var
            # s = (s<=1.05*gt_var) & (s>gt_var)
            # s = ( s== False ) & (mask)
            # s = s.type(torch.float32)
            # gt_var
            # mask = mask.type(torch.float32)
            # var_loss = torch.mean(mask*torch.abs(((1.05*gt_var-output_var)/(gt_var**5+1e-6)))/(torch.sum(mask)+1e-6))
            # grad_x_loss = loss0(output_grad_x,gt_grad_x)
            # grad_y_loss = loss0(output_grad_y,gt_grad_y)
            # output_global_var = torch.var(output,dim=[2,3],keepdim=True)
            # img_global_var = torch.var(img,dim=[2,3],keepdim=True)
            # global_var_loss = scale*loss0(output_global_var,img_global_var)

            loss = (grad_down_x_loss+grad_down_y_loss)+0.05* detail_loss#+0.01*var_loss#+0.25*global_var_loss

            loss.backward()
            optimizer.step()

            print("epoch ",epoch,"step %d/%d(%02f) :%02f"%(step,lens,step/lens,loss.item()))
            # output = torch.clamp(output,0,1.0)
            # mask = output_var>1.2*gt_var

            if (step)%20 == 0:
                # detail = torch.clamp((output-img_up+1.0)/2.0,0,1.0)
                # vis.images(torch.cat([img_up[:8], torch.clamp(output[:8],0,1.0),detail[:8]], 0) * 255.0,win="super")
                detail = torch.clamp(((output_down-img) + 1.0) / 2.0, 0, 1.0)
                coarse = (img-detail+1.0)/1.8
                vis.images(torch.cat([img[:8], torch.clamp(output_down[:8], 0, 1.0),coarse[:8], detail[:8]], 0) * 255.0, win="sr_base_scharr_noise_0.01")
                vis.line(X=np.column_stack([step+lens*epoch]),Y=np.column_stack([loss.item()]),
                win="sr_loss_base_scharr_noise_0.01",update='append',opts=dict(showlegend=True,legend=["sr_loss_base_scharr_noise_0.01"]))
                vis.line(X=np.column_stack([step+lens*epoch]),Y=np.column_stack([lr_scheduler.get_lr()]),
                win="sr_lr_base_scharr_noise_0.01",update='append',opts=dict(showlegend=True,legend=["sr_lr_base_scharr_noise_0.01"]))
                # vis.images(() * 255.0, win="detail")
        # logdir =
        lr_scheduler.step()
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()},
            "{}/snet_model_{:0>6}.ckpt".format(logdir, epoch+1))


train()