import torch
import torch.nn.functional as F
import torch.nn as nn
import math
def loss_nr(target_image,gt_image):
    batch_size,channel,height,width=gt_image.shape
    target_image = torch.clamp(target_image,0.0,1.0)

    target_patch = torch.nn.functional.unfold(target_image,kernel_size=(11,11),padding=5).reshape([batch_size,-1,height,width]) #b,25,h,w

    # target_patch = torch.nn.functional.unfold(target_patch,kernel_size=(3,3),padding=1).reshape([batch_size,-1,9,height,width]) #b,121,h,w
    # sad -> (0,1)
    # target_image_match =  torch.nn.functional.unfold(target_image,kernel_size=(3,3),padding=1).reshape([batch_size,-1,9,height,width])

    sad_target = (target_patch-target_image)#-1,25,h,w

    # sad_target = torch.mean(sad_target,2).reshape([batch_size,-1,height,width]) #b,121,hxw

    # sad_target = (sad_target - math.exp(-1))/(math.exp(1)-math.exp(-1))
    # sad_target = (sad_target-sad_target.min())/(sad_target.max()-sad_target.min(dim=[2,3]))
    gt_patch = torch.nn.functional.unfold(gt_image,kernel_size=(11,11),padding=5).reshape([batch_size,-1,height,width]) #b,121,h,w

    # gt_patch = torch.nn.functional.unfold(gt_patch,kernel_size=(3,3),padding=1).reshape([batch_size,-1,9,height,width]) #b,121,h,w
    # sad -> (0,1)
    # gt_image_match =  torch.nn.functional.unfold(gt_image,kernel_size=(3,3),padding=1).reshape([batch_size,-1,9,height,width])
    sad_gt = (gt_patch-gt_image)#-1,25,h,w

    # sad_gt = torch.mean((sad_gt),2).reshape([batch_size,-1,height,width])

    # sad_gt = (sad_gt - math.exp(-1))/(math.exp(1)-math.exp(-1))

    # print(sad_target.min(),sad_target.max())
    scale = 1.5
    alpha = 0.05
    inv_mask = sad_gt<0
    sad_gt = torch.abs(sad_gt)
    mask = sad_gt<alpha

    sad_gt[mask]= ((sad_gt[mask]/alpha)**scale)*alpha
    mask = mask == False
    sad_gt[mask]= (((sad_gt[mask]-alpha)/(1-alpha))**(1.0/scale))*(1-alpha)
    sad_gt[inv_mask]=-sad_gt[inv_mask]

    return F.l1_loss(sad_target,sad_gt)+F.mse_loss(target_image,gt_image)


def loss_se(output,target):
    """
    @param output shape [batch_size,channel,height,width
    """
    # output = output.clamp(0,1.0)
    batch_size,channel,height,width = output.shape
    scharrx=torch.Tensor([-3,0,3,-10,0,10,-3,0,3])/16.0
    scharrx.requires_grad = False
    scharry=torch.Tensor([-3,-10,-3,0,0,0,3,10,3])/16.0
    scharry.requires_grad = False
    scharrx  = scharrx.reshape([1,1,3,3]).cuda()
    scharry  = scharry.reshape([1,1,3,3]).cuda()

    output_xy = output.reshape([-1,1,height,width])
    dx = F.conv2d(output_xy,scharrx,stride = 1, padding =1)
    dy = F.conv2d(output_xy,scharry,stride = 1, padding =1)

    target_xy = target.reshape([-1,1,height,width])
    dx_gt = F.conv2d(target_xy,scharrx,stride = 1, padding =1)
    dy_gt = F.conv2d(target_xy,scharry,stride = 1, padding =1)

    # mask = dx_gt < 0
    # dx_gt[mask]= -dx_gt[mask]
    # dx_gt = dx_gt**(1/1.5)
    # dx_gt[mask] = -dx_gt[mask]

    # mask = dy_gt < 0
    # dy_gt[mask]= -dy_gt[mask]
    # dy_gt = dy_gt**(1/1.5)
    # dy_gt[mask] = -dy_gt[mask]

    # gen=1.0/1.5
    # mask = dx_gt >0
    # th=0.025


    # dx_gt[mask]-=th
    # dx_gt[mask]=torch.where(dx_gt<0,torch.sin(dx_gt/th*math.pi/2.0)*th,((dx_gt)/(1.0-th))**gen*(1.0-th))[mask]
    # dx_gt[mask]+=th

    # mask = mask ==False
    # dx_gt[mask]+=th
    # dx_gt[mask]= torch.where(dx_gt>0,torch.sin(dx_gt/th*math.pi/2.0)*th,-(((-dx_gt)/(1.0-th))**gen)*(1.0-th))[mask]
    # dx_gt[mask]-=th

    # mask = dy_gt > 0


    # dy_gt[mask]-=th
    # dy_gt[mask]=torch.where(dy_gt<0,torch.sin(dy_gt/th*math.pi/2.0)*th,((dy_gt)/(1.0-th))**gen*(1.0-th))[mask]
    # dy_gt[mask]+=th

    # mask = mask ==False
    # dy_gt[mask]+=th
    # dy_gt[mask]= torch.where(dy_gt>0,torch.sin(dy_gt/th*math.pi/2.0)*th,-(((-dy_gt)/(1.0-th))**gen)*(1.0-th))[mask]
    # dy_gt[mask]-=th
    return  F.l1_loss(dx,dx_gt)+F.l1_loss(dy,dy_gt)



def loss_sr(output,target):
    """
    @param output shape [batch_size,channel,3,height,width
    """


    # var
    o_mean = F.avg_pool3d(output,3,stride=(1,1,1),padding=(0,1,1)) #b,c,1,h,w
    # o_var = F.avg_pool3d(output**2,3,stride=(1,1,1),padding=(0,1,1)) - o_mean**2 #b,c,1,h,w

    g_mean = F.avg_pool3d(target,3,stride=(1,1,1),padding=(0,1,1)) #b,c,1,h,w
    # g_var = F.avg_pool3d(target**2,3,stride=(1,1,1),padding=(0,1,1)) - g_mean**2 #b,c,1,h,w
    # g_mean.squeeze_(2)
    # g_var.squeeze_(2)
    # o_mean.squeeze_(2)
    # o_var.squeeze_(2)
    og_mean = F.avg_pool3d(output*target,3,stride=(1,1,1),padding=(0,1,1)) #b,c,1,h,w
    mv_loss =  F.l1_loss((o_mean*g_mean),og_mean)
    # mv_loss = F.l1_loss(o_var,g_var)
    # return mv_loss
    output.squeeze_(1)
    target.squeeze_(1)
    # output = output.clamp(0,1.0)
    batch_size,channel,height,width = output.shape
    scharrx=torch.Tensor([-3,0,3,-10,0,10,-3,0,3])/16.0
    scharrx.requires_grad = False
    scharry=torch.Tensor([-3,-10,-3,0,0,0,3,10,3])/16.0
    scharry.requires_grad = False
    scharrx  = scharrx.reshape([3,3]).cuda()
    scharry  = scharry.reshape([3,3]).cuda()

    scharrxs = torch.zeros([max(output.shape[1:]),max(output.shape[1:]),3,3],dtype=torch.float32)
    scharrys = torch.zeros([max(output.shape[1:]),max(output.shape[1:]),3,3],dtype=torch.float32)
    for i in range(max(output.shape[1:])):
        scharrxs[i,i,:,:] = scharrx
        scharrys[i,i,:,:] = scharry
    scharrxs = scharrxs.type(torch.float32).reshape(max(output.shape[1:]),max(output.shape[1:]),3,3).cuda()
    scharrys = scharrys.type(torch.float32).reshape(max(output.shape[1:]),max(output.shape[1:]),3,3).cuda()

    # xy
    # output_xy = output.reshape([-1,1,height,width])
    # dx = F.conv2d(output_xy,scharrx,stride = 1, padding =1)
    # dy = F.conv2d(output_xy,scharry,stride = 1, padding =1)

    # target_xy = target.reshape([-1,1,height,width])
    # dx_gt = F.conv2d(target_xy,scharrx,stride = 1, padding =1)
    # dy_gt = F.conv2d(target_xy,scharry,stride = 1, padding =1)

    # dx_gt = torch.sin(dx_gt)
    # dy_gt = torch.sin(dy_gt)
    # mask = torch.abs(dx_gt)>0.05
    # masky = torch.abs(dy_gt)>0.05
    # loss_xy = torch.sum(torch.abs(dx[mask]-dx_gt[mask]))/(1+torch.sum(mask.type(torch.float32))) +\
    #     torch.sum(torch.abs(dy[masky]-dy_gt[masky]))/(1+torch.sum(masky.type(torch.float32)))
    # loss_xy = F.mse_loss(dx,dx_gt)+F.mse_loss(dy,dy_gt)
    #yz
    output_xy = output.permute([0,2,1,3])
    dx = F.conv2d(output_xy,scharrxs[:output_xy.shape[1],:output_xy.shape[1]],stride = 1, padding =1)
    dy = F.conv2d(output_xy,scharrys[:output_xy.shape[1],:output_xy.shape[1]],stride = 1, padding =1)

    target_xy = target.permute([0,2,1,3])
    dx_gt = F.conv2d(target_xy,scharrxs[:output_xy.shape[1],:output_xy.shape[1]],stride = 1, padding =1)
    dy_gt = F.conv2d(target_xy,scharrys[:output_xy.shape[1],:output_xy.shape[1]],stride = 1, padding =1)
    # mask = torch.abs(dx_gt)>0.05
    # masky = torch.abs(dy_gt)>0.05
    # loss_yz = torch.sum(torch.abs(dx[mask]-dx_gt[mask]))/(1+torch.sum(mask.type(torch.float32))) +\
    #     torch.sum(torch.abs(dy[masky]-dy_gt[masky]))/(1+torch.sum(masky.type(torch.float32)))
    loss_yz = F.l1_loss(dx[:,:,1],dx_gt[:,:,1])+F.l1_loss(dy[:,:,1],dy_gt[:,:,1])

    # zx
    output_xy = output.permute([0,3,1,2])
    dx = F.conv2d(output_xy,scharrxs[:output_xy.shape[1],:output_xy.shape[1]],stride = 1, padding =1)
    dy = F.conv2d(output_xy,scharrys[:output_xy.shape[1],:output_xy.shape[1]],stride = 1, padding =1)

    target_xy = target.permute([0,3,1,2])
    dx_gt = F.conv2d(target_xy,scharrxs[:output_xy.shape[1],:output_xy.shape[1]],stride = 1, padding =1)
    dy_gt = F.conv2d(target_xy,scharrys[:output_xy.shape[1],:output_xy.shape[1]],stride = 1, padding =1)
    # mask = torch.abs(dx_gt)>0.05
    # masky = torch.abs(dy_gt)>0.05
    # loss_zx = torch.sum(torch.abs(dx[mask]-dx_gt[mask]))/(1+torch.sum(mask.type(torch.float32))) +\
    #     torch.sum(torch.abs(dy[masky]-dy_gt[masky]))/(1+torch.sum(masky.type(torch.float32)))
    loss_zx = F.l1_loss(dx[:,:,1],dx_gt[:,:,1])+F.l1_loss(dy[:,:,1],dy_gt[:,:,1])


    # loss_content = F.mse_loss(output,target)
    return loss_yz + loss_zx + mv_loss.item()


def psnr(output,target):
    return 10*torch.log10(1/(1e-6+F.mse_loss(output,target)))