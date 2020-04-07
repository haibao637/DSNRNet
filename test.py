import argparse
import os

import cv2
import numpy as np
from ctypes import *
import ctypes
import cv2
import numpy as np
import sys
import os
import cv2
import sys
import importlib
import imageio
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# vod =cdll.LoadLibrary("/home/yanjianfeng/.local/libs/libvod.so")

# vls=cdll.LoadLibrary("../build/src/libmobile_vls.so")
#
#
#
# def sharpening(instance,image,size=512):
#     img0=image.astype(np.uint8)
#     input = cast(image.ctypes.data,POINTER(c_uint8))
#     nullptr=cast(None,POINTER(c_uint8))
#
#     output = np.zeros_like(image,np.uint8)
#     output = cast(output.ctypes.data,POINTER(c_uint8))
#
#     state=vls.vls_process_frame_luminance(instance,
#                input,nullptr,0,image.shape[1],image.shape[0],
#                image.shape[1],image.shape[1],output)
#     print(state)
#     output1=np.ctypeslib.as_array(
#         (ctypes.c_uint8*image.size).from_address(ctypes.addressof(output.contents)))
#     output1=output1.reshape(img0.shape).astype(np.uint8)
#
#
#     # output2 = np.zeros_like(image,np.uint8)
#     # output2 = cast(output2.ctypes.data,POINTER(c_uint8))
#
#     # vls.vls_process_frame_luminance(instance,
#     #            input,nullptr,image.shape[1],image.shape[0],
#     #            image.shape[1],image.shape[1],output2,False)
#     # output2=np.ctypeslib.as_array(
#     #     (ctypes.c_uint8*image.size).from_address(ctypes.addressof(output2.contents)))
#     # output2=output2.reshape(img0.shape).astype(np.uint8)
#     return  output1
import torch
import torch.nn.functional as F
import model
loadckpt = "/Users/yanjianfeng/Downloads/snet_model_000199.ckpt"
snet= model.SRNet(1)
# snet = snet.cuda()
state_dict = torch.load(loadckpt,map_location=torch.device('cpu'))
# print(state_dict['model'])
snet.load_state_dict(state_dict['model'], strict=False)
snet = snet.cpu()
import matplotlib.pyplot as plt
def sharpen1(y,max_scale=5):
    y = y.astype(np.float32)/255.0

    # cv2.waitKey()
    y_tensor = torch.Tensor(y[np.newaxis,np.newaxis])
    #
    b = snet(y_tensor)
    # b = cv2.ximgproc.guidedFilter(y,y,radius=1,eps=0.01)

    patch = torch.nn.functional.unfold(y_tensor, kernel_size=3, dilation=1, padding=1, stride=1)
    patch = patch.reshape([1,9,y_tensor.shape[2],y_tensor.shape[3]])
    mean_patch = patch.mean(dim=1,keepdim=True)
    vp = torch.mean((patch- mean_patch)**2,1,keepdim=True)
    # mean = y_tensor.mean(dim=1,keepdim)
    # patch = torch.nn.functional.unfold(y_tensor, kernel_size=3, dilation=1, padding=1, stride=1)
    # patch = patch.reshape([1, 9, y_tensor.shape[2], y_tensor.shape[3]])
    # mean_patch = patch.mean(dim=1, keepdim=True)
    # vp = torch.mean((patch - mean_patch) ** 2, 1, keepdim=True)

    detail =  y_tensor-b
    mean_detail = detail.mean(dim=1,keepdim=True)
    ve = torch.mean((detail- mean_detail)**2,1,keepdim=True)

    mask = ve<1e-4
    ve[mask]=1e-4
    k=1.03
    scale1=torch.clamp(vp*100,0,5.0)
    scale1 = scale1.squeeze().squeeze().detach().cpu().numpy()

    e = (y_tensor-b).squeeze().squeeze().detach().cpu().numpy()
    print(e.min(),e.max(),scale1.min(),scale1.max())
    # cv2.imshow("scale", scale1)
    # cv2.waitKey()
    # plt.imshow(scale1,'rainbow'),plt.show()
    # dpx = cv2.Sobel(y,cv2.CV_32F,1,0,ksize=3)
    # dpy = cv2.Sobel(y,cv2.CV_32F,0,1,ksize=3)
    # # cv2.imshow("orid",15*d)
    # dex = cv2.Sobel(e,cv2.CV_32F,1,0,ksize=3)
    # dey = cv2.Sobel(e,cv2.CV_32F,0,1,ksize=3)
    # pxex = dpx*dex
    # mask = np.abs(pxex)<1e-4
    # pxex[mask]=1.0
    # scale = -dpx*dpx/pxex
    #
    # scale = np.where(pxex<0,scale,max_scale)
    # scale[np.bitwise_and(mask,np.abs(dpx)<1e-3)]=0
    #
    # scale = np.where(scale<scale1,scale,scale1)
    # scale1[np.bitwise_and(mask,np.abs(dpx)<1e-3)]=0
    #
    # scale = np.clip(scale,0,max_scale)
    # cv2.imshow("scale",scale/max_scale)
    sharp = y+e*1
    # maxs = F.max_pool2d(torch.Tensor(y[np.newaxis,np.newaxis]),3,1,1)
    # mins = -F.max_pool2d(torch.Tensor(-y[np.newaxis,np.newaxis]),3,1,1)
    # maxs = maxs.reshape(y.shape).numpy()
    # mins = mins.reshape(y.shape).numpy()
    # alpha=0.5/max_scale
    # sharp[sharp<mins] = (mins+alpha*(sharp-mins))[sharp<mins]
    # sharp[sharp>maxs] = (maxs+alpha*(sharp-maxs))[sharp>maxs]
    # mask = (maxs-mins)<10.0/255
    # sharp[mask]=y[mask]
    sharp = np.clip(sharp*255.0,0,255.0).astype(np.uint8)
    return sharp
import time
def enhance_video(instance,video_path,output_dir):
    # input_dir="/home/yanjianfeng/data/like_data/"
    # output_dir=os.path.join(input_dir,'enhanced')
    if os.path.exists(output_dir)==False:
            os.makedirs(output_dir)
    # video_path='/home/yanjianfeng/data/test_video/clip110_400x720_24_728.mp4'
    cap = cv2.VideoCapture(video_path)

    # output_dir="/home/yanjianfeng/data/outut_video"
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)
    if cap.isOpened()==False:
        exit
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 保存视频的编码
    fourcc=int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = cap.get(cv2.CAP_PROP_PVAPI_MULTICASTIP)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = None
    # print(out)
    idx=0
    times=[]
    while(1):
        ret, img = cap.read()
        print(cap.get(cv2.CAP_PROP_FORMAT))
        # print("{",img.shape[1],",",img.shape[0],"},")
        # break
        if ret==False:
            break
        print(idx)
        idx+=1
        if idx==1:
                print(img.shape)
                if (img.shape[0]%4 !=0)|(img.shape[1]%4 !=0):
                    break
                # vls.vls_set_new_video_info(instance,img.shape[1],img.shape[0],0,255)
        if  idx>50:
            break
        img=img.astype(np.uint8)
        yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
        lumi = yuv[...,0]+0.0
        y=np.ascontiguousarray(yuv[...,0],dtype=np.uint8)
        # cv2.imshow("ori",yuv[...,0])
        # print(y.flags['C_CONTIGUOUS'])
        # print(y.shape)
        # output=sharpening(instance,y)
        # output=o0.astype(np.uint8)
        # print(output.astype(np.float32).mean())
        # print(y.shape)
        # yuv[...,0]=o0
        # rgb1=cv2.cvtColor(yuv,cv2.COLOR_YUV2BGR)
        o1 =sharpen1(y,20.0)
        yuv[...,0]=o1
        rgb2= cv2.cvtColor(yuv,cv2.COLOR_YUV2BGR)
        # yuv[...,0]=o1
        # rgb2=cv2.cvtColor(yuv,cv2.COLOR_YUV2RGB)
        # img[:,:img.shape[1]//2:,:]=rgb1[:,img.shape[1]//2:,:]
        # rgb2[:,rgb2.shape[1]//2:,...]=img[:,rgb2.shape[1]//2:,...]
        # rgb2[:,rgb2.shape[1]//2,...]=0
        # output=np.concatenate([lumi,o0*20,o1*20],1)
        output = np.concatenate([img,rgb2],1)
        # output=o0
        # print(output.shape)
        # output=rgb1
        if out is None:
            video_name=os.path.splitext(os.path.basename(video_path))[0]
            out = cv2.VideoWriter(os.path.join(output_dir,video_name+'.avi'), cv2.VideoWriter_fourcc(*'I420'), 17, (output.shape[1],output.shape[0]))
        out.write(output.astype(np.uint8))
        video_name=os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join(output_dir,video_name)
        if os.path.exists(out_dir) == False:
            os.makedirs(out_dir)
        cv2.imwrite(os.path.join(out_dir,"enhance_%d.jpg"%idx),output)
        # cv2.imwrite(os.path.join(out_dir,"thumb%03d.jpg"%idx),img)
        # break
        # cv2.imshow("enhanced",output.astype(np.uint8))
        # cv2.waitKey()
        # cv2.imwrite("enhance_%d.jpg"%idx,output)
        # break
    # cap.release()
    # if out is not None:

    #     out.release()
    # input = cast(np.zeros([6],dtype=np.float32).ctypes.data,POINTER(c_float))
    # vls.vls_get_report(instance,input,6)
    # output=np.ctypeslib.as_array(
    #     (ctypes.c_float*6).from_address(ctypes.addressof(input.contents)))
    # output[-2]=output[-2]/output[1]
    # print(output)


# if __name__ == "__main__":
#
#
#     output_dir="/home/haibao637/data/download_out_cnn/"
#     input_dir="/home/haibao637/data/download"
#     instance = c_uint64(0)
#     # status=vls.vls_create_instance(byref(instance))
#     #
#     # status = vls.vls_init(instance,0)
#     # vls.vls_set_enhance_scale(instance,10)
#     # vls.vls_set_noise_scale(instance,3)
#     # print(status)
#     for video in sorted(os.listdir(input_dir))[10:20]:
#         # if (video.find("clip15_")==-1 ) :
#         #     continue
#         # if os.path.exists(os.path.join(output_dir,video)):
#         #     continue
#         video_path=os.path.join(input_dir,video)
#         print(video_path,'enhancing...')
#         enhance_video(instance,video_path,output_dir)
#     # enhance_video(instance,"/home/haibao637/data/bigolive_dumping/uid_1537369101_date_1-29-21-5-7.264",output_dir)
#     # img = cv2.imread("/home/haibao637/data/like_real_test_out_detail/256LNc/thumb_1.bmp")
#     # yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
#     # vls.vls_set_new_video_info(instance,img.shape[1],img.shape[0],0,255)
#     # o = sharpening(instance,yuv[...,0])
#     # cv2.imwrite("o.bmp",o)
#     # print(o.astype(np.float32).mean())
#     # vls.vls_release_resource(instance)
#     # vls.vls_destroy_instance(byref(instance))

enhance_video(0,"data/clip36_400x720_18_1667.mp4","output/")
# img = cv2.imread("//home/haibao637/data/like_real_test_out_detail/256LNc/thumb_302.bmp")
# img = img.astype(np.float32)/255.0
# yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
# y = np.ascontiguousarray(yuv[...,0])
# sharp = sharpen1(y)
# sharp = np.clip(sharp,0,1.0)
# yuv[...,0]=sharp
# rgb = cv2.cvtColor(yuv,cv2.COLOR_YUV2BGR)
# rgb  = np.clip(rgb,0,1.0)
# cv2.imshow("ori",img)
# cv2.imshow("rgb",rgb)
# delta = sharp - y
# cv2.imshow("delta",(delta*10.0+1.0)/2.0)
# cv2.waitKey()