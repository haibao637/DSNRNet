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

def psnr(output,target):
    return sum([10*np.log10(1/(np.mean((output[...,i]-target[...,i])**2))) for i in range(target.shape[-1])])/(target.shape[-1])

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
from model import SRNet
loadckpt = "/home/yanjianfeng/data/srnet_4.0/snet_model_000033.ckpt"
snet= SRNet(3).cuda()
snet = snet.eval()
# snet = snet.cuda()
state_dict = torch.load(loadckpt,map_location=torch.device('cuda'))
print(state_dict["model"])
# print(state_dict['model'])
snet.load_state_dict(state_dict['model'], strict=False)
# snet = snet.cpu()
import matplotlib.pyplot as plt
import time
times =  []
def super(ys):

    start = time.time()
    # cv2.waitKey()
    # if prev_y is None:
    #     prev_y = y
    output_pad = torch.nn.ReplicationPad2d(1)
    ys = np.stack(ys,0)
    y_tensor = torch.Tensor(ys).cuda()#view,h,w
    y_tensor = y_tensor.unsqueeze(0).permute(0,4,1,2,3)

    # y_tensor=y_tensor.unsqueeze(0).unsqueeze(0) #1,1,3,h,w
    # print(y_tensor.shape)
    # y_tensor = F.interpolate(y_tensor,scale_factor=0.5,mode='bicubic',align_corners=True)

    # y_tensor = F.avg_pool2d(output_pad(y_tensor),kernel_size=3,stride=1)

    # gaussian = torch.Tensor([1,2,1,2,4,2,1,2,1]).type(torch.float32).reshape(1,1,3,3)/16.0
    # gaussian = gaussian.cuda()
    # laplacian[:,:,1,1]=8
    # scale=2.0
    # output_pad = torch.nn.ReplicationPad2d(1)
    # img_down = F.interpolate(y_tensor,scale_factor=0.5,mode="bilinear")

    # img_down = img_down.reshape([img_down.shape[0],2,-1,img_down.shape[2],img_down.shape[3]])
    with torch.no_grad():
        sup =  snet(y_tensor).permute(0,2,3,1).squeeze(0).detach().cpu().numpy()#b,v,1,h,w
        end  = time.time()
        times.append(end-start)
        return sup


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
    prev_y = None
    prev_sup = None
    ys = []
    psnrs =[]
    prev = None
    while(1):
        ret, img = cap.read()
        print(cap.get(cv2.CAP_PROP_FORMAT))
        # print("{",img.shape[1],",",img.shape[0],"},")
        # break
        if ret==False:
            break
        idx+=1
        # if  idx>200:
        #     break

        img=img.astype(np.float32)/255.0
        # print(img.shape)
        if idx>3:
            # b = cv2.GaussianBlur(img, (0, 0), sigmaX=1.5)
            b = cv2.resize(img,(0,0),fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
            # b =  cv2.bilateralFilter(b,3,1.0,1.0)
            # b = img
            ys = ys[1:]+[b]
            o1= super(ys)
            # print(o1.min(),o1.max())
            # print(o1.shape,yuv[...,0].shape)
            # psnrs.append(psnr(o1,prev))
        else:

            o1 = img#cv2.resize(y,(0,0),fx=2.0,fy=2.0,interpolation=cv2.INTER_CUBIC)
            # y = cv2.GaussianBlur(y, (0, 0), sigmaX=1.5)
            # b = cv2.GaussianBlur(img, (0, 0), sigmaX=1.5)
            b = cv2.resize(img,(0,0),fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
            # b =  cv2.bilateralFilter(b,3,1.0,1.0)
            # y =
            # b =img
            ys = ys+[b]
            o1 = super((ys+[b,b])[:3])
        rgb3 = cv2.resize(b,(0,0),fx=2.0,fy=2.0,interpolation=cv2.INTER_LINEAR)
        output = np.concatenate([img,o1],1)
        output = output*255.0
        if out is None:
            video_name=os.path.splitext(os.path.basename(video_path))[0]
            out = cv2.VideoWriter(os.path.join(output_dir,video_name+'.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (output.shape[1],output.shape[0]))
        out.write(output.astype(np.uint8))
        video_name=os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join(output_dir,video_name)
        if os.path.exists(out_dir) == False:
            os.makedirs(out_dir)
        cv2.imwrite(os.path.join(out_dir,"enhance_%d.jpg"%idx),output)

        prev = img
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

def enhance_images(img_path,output_dir):
    prev_y = None
    prev_sup = None
    out = None
    idx=0
    ys = []
    lens = len(os.listdir(img_path))
    need_view = 5
    psnrs = []
    prev = None
    for img_name in sorted(os.listdir(img_path)):
        img_name  = os.path.join(img_path,img_name)
        img = cv2.imread(img_name)
        idx+=1
        # print("{",img.shape[1],",",img.shape[0],"},")
        # break
        # if ret==False:
        #     break

        # if  idx>200:
        #     break


        img=img.astype(np.float32)/255.0
        # img1 = cv2.resize(img,(0,0),fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
        # print(img.shape)
        # yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)

        # lumi = yuv[...,0]+0.0
        # y=np.ascontiguousarray(yuv[...,0],dtype=np.float32)
        # print(y.min(),y.max())

        if idx>3:
            # b = cv2.GaussianBlur(img, (0, 0), sigmaX=1.5)
            b = cv2.resize(img,(0,0),fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)

            ys = ys[1:]+[b]
            o1= super(ys)
            # print(o1.min(),o1.max())
            # print(o1.shape,yuv[...,0].shape)
            psnrs.append(psnr(o1,prev))
        else:

            # o1 = img#cv2.resize(y,(0,0),fx=2.0,fy=2.0,interpolation=cv2.INTER_CUBIC)
            # y = cv2.GaussianBlur(y, (0, 0), sigmaX=1.5)
            # b = cv2.GaussianBlur(img, (0, 0), sigmaX=1.5)
            b = cv2.resize(img,(0,0),fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
            # y =
            ys = ys+[b]
            o1 = super((ys+[b,b])[:3])
        rgb3 = cv2.resize(b,(0, 0),fx=2.0,fy=2.0,interpolation=cv2.INTER_LINEAR)
        # yuv = cv2.resize(yuv,(0,0),fx=2.0,fy=2.0,interpolation=cv2.INTER_CUBIC)
        # y = yuv[...,0]+0.0
        # yuv[...,0]=o1
        # rgb2= cv2.cvtColor(yuv,cv2.COLOR_YUV2BGR)
        # rgb2 = np.clip(rgb2,0,1.0)
        # if idx > 3:
        #     psnrs.append(psnr(rgb2,img))
        # y = cv2.resize(y,(0,0),fx=0.5,fy=0.5,interpolation=cv2.INTER_NEAREST)
        # y = cv2.resize(y,(0,0),fx=2.0,fy=2.0,interpolation=cv2.INTER_CUBIC)
        # psnrs.append(psnr(img,rgb3))
        # yuv[...,0]=y
        # rgb3= cv2.cvtColor(yuv,cv2.COLOR_YUV2BGR)
        # # yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
        # rgb3 = np.clip(rgb3,0,1.0)
        # if idx > 3:
        #     psnrs.append(psnr(img,o1))
        output = np.concatenate([img,o1,rgb3],1)
        output = (output*255.0).astype(np.uint8)

        prev = img
        # if out is None:
        #     video_name=os.path.splitext(os.path.basename(img_path))[0]
        #     out = cv2.VideoWriter(os.path.join(output_dir,video_name+'.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 1, (output.shape[1],output.shape[0]))
        # out.write(output.astype(np.uint8))
        video_name=os.path.splitext(os.path.basename(img_path))[0]
        out_dir = os.path.join(output_dir,video_name)
        if os.path.exists(out_dir) == False:
            os.makedirs(out_dir)
        cv2.imwrite(os.path.join(out_dir,"%08d.jpg"%idx),output)
    print("psnr:",sum(psnrs)/len(psnrs))
    # print(psnrs)
if __name__ == "__main__":


    output_dir="/home/yanjianfeng/data/douyin_super_out_guided_2/"
    input_dir="/home/yanjianfeng/data/douyin/"
    instance = c_uint64(0)
    # status=vls.vls_create_instance(byref(instance))
    #
    # status = vls.vls_init(instance,0)
    # vls.vls_set_enhance_scale(instance,10)
    # vls.vls_set_noise_scale(instance,3)
    # print(status)
    # for video in sorted(os.listdir(input_dir)):
    #     # if (video.find("gan_hr.mp4")==-1 ) :
    #     #     continue
    #     # if os.path.exists(os.path.join(output_dir,video)):
    #     #     continue
    #     video_path=os.path.join(input_dir,video)
    #     print(video_path,'enhancing...')
    #     enhance_video(instance,video_path,output_dir)

    enhance_images("/home/yanjianfeng/data/Vid4/GT/calendar","/home/yanjianfeng/data/~/data/images_super_out_cnn_5_std_4/")
    # print(sum(times[3:])/float(len(times[3:])))
#     # enhance_video(instance,"/home/haibao637/data/bigolive_dumping/uid_1537369101_date_1-29-21-5-7.264",output_dir)
#     # img = cv2.imread("/home/haibao637/data/like_real_test_out_detail/256LNc/thumb_1.bmp")
#     # yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
#     # vls.vls_set_new_video_info(instance,img.shape[1],img.shape[0],0,255)
#     # o = sharpening(instance,yuv[...,0])
#     # cv2.imwrite("o.bmp",o)
#     # print(o.astype(np.float32).mean())
#     # vls.vls_release_resource(instance)
#     # vls.vls_destroy_instance(byref(instance))

# enhance_video(0,"/home/haibao637/data/clip36_400x720_18_1667.mp4","output/")
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