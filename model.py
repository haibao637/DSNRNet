import torch
import torch.nn as nn
import torch.nn.functional as F
from module import *
class SRNet(nn.Module):
    def __init__(self,in_channel):
        super(SRNet, self).__init__()
        ## encode
        self.feat_conv=nn.Sequential(
            ConvGnReLU(in_channel,16,3,1,1),
            ConvGnReLU(16,16,3,1,2,2), #224,224
            ConvGnReLU(16, 16, 3, 1, 2, 2),  # 224,224
            ConvGnReLU(16, 16, 3, 1, 2, 2),  # 224,224

            # torch.nn.Tanh()
        )
        # self.rnn_conv0 = ConvLSTMCell(16,16)
        # self.rnn_conv1 = ConvLSTMCell(16,16)

        self.up_conv1 = nn.Sequential(
            ConvGnReLU(16+in_channel, 16, 3, 1, 2, 2),  # 224,224
            nn.PixelShuffle(2), #b,8,h/2,w/2 -> b,2,h,w
            nn.Conv2d(4,in_channel,3,1,1)

        )
        # self.up_conv2  = nn.Sequential(
        #     ConvGnReLU(16+in_channel*3,16,3,1,2,2), #224,224
        #     ConvGnReLU(16, 16, 3, 1, 2, 2),  # 224,224
        #     ConvGnReLU(16, 16, 3, 1, 2, 2),  # 224,224
        #     ConvGnReLU(16, 16, 3, 1, 2, 2),  # 224,224
        #     nn.PixelShuffle(2), #b,8,h/2,w/2 -> b,2,h,w
        #     nn.Conv2d(4,in_channel,3,1,1)
        # )
        self.tripleFilter = TripleGuidedFiter([3,3,3],[0.1])

    def forward(self, images) :
        """
        @param images: b,c,v,h,w
        """
        batch_size,channel,_,height,width = images.shape
        #三维双边滤波

        img = self.tripleFilter(images)
        feat = self.feat_conv(img)
        # print(feat.shape,img.shape)
        up_conv1 = self.up_conv1(torch.cat([feat,img],1))
        # img_up = F.interpolate(images.reshape([batch_size,-1,height,width]),scale_factor=2.0,align_corners=True,mode='bicubic').view(batch_size,channel,-1,height*2,width*2)
        # img_up[:,:,1] = up_conv1


        # up_conv2 = self.up_conv2(torch.cat([up_conv1,img_up],1))
        img_up = F.interpolate(images[:,:,1],scale_factor=2.0,align_corners=True,mode='bicubic')
        # img_up = self.tripleFilter(imgs_up)
        enhanced =  img_up*(1+up_conv1)
        return enhanced.clamp(0,1.0)

    # def init(self,shape):
    #     self.state0 = None
    #     self.state1 = None
    # def super(self,image):#b,1,h,w
    #     batch_size,channel,height,width = image.shape
    #     img_up = F.interpolate(image,scale_factor=2.0,mode='bicubic',align_corners=True)
    #     img_up = img_up.reshape([batch_size,channel,height*2,width*2])
    #     img_feat = self.feat_conv(img_up)
    #     self.state0 = self.rnn_conv0(img_feat,self.state0)
    #     self.state1 = self.rnn_conv1(self.state0[0],self.state1)
    #     delta  = self.final_conv(self.state1[0])
    #     print(torch.abs(delta).mean())
    #     # img_up = F.avg_pool2d(img_up,3,1,1)
    #     enhanced =  img_up*(1+delta)
    #     return enhanced.clamp(0,1.0)



class SENet(nn.Module):
    def __init__(self,in_channel):
        super(SENet, self).__init__()
        ## encode

        self.conv1s=nn.Sequential(
            ConvGnReLU(in_channel,16,3,1,1),
            ConvGnReLU(16,16,3,1,2,2), #224,224
            ConvGnReLU(16, 16, 3, 1, 2, 2),  # 224,224
            ConvGnReLU(16, 16, 3, 1, 2, 2),  # 224,224

            # torch.nn.Tanh()
        )
        self.final_conv = torch.nn.Conv2d(16+in_channel,1,3,1,1,bias=False)
        # self.conv2s = nn.Sequential(
        #     ConvGnReLU(8, 16, 3, 2, 1),#112,112
        #     ConvGnReLU(16, 16, 3, 1, 1)
        # )
        # self.conv3s = nn.Sequential(
        #     ConvGnReLU(16, 32, 3, 2, 1),  # 56,56
        #     ConvGnReLU(32, 32, 3, 1, 1)
        # )
        #
        # #decode
        # self.conv4s  = nn.Sequential(
        #     DeConvGnReLU(32, 16, 3, 2, 1),  # 112,112
        #     ConvGnReLU(16, 16, 3, 1, 1)
        # )
        # self.conv5s  = nn.Sequential(
        #     DeConvGnReLU(32, 8, 3, 2, 1),  # 224,224
        #     ConvGnReLU(8, 8, 3, 1, 1)
        # )
        #
        # self.coarse = nn.Sequential(
        #     ConvGnReLU(16,8,3,1,1),
        #     ConvGnReLU(8,4,3,1,1),
        #     nn.Conv2d(4,in_channel,3,1,1)
        # )
        # self.super =nn.Sequential(
        #     ConvGnReLU(in_channel*2,8,3,1,1),
        #     DeConvGnReLU(8, 4, 3, 2, 1),
        #     ConvGnReLU(4,4,3,1,1),
        #     nn.Conv2d(4,in_channel,3,1,1)
        # )
        # self.conv6s = nn.Sequential(
        #     # DeConvGnReLU(16, 8, 3, 2, 1),
        #     ConvGnReLU(in_channel, 8, 3, 1, 1),
        #     ConvGnReLU(8, 8, 3, 1, 1),
        #     nn.PixelShuffle(2),
        #     nn.Conv2d(2,1,3,1,1)
        #     # nn.Tanh()
        # )

    def forward(self, image):
        # conv1s = self.conv1s(image)
        # conv2s = self.conv2s(conv1s)
        # conv3s = self.conv3s(conv2s)
        # conv4s = self.conv4s(conv3s)
        # conv5s = self.conv5s(torch.cat([conv4s,conv2s],1))
        # feat = torch.cat([conv5s,conv1s],1)
        # # print(feat.shape)
        # coarse = self.coarse(feat)
        #
        # feat = torch.cat([image,coarse],1)
        # # super = self.super(feat)
        # return coarse
        height,width = image.shape[2:4]
        # print(height,width)
        # image = F.interpolate(image,size=[224,224],mode='bilinear')
        x= self.conv1s(image)
        x=self.final_conv(torch.cat([x,image],1))
        # return F.interpolate(x,size=(height,width),mode='bilinear')
        return x

