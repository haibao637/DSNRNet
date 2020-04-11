import torch
import torch.nn as nn
import torch.nn.functional as F
from module import *
class SRNet(nn.Module):
    def __init__(self,in_channel):
        super(SRNet, self).__init__()
        ## encode
        self.conv1s=nn.Sequential(
            ConvGnReLU(in_channel,8,3,1,1),
            ConvGnReLU(8,8,3,1,1), #224,224
            # torch.nn.Conv2d(8,8,3,1,1)
            # torch.nn.Tanh()
        )
        self.conv2s = nn.Sequential(
            ConvGnReLU(8, 16, 3, 2, 1),#112,112
            ConvGnReLU(16, 16, 3, 1, 1)
        )
        self.conv3s = nn.Sequential(
            ConvGnReLU(16, 32, 3, 2, 1),  # 56,56
            ConvGnReLU(32, 32, 3, 1, 1)
        )

        #decode
        self.conv4s  = nn.Sequential(
            DeConvGnReLU(32, 16, 3, 2, 1),  # 112,112
            ConvGnReLU(16, 16, 3, 1, 1)
        )
        self.conv5s  = nn.Sequential(
            DeConvGnReLU(32, 8, 3, 2, 1),  # 224,224
            ConvGnReLU(8, 8, 3, 1, 1)
        )
        self.conv6s = nn.Sequential(
            ConvGnReLU(16, 8, 3, 1, 1),
            ConvGnReLU(8,in_channel*8,3,1,1),
            # nn.Tanh()
        )
        self.final_conv = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(in_channel*2,in_channel,3,1,1)
        )

    def forward(self, image) :
        conv1s = self.conv1s(image)
        conv2s = self.conv2s(conv1s)
        conv3s = self.conv3s(conv2s)
        conv4s = self.conv4s(conv3s)
        conv5s = self.conv5s(torch.cat([conv4s,conv2s],1))
        conv6s = self.conv6s(torch.cat([conv5s,conv1s],1))
        return self.final_conv(conv6s)



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

