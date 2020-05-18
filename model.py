import torch
import torch.nn as nn
import torch.nn.functional as F
from module import *
from convlstm import ConvLSTMCell


class SRNet(nn.Module):
    def __init__(self):
        super(SRNet,self).__init__()

        self.LapPyrNet = LapPyrNet()
        self.PyrFusionNet = PyrFusionNet()
        self.ReconNet = ReconNet()

        # to enhance edge
        # self.EnhanceFusionNet = PyFusionNet()
    # def freezesr(self):
    #     for param in self.LapPyrNet.parameters():
    #         param.requires_grad = False
    #     for param in self.PyrFusionNet.parameters():
    #         param.requires_grad = False

    def forward(self,images):
        """
        @param: images  b,v,c,h,w
        """
        b,v,c,h,w = images.shape
        image = images[:,v//2]
        # images = images.view(-1,c,h,w)

        pyrfeat = self.LapPyrNet(images)
        feats = self.PyrFusionNet(pyrfeat)
        detail = self.ReconNet(feats)
        base = F.interpolate(image,scale_factor = 4.0,align_corners=False,mode='bicubic').clamp(0,1.0)
        enhanced = base+detail
        # enhanced_1 = self.senet(enhanced)
        # enhance_detail = enhanced+ self.EnhanceFusionNet(pyrfeat)
        return base.clamp(0,1.0), enhanced.clamp(0,1.0)#,enhanced_1.clamp(0,1.0)


# class SRNet(nn.Module):
#     def __init__(self):
#         super(SRNet, self).__init__()
#         ## encode
#         in_channel = 3
#         self.feat_conv=nn.Sequential(
#             nn.Conv2d(in_channel,64,3,1,1),
#             *([ResidualBlock(64)]*5),
#             nn.Conv2d(64,64*4,3,1,1),
#             nn.PixelShuffle(2),
#             nn.ReLU(),
#             nn.Conv2d(64,64*4,3,1,1),
#             nn.PixelShuffle(2),
#             nn.ReLU(),
#             nn.Conv2d(64,64,3,1,1),
#             nn.ReLU(),
#             nn.Conv2d(64,in_channel,3,1,1),
#             # torch.nn.Tanh()
#         )
#         # self.rnn_conv0 = ConvLSTMCell(16,16)
#         # self.rnn_conv1 = ConvLSTMCell(16,16)

#         # self.up_conv1 = nn.Sequential(
#         #     nn.ReplicationPad2d(1),
#         #     ConvBnReLU(16+in_channel, 16, 3, 1,  0),  # 224,224
#         #     nn.PixelShuffle(2), #b,8,h/2,w/2 -> b,2,h,w
#         #     # ConvGnReLU(4, 4, 3, 1, 1, 1),  # 224,224
#         #     # ConvGnReLU(4, 2, 3, 1, 1, 1),  # 224,224
#         #     nn.Conv2d(4,64,3,1,0),
#         #     nn.PixelShuffle(2),
#         #     nn.Conv2d()

#         # )
#         # self.up_conv2  = nn.Sequential(
#         #     ConvGnReLU(16+in_channel*3,16,3,1,2,2), #224,224
#         #     ConvGnReLU(16, 16, 3, 1, 2, 2),  # 224,224
#         #     ConvGnReLU(16, 16, 3, 1, 2, 2),  # 224,224
#         #     ConvGnReLU(16, 16, 3, 1, 2, 2),  # 224,224
#         #     nn.PixelShuffle(2), #b,8,h/2,w/2 -> b,2,h,w
#         #     nn.Conv2d(4,in_channel,3,1,1)
#         # )
#         self.final_conv = nn.Sequential(
#             nn.ReplicationPad2d(1),
#             ConvBnReLU(in_channel*2, 16, 3,  1, 1),  # 224,224
#             nn.ReplicationPad2d(1),
#             ConvBnReLU(16, 16, 3, 1, 1),  # 224,224
#             nn.ReplicationPad2d(1),
#             nn.Conv2d(16,in_channel,3,1,1)
#         )
#         self.tripleFilter = TripleFiter(3)
#         self.tripleGuidedFilter = TripleGuidedFiter(3)
#         self.guidedFilter = GuidedFilter(0.0001)
#     def forward(self, images) :
#         """
#         @param images: b,c,v,h,w
#         """
#         images = images.permute(0,2,1,3,4)
#         batch_size,channel,_,height,width = images.shape
#         #三维双边滤波
#         # print(images.shape)
#         img = self.tripleFilter(images)
#         # img = images[:,:,1]
#         feat = self.feat_conv(img)
#         # print(feat.shape,img.shape)
#         # up_conv1 = self.up_conv1(torch.cat([feat,img],1))
#         # img_up = F.interpolate(images.reshape([batch_size,-1,height,width]),scale_factor=2.0,align_corners=True,mode='bicubic').view(batch_size,channel,-1,height*2,width*2)
#         # img_up[:,:,1] = up_conv1


#         # up_conv2 = self.up_conv2(torch.cat([up_conv1,img_up],1))
#         images_strip = images.permute(0,2,1,3,4).reshape([-1,channel,height,width])
#         # imgs = self.guidedFilter(images_strip)
#         # images = images.reshape([batch_size,-1,height,width])
#         img_up = F.interpolate(images_strip,scale_factor=4.0,align_corners=False,mode='bicubic').view(batch_size,-1,channel,height*4,width*4).permute(0,2,1,3,4)
#         # imgs = imgs.reshape(batch_size,-1,channel,height,width).permute(0,2,1,3,4)
#         # img_up = F.interpolate(images_strip,scale_factor=2.0,align_corners=False,mode='bilinear').view([batch_size,channel,-1,height*2,width*2])
#         img_up = self.tripleGuidedFilter(img_up,images)
#         # print(img_up.shape)
#         # img_up = self.tripleFilter(imgs_up)
#         # final_conv = self.final_conv(torch.cat([img_up,up_conv1],1))
#         enhanced =  img_up + feat

#         return _,enhanced.clamp(0,1.0)

#     # def init(self,shape):
#     #     self.state0 = None
#     #     self.state1 = None
#     # def super(self,image):#b,1,h,w
#     #     batch_size,channel,height,width = image.shape
#     #     img_up = F.interpolate(image,scale_factor=2.0,mode='bicubic',align_corners=True)
#     #     img_up = img_up.reshape([batch_size,channel,height*2,width*2])
#     #     img_feat = self.feat_conv(img_up)
#     #     self.state0 = self.rnn_conv0(img_feat,self.state0)
#     #     self.state1 = self.rnn_conv1(self.state0[0],self.state1)
#     #     delta  = self.final_conv(self.state1[0])
#     #     print(torch.abs(delta).mean())
#     #     # img_up = F.avg_pool2d(img_up,3,1,1)
#     #     enhanced =  img_up*(1+delta)
#     #     return enhanced.clamp(0,1.0)

# class SENet(nn.Module):
#     def __init__(self,in_channel):
#         super(SENet, self).__init__()
#         ## encode

#         self.conv1s=nn.Sequential(
#             ConvGnReLU(in_channel,16,3,1,1),
#             ConvGnReLU(16,16,3,1,2,2), #224,224
#             ConvGnReLU(16, 16, 3, 1, 2, 2),  # 224,224
#             ConvGnReLU(16, 16, 3, 1, 2, 2),  # 224,224

#             # torch.nn.Tanh()
#         )
#         self.final_conv = torch.nn.Conv2d(16+in_channel,1,3,1,1,bias=False)
#         # self.conv2s = nn.Sequential(
#         #     ConvGnReLU(8, 16, 3, 2, 1),#112,112
#         #     ConvGnReLU(16, 16, 3, 1, 1)
#         # )
#         # self.conv3s = nn.Sequential(
#         #     ConvGnReLU(16, 32, 3, 2, 1),  # 56,56
#         #     ConvGnReLU(32, 32, 3, 1, 1)
#         # )
#         #
#         # #decode
#         # self.conv4s  = nn.Sequential(
#         #     DeConvGnReLU(32, 16, 3, 2, 1),  # 112,112
#         #     ConvGnReLU(16, 16, 3, 1, 1)
#         # )
#         # self.conv5s  = nn.Sequential(
#         #     DeConvGnReLU(32, 8, 3, 2, 1),  # 224,224
#         #     ConvGnReLU(8, 8, 3, 1, 1)
#         # )
#         #
#         # self.coarse = nn.Sequential(
#         #     ConvGnReLU(16,8,3,1,1),
#         #     ConvGnReLU(8,4,3,1,1),
#         #     nn.Conv2d(4,in_channel,3,1,1)
#         # )
#         # self.super =nn.Sequential(
#         #     ConvGnReLU(in_channel*2,8,3,1,1),
#         #     DeConvGnReLU(8, 4, 3, 2, 1),
#         #     ConvGnReLU(4,4,3,1,1),
#         #     nn.Conv2d(4,in_channel,3,1,1)
#         # )
#         # self.conv6s = nn.Sequential(
#         #     # DeConvGnReLU(16, 8, 3, 2, 1),
#         #     ConvGnReLU(in_channel, 8, 3, 1, 1),
#         #     ConvGnReLU(8, 8, 3, 1, 1),
#         #     nn.PixelShuffle(2),
#         #     nn.Conv2d(2,1,3,1,1)
#         #     # nn.Tanh()
#         # )

#     def forward(self, image):

#         height,width = image.shape[2:4]
#         # print(height,width)
#         # image = F.interpolate(image,size=[224,224],mode='bilinear')
#         x= self.conv1s(image)
#         x=self.final_conv(torch.cat([x,image],1))#.clamp(0,1.0)
#         # return F.interpolate(x,size=(height,width),mode='bilinear')
#         return image*(1+x)






# class SDNet(nn.Module):
#     def __init__(self,in_channel):
#         super(SDNet, self).__init__()
#         ## encode

#         self.conv1s=nn.Sequential(
#             ConvGnReLU(in_channel*3,16,3,1,1),
#             ConvGnReLU(16,16,3,1,2,2), #224,224
#             ConvGnReLU(16, 16, 3, 1, 2, 2),  # 224,224
#             ConvGnReLU(16, 16, 3, 1, 2, 2),  # 224,224

#             # torch.nn.Tanh()
#         )
#         self.final_conv = nn.Sequential(
#             nn.PixelShuffle(2), #b,8,h/2,w/2 -> b,2,h,w
#             nn.Conv2d(4,in_channel,3,1,1)
#         )

#     def forward(self, images):
#         """
#         images : [img_t0,img_t1] b,2,h,w
#         """
#         # height,width = images.shape[-2:]
#         # height = height*2
#         # width = width*2
#         # print(images.shape)

#         x= self.conv1s(torch.cat([images,images[:,1:]-images[:,:1]],1))
#         x=self.final_conv(torch.cat([x],1)).clamp(-1.0,1.0)
#         delta = images[:,1:] - images[:,:1]
#         delta_up = F.interpolate(delta[:,:1],scale_factor=2.0,mode='bicubic') # b,1,h,w
#         # img_up = F.interpolate(images[:,:1],scale_factor=2.0,mode='bicubic') # b,1,h,w
#         return delta_up*x

