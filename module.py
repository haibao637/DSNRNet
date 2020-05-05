import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from  dcn.deform_conv import ModulatedDeformConvPack as DCN
def ConvGnReLU(in_channels,out_channels,kernel_size,stride,padding,dilation=1,bias=False):
    num_group = max(1,out_channels//8)
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias),
            nn.GroupNorm(num_groups=num_group,num_channels=out_channels),
            nn.LeakyReLU()
        )

def ConvGnReLU(in_channels,out_channels,kernel_size,stride,padding,dilation=1,bias=False):
    num_group = max(1,out_channels//8)
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias),
            nn.GroupNorm(num_groups=num_group,num_channels=out_channels),
            nn.LeakyReLU()
        )

def Conv3dGnReLU(in_channels,out_channels,kernel_size,stride,padding,dilation=1,bias=False):
    num_group = max(1,out_channels//8)
    return nn.Sequential(
            nn.Conv3d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias),
            nn.GroupNorm(num_groups=num_group,num_channels=out_channels),
            nn.LeakyReLU()
        )

def ConvBnReLU(in_channels,out_channels,kernel_size,stride,padding,dilation=1,bias=False):
    # num_group = max(1,out_channels//8)
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
def DeConvBnReLU(in_channels, out_channels, kernel_size, stride, padding,out_padding=1, bias=False):
    # num_group = max(1, out_channels // 8)
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding,
                           output_padding=out_padding,
                          bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU()
    )

def DeConvGnReLU(in_channels, out_channels, kernel_size, stride, padding,out_padding=1, bias=False):
    num_group = max(1, out_channels // 8)
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding,
                           output_padding=out_padding,
                          bias=bias),
        nn.GroupNorm(num_groups=num_group, num_channels=out_channels),
        nn.LeakyReLU()
    )


class TripleFiter(nn.Module):
    def __init__(self,kernel_size):
        super(TripleFiter, self).__init__()
        self.epsilon = Parameter(torch.Tensor([0.1]))#torch.autograd.Variable(torch.Tensor([1.0]),requires_grad = True)
        self.kernel_size = kernel_size
        # self.guide = GuidedFilter(0.001)
        # self.epsilon_conv = nn.Sequential(
        #     nn.Conv3d(3,16,3,1,1),
        #     nn.LeakyReLU(),
        #     nn.Conv3d(16,16,3,1,1),
        #     nn.AdaptiveAvgPool3d((1,1,1))
        # )
        # self.fc = nn.Sequential(
        #     nn.Linear(16,1),
        #     nn.LeakyReLU()
        # )
    def forward(self,images):
        device = torch.cuda.current_device()
        # self.epsilon = self.epsilon.to(device)
        """
        @param images: b,c,3,h,w

        @return b,c,h,w
        """

        mean = F.avg_pool3d(images,self.kernel_size,stride=(1,1,1),padding=(0,self.kernel_size//2,self.kernel_size//2)) #b,c,1,h,w
        var = F.avg_pool3d(images**2,self.kernel_size,stride=(1,1,1),padding=(0,self.kernel_size//2,self.kernel_size//2)) - mean**2 #b,c,1,h,w

        mean.squeeze_(2)
        var.squeeze_(2)
        # guids = []
        # epsilon = self.epsilon_conv(images)

        # epsilon = epsilon.view(epsilon.shape[0],epsilon.shape[1])
        # epsilon = self.fc(epsilon).view(epsilon.shape[0],1,1,1)
        # print(epsilon)
        #  epsilon = self.epsilon**2
        # for ep in self.epsilon:
        a = var/(var+self.epsilon**2)
        b = (1-a)*mean

        a = F.avg_pool2d(a,3,stride=(1,1),padding=(1,1)) #b,c,h,w
        b  = F.avg_pool2d(b,3,stride=(1,1),padding=(1,1)) #b,c,h,w
        q=a*images[:,:,images.shape[2]//2]+b
        # print(q.shape)
        # return q
        return 2*images[:,:,images.shape[2]//2]-q#b,c,h,w
class GuidedFilter(nn.Module):
    def __init__(self,epsilon=0.01):
        super(GuidedFilter,self).__init__()
        self.epsilon = abs(epsilon)
        # self.epsilon_conv = nn.Sequential(
        #     nn.Conv2d(3,16,3,1,1),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(16,16,3,1,1),
        #     nn.AdaptiveAvgPool2d((1,1,1))
        # )
        # self.fc = nn.Sequential(
        #     nn.Linear(16,1),
        #     nn.Sigmoid()
        # )
    def forward(self,images):
        mean = F.avg_pool2d(images,3,stride=1,padding=1) #b,c,h,w
        var = F.avg_pool2d(images**2,3,1,1) - mean**2 #b,c,h,w
        # epsilon = self.epsilon_conv(images)

        # epsilon = epsilon.view(epsilon.shape[0],epsilon.shape[1])
        # epsilon = self.fc(epsilon).view(epsilon.shape[0],1,1,1)

        #  epsilon = self.epsilon**2
        # for ep in self.epsilon:
        a = var/(var+self.epsilon)
        b = (1-a)*mean

        a = F.avg_pool2d(a,3,stride=(1,1),padding=(1,1)) #b,c,h,w
        b  = F.avg_pool2d(b,3,stride=(1,1),padding=(1,1)) #b,c,h,w
        q=a*images+b
        # print(q.shape)
        return q#b,c,h,w
class ResidualBlock(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

class TripleGuidedFiter(nn.Module):
    def __init__(self,kernel_size):
        super(TripleGuidedFiter, self).__init__()
        self.epsilon = Parameter(torch.Tensor([0.1]))
        self.kernel_size = kernel_size
    def forward(self,images,guided):
        # device = torch.cuda.current_device()
        # self.epsilon = self.epsilon.to(device)
        """
        @param images: b,c,3,h,w

        @return b,c,h,w
        """
        # if images.shape[-1] == guided.shape[-1]:

        #     mean_p = F.avg_pool3d(images,(3,3,3),stride=(1,1,1),padding=(0,1,1)) #b,c,1,h,w
        #     var_p = F.avg_pool3d(images**2,3,stride=(1,1,1),padding=(0,1,1)) - mean_p**2 #b,c,1,h,w
        #     mean_I = F.avg_pool3d(images,(3,3,3),stride=(1,1,1),padding=(0,1,1)) #b,c,1,h,w
        #     var_I = F.avg_pool3d(images**2,3,stride=(1,1,1),padding=(0,1,1)) - mean_I**2 #b,c,1,h,w
        #     mean_Ip = F.avg_pool3d(images*guided,(3,3,3),stride=(1,1,1),padding=(0,1,1)) #b,c,1,h,w
        #     cov_Ip = mean_Ip - mean_I* mean_p

        #     a = cov_Ip/(var_I+self.epsilon**2)
        #     b = mean_p - a*mean_I

        #     a.squeeze_(2)
        #     b.squeeze_(2)
        #     # print(a.shape),b.shape
        #     a = F.avg_pool2d(a,3,stride=(1,1),padding=(1,1)) #b,c,h,w
        #     b  = F.avg_pool2d(b,3,stride=(1,1),padding=(1,1)) #b,c,h,w
        #     q = a * guided[:,:,1] + b

        #     return 3*images[:,:,1] - 2*q
        # else:
        batch_size,channel,view,height,width = images.shape
        scale = images.shape[-1]//guided.shape[-1]
        mean_p = F.avg_pool3d(images,self.kernel_size,stride=(1,scale,scale),padding=(0,self.kernel_size//2,self.kernel_size//2)) #b,c,1,h,w
        var_p = F.avg_pool3d(images**2,self.kernel_size,stride=(1,scale,scale),padding=(0,self.kernel_size//2,self.kernel_size//2)) - mean_p**2 #b,c,1,h,w

        mean_I = F.avg_pool3d(guided,self.kernel_size,stride=(1,1,1),padding=(0,self.kernel_size//2,self.kernel_size//2)) #b,c,1,h,w
        var_I = F.avg_pool3d(guided**2,self.kernel_size,stride=(1,1,1),padding=(0,self.kernel_size//2,self.kernel_size//2)) - mean_I**2 #b,c,1,h,w
        images_strip = images.reshape([batch_size,-1,height,width])
        images_strip = F.interpolate(images_strip,scale_factor = 1.0/scale,align_corners=False,mode='bicubic')
        images_strip = images_strip.view(batch_size,channel,view,height//scale,width//scale)
        # print(images_strip.shape,guided.shape)
        mean_Ip = F.avg_pool3d(images_strip*guided,self.kernel_size,stride=(1,1,1),padding=(0,self.kernel_size//2,self.kernel_size//2)) #b,c,1,h,w
        # print(mean_IG.shape,mean_I.shape,mean_G.shape)
        cov_Ip = mean_Ip - mean_I* mean_p
        # print("q",cov_IG.shape,var_I.shape,images_strip.shape,mean_IG.shape,mean_I.shape,mean_G.shape)
        a = cov_Ip/(var_I+self.epsilon**2)
        b = mean_p - a*mean_I

        a.squeeze_(2)
        b.squeeze_(2)

        a = F.avg_pool2d(a,self.kernel_size,stride=(1,1),padding=(self.kernel_size//2,self.kernel_size//2)) #b,c,h,w
        b  = F.avg_pool2d(b,self.kernel_size,stride=(1,1),padding=(self.kernel_size//2,self.kernel_size//2)) #b,c,h,w
        a = F.interpolate(a,scale_factor = scale,align_corners=False,mode='bicubic')
        b = F.interpolate(b,scale_factor = scale,align_corners=False,mode='bicubic')

        guided = guided.reshape([batch_size,-1,height//scale,width//scale])
        guided = F.interpolate(guided,scale_factor = scale,align_corners=False,mode='bicubic')
        guided = guided.view(batch_size,channel,view,height,width)

        q = a * guided[:,:,images.shape[2]//2] + b

        return 2*images[:,:,images.shape[2]//2]-q #

def downSample():
    return nn.Sequential(
        ResidualBlock(64),
        ResidualBlock(64),
        nn.Conv2d(64,64,3,1,1),
        nn.AvgPool2d(3,2,1),
        nn.LeakyReLU(),
    )
def upSample(in_channel = 128):
    return nn.Sequential(
        nn.Conv2d(in_channel,64,3,1,1),
        ResidualBlock(64),
        nn.Conv2d(64,64*4,3,1,1),
        nn.PixelShuffle(2),
        nn.LeakyReLU(),
        nn.Conv2d(64,64,3,1,1),
        nn.LeakyReLU(),
    )
def offset(in_channel=128):
    return nn.Sequential(
        nn.Conv2d(in_channel,64,3,1,1),
        nn.LeakyReLU(),
        nn.Conv2d(64,64,3,1,1),
        nn.LeakyReLU()

    )
class LapPyrNet(nn.Module):
    def __init__(self,nLevel=3,nUp = 2):
        super(LapPyrNet,self).__init__()
        self.nLevel = nLevel
        self.nUp = nUp
        self.conv_0 = nn.Sequential(
            nn.Conv2d(3,64,3,1,1),
            ResidualBlock(64),
            nn.Conv2d(64,64,3,1,1),
            nn.LeakyReLU()
        )
        self.downs = nn.ModuleList([downSample()]*nLevel)
        ups =  nn.ModuleList(([upSample()]*(nLevel-1) ) +[upSample(64)])
        self.ups = nn.ModuleList( [ups]*nUp)

        self.offsets = nn.ModuleList([offset(128)]*(self.nLevel))

        self.offsetmerge = nn.ModuleList([offset(128)]*(self.nLevel-1))
        # self.L_feat = nn.ModuleList([nn.Sequential(
        #     nn.Conv2d(64,64,3,1,1),
        #     nn.LeakyReLU()
        # )]*self.nLevel)
        self.dcns = nn.ModuleList([DCN(64, 64, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
                              extra_offset_mask=True)]* nLevel)
    def forward(self,images):
        """
        @return detail pyramid
            for example: nLevel = 3,nUp = 2
               return shape :
                    [b,c,h*4,w*4]
                    [b,c,h*2,w*2]
                    [b,c,h,w]
        """

        b,n,c,h,w = images.shape
        images = images.view(-1,c,h,w)

        feat = self.conv_0(images)
        pyrs = [[] for _ in range(self.nLevel)]
        pyrs[0] = feat

        for level in range(1,self.nLevel):

            pyrs[level] = self.downs[level](pyrs[level-1])

        lap_pyrs = [[] for _ in range(self.nLevel)]
        lap_pyrs[self.nLevel -1 ] = pyrs[-1]

        for level in range(self.nLevel-2,-1,-1):
            lap_pyrs[level] = pyrs[level] -  F.interpolate(pyrs[level+1],scale_factor = 2.0,align_corners=False,mode='bilinear')


        # ref_pyrs = [lap[:,n//2] for lap in lap_pyrs]





        # lap unet up
        for up in range(self.nUp):
            lap_pyrs[- 1] = self.ups[up][-1](lap_pyrs[-1])
            for level in range(self.nLevel-2,-1,-1):
                lap_pyrs[level] = self.ups[up][level](torch.cat([lap_pyrs[level],lap_pyrs[level+1]],1))

        # pcd alignment
        lap_pyrs = [lap.view(b,n,-1,lap.shape[-2],lap.shape[-1])  for lap in lap_pyrs]

        for view in range(n):
            if view == n//2:
                continue
            prev_offset =self.offsets[-1](torch.cat([lap_pyrs[-1][:,n//2],lap_pyrs[-1][:,view]],1))
            lap_pyrs[-1][:,view] = self.dcns[-1]([lap_pyrs[-1][:,view].contiguous(),prev_offset])

            for level in range(self.nLevel-2,-1,-1):

                cent_feat = lap_pyrs[level][:,n//2]

                level_feat = lap_pyrs[level][:,view]

                offset = self.offsets[level](torch.cat([cent_feat,level_feat],1))

                prev_offset = F.interpolate(prev_offset,scale_factor = 2.0,align_corners=False,mode='bilinear')

                offset = self.offsetmerge[level](torch.cat([offset,prev_offset],1))

                lap_pyrs[level][:,view] = self.dcns[level]([lap_pyrs[level][:,view].contiguous(),prev_offset])

                prev_offset = offset
        lap_pyrs = [lap.view(b*n,-1,lap.shape[-2],lap.shape[-1])  for lap in lap_pyrs]

        # print(len(lap_pyrs))
        return lap_pyrs

def GateNet():
    return nn.Sequential(
        nn.Conv2d(128,64,3,1,1),
        ResidualBlock(64),
        nn.Conv2d(64,1,3,1,1),
        nn.Sigmoid()
    )
# def fusion(in_channel=128):
#     return nn.Sequential(
#         nn.Conv2d(in_channel,64,3,1,1),
#         ResidualBlock(64),
#         nn.Conv2d(64,64,3,1,1),
#         nn.LeakyReLU()
#     )
def last_conv():
    return nn.Conv2d(128,1,3,1,1)
class PyrFusionNet(nn.Module):
    def __init__(self,view_num=3,nlevel=3):
        super(PyrFusionNet,self).__init__()
        self.gateNet = GateNet()
        self.view_num = view_num
        self.nlevel = nlevel
        self.merge = nn.ModuleList([last_conv()] +([upSample(128)]*(nlevel-2)) + [upSample(64)])
    def forward(self,pyrFeat):
        """
        pyrFeat nlevel feat
        """
        device = torch.cuda.current_device()

        feats = []
        for level in range(self.nlevel):
            b,c,h,w = pyrFeat[level].shape
            pyrFeat[level] = pyrFeat[level].view(-1,self.view_num,c,h,w)
            weights = torch.zeros(b//self.view_num,c,h,w).to(device)
            feat = torch.zeros(b//self.view_num,c,h,w).to(device)
            for view in range(self.view_num):
                weight =self.gateNet(torch.cat([pyrFeat[level][:,view],pyrFeat[level][:,self.view_num//2]],1))
                feat += weight*pyrFeat[level][:,view]
                # weights =weights + weight
            # feat /=weights
            feats.append(feat)
        prev = self.merge[-1](feats[-1])
        for level in range(self.nlevel-2,-1,-1):
            prev = self.merge[level](torch.cat([feats[level],prev],1))
        return prev

class SRNet(nn.Module):
    def __init__(self):
        super(SRNet,self).__init__()
        self.LapPyrNet = LapPyrNet()
        self.PyrFusionNet = PyrFusionNet()

    def forward(self,images):
        """
        @param: images  b,v,c,h,w
        """
        b,v,c,h,w = images.shape
        image = images[:,v//2]
        images = images.view(-1,c,h,w)
        pyrfeat = self.LapPyrNet(images)
        detail = self.PyrFusionNet(pyrfeat)
        base = F.interpolate(image,scale_factor = 4.0,align_corners=False,mode='bilinear')
        return base + detail
