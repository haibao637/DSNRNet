import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from  dcn.deform_conv import ModulatedDeformConvPack as DCN
from cbamGRU import CCBGRU
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


class ResidualBlock3D(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv3d(nf, nf, 3, 1, 1, bias=True)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

def downSample(nf=64):
    return nn.Sequential(
        ResidualBlock(nf),
        ResidualBlock(nf),
        nn.Conv2d(nf,nf,3,1,1),
        nn.MaxPool2d(3,2,1),
        nn.LeakyReLU(inplace=False),
    )
def upSample(in_channel = 128,nf=64):
    return nn.Sequential(
        nn.Conv2d(in_channel,nf,3,1,1),
        nn.LeakyReLU(),
        *([ResidualBlock(nf)]*2),
        nn.Conv2d(nf,nf*4,3,1,1),
        nn.PixelShuffle(2),
        nn.LeakyReLU(),
        nn.Conv2d(nf,nf,3,1,1),
        nn.LeakyReLU(inplace=False),
    )
def offset(in_channel=128,nf=64):
    return nn.Sequential(
        nn.Conv2d(in_channel,nf,3,1,1),
        nn.LeakyReLU(),
        *([ResidualBlock(nf)]*2),
        nn.Conv2d(nf,nf,3,1,1),
        nn.LeakyReLU()

    )
# class LapPyrNet(nn.Module):
#     def __init__(self,nLevel=3,nf=64):
#         super(LapPyrNet,self).__init__()
#         self.nLevel = nLevel
#         # self.nUp = nUp
#         self.conv_0 = nn.Sequential(
#             nn.Conv2d(3,nf,3,1,1),
#             ResidualBlock(nf),
#             nn.Conv2d(nf,nf,3,1,1),
#             nn.LeakyReLU()
#         )
#         self.downs = nn.ModuleList([downSample()]*(nLevel))
#         # ups =  nn.ModuleList(([upSample()]*(nLevel-1) ) +[upSample(64)])
#         # self.ups = nn.ModuleList( [ups]*nUp)

#         self.offsets = nn.ModuleList([offset(nf*2)]*(self.nLevel))

#         self.offsetmerge = nn.ModuleList([offset(nf*2)]*(self.nLevel-1))
#         # self.L_feat = nn.ModuleList([nn.Sequential(
#         #     nn.Conv2d(64,64,3,1,1),
#         #     nn.LeakyReLU()
#         # )]*self.nLevel)
#         self.dcns = nn.ModuleList([DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
#                               extra_offset_mask=True)]* nLevel)
#     def forward(self,images):
#         """
#         @return detail pyramid
#             for example: nLevel = 3,nUp = 2
#                return shape :
#                     [b,c,h*4,w*4]
#                     [b,c,h*2,w*2]
#                     [b,c,h,w]
#         """

#         b,n,c,h,w = images.shape
#         images = images.view(-1,c,h,w)

#         feat = self.conv_0(images)
#         pyrs = [[] for _ in range(self.nLevel)]
#         pyrs[0] = feat

#         for level in range(1,self.nLevel):
#             pyrs[level] = self.downs[level](pyrs[level-1])



#          # lap unet up
#         # for up in range(self.nUp):
#         #     pyrs[- 1] = self.ups[up][-1](pyrs[-1])
#         #     for level in range(self.nLevel-2,-1,-1):
#         #         # print(pyrs[level].shape,lap_pyrs[level+1].shape)
#         #         pyrs[level] = self.ups[up][level](torch.cat([pyrs[level],pyrs[level+1]],1))


#         # lap_pyrs = [[] for _ in range(self.nLevel)]
#         # lap_pyrs[self.nLevel -1 ] = pyrs[-1]
#         # for level in range(self.nLevel-2,-1,-1):
#         #     lap_pyrs[level] = pyrs[level] -  F.interpolate(pyrs[level+1],scale_factor = 2.0,align_corners=False,mode='bilinear')


#         # ref_pyrs = [lap[:,n//2] for lap in lap_pyrs]



#         # pcd alignment
#         lap_pyrs = [lap.clone().reshape(b,n,-1,lap.shape[-2],lap.shape[-1])  for lap in pyrs]

#         for view in range(n):
#             # if view == n//2:
#             #     continue
#             prev_offset =self.offsets[-1](torch.cat([lap_pyrs[-1][:,n//2],lap_pyrs[-1][:,view]],1))
#             lap_pyrs[-1][:,view] = self.dcns[-1]([lap_pyrs[-1][:,view].contiguous(),prev_offset])

#             for level in range(self.nLevel-2,-1,-1):
#                 cent_feat = lap_pyrs[level][:,n//2]
#                 level_feat = lap_pyrs[level][:,view]
#                 offset = self.offsets[level](torch.cat([cent_feat,level_feat],1))
#                 prev_offset = F.interpolate(prev_offset,scale_factor = 2.0,align_corners=False,mode='bilinear')
#                 # prev_offfset = prev_offset*2
#                 offset = self.offsetmerge[level](torch.cat([offset,prev_offset*2.0],1))
#                 lap_pyrs[level][:,view] = self.dcns[level]([lap_pyrs[level][:,view].contiguous(),offset])
#                 prev_offset = offset

#         lap_pyrs = [lap.view(b,n,-1,lap.shape[-2],lap.shape[-1])  for lap in lap_pyrs]

#         return lap_pyrs

class LapPyrNet(nn.Module):
    def __init__(self,nLevel=3,nf=64):
        super(LapPyrNet,self).__init__()
        self.nLevel = nLevel
        # self.nUp = nUp
        self.conv_0 = nn.Sequential(
            nn.Conv2d(3,nf,3,1,1),
            ResidualBlock(nf),
            nn.Conv2d(nf,nf,3,1,1),
            nn.LeakyReLU()
        )
        self.downs = nn.ModuleList([downSample()]*(nLevel))
        # ups =  nn.ModuleList(([upSample()]*(nLevel-1) ) +[upSample(64)])
        # self.ups = nn.ModuleList( [ups]*nUp)

        self.offsets = nn.ModuleList([offset(nf*2)]*(self.nLevel))

        self.offsetmerge = nn.ModuleList([offset(nf*2)]*(self.nLevel-1))
        # self.L_feat = nn.ModuleList([nn.Sequential(
        #     nn.Conv2d(64,64,3,1,1),
        #     nn.LeakyReLU()
        # )]*self.nLevel)
        self.dcns = nn.ModuleList([DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
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

        # pcd alignment
        lap_pyrs = pyrs#[lap.clone().reshape(b*n,lap.shape[-3],lap.shape[-2],lap.shape[-1])  for lap in pyrs]
        cent = [lap[n//2:b*n:n].unsqueeze(1).repeat(1,n,1,1,1).reshape(n*b,lap.shape[-3],lap.shape[-2],lap.shape[-1]) for lap in pyrs]
        # for view in range(n):
            # if view == n//2:
            #     continue
#         print(cent[-1].shape,lap_pyrs[-1].shape)
        prev_offset =self.offsets[-1](torch.cat([cent[-1],lap_pyrs[-1]],1))
        lap_pyrs[-1] = self.dcns[-1]([lap_pyrs[-1].contiguous(),prev_offset])

        for level in range(self.nLevel-2,-1,-1):
            cent_feat = cent[level]
            level_feat = lap_pyrs[level]
            offset = self.offsets[level](torch.cat([cent_feat,level_feat],1))
            prev_offset = F.interpolate(prev_offset,scale_factor = 2.0,align_corners=False,mode='bilinear')
            offset = self.offsetmerge[level](torch.cat([offset,prev_offset*2.0],1))
            lap_pyrs[level] = self.dcns[level]([lap_pyrs[level].contiguous(),offset])
            prev_offset = offset

        lap_pyrs = [lap.view(b,n,-1,lap.shape[-2],lap.shape[-1])  for lap in lap_pyrs]
        return lap_pyrs

def CentNet(channel=128,nf=64):
    return nn.Sequential(
        nn.Conv2d(channel,nf,3,1,1),
        nn.LeakyReLU(),
        *([ResidualBlock(nf)]*2),
        nn.Conv2d(nf,1,3,1,1),
        nn.Sigmoid()
    )
# def fusion(in_channel=128):
#     return nn.Sequential(
#         nn.Conv2d(in_channel,64,3,1,1),
#         ResidualBlock(64),
#         nn.Conv2d(64,64,3,1,1),
#         nn.LeakyReLU()
#     )
# def GateNet3D(nf=64):
#     return nn.Sequential(
#         nn.Conv3d(nf,nf,3,1,1),
#         nn.LeakyReLU(),
#          *([ResidualBlock3D(nf)]*2),
#         nn.Conv3d(nf,32,3,1,1),
#         nn.LeakyReLU(),
#         nn.Conv3d(32,1,3,1,1),
#         nn.Softmax(dim=2)
#     )



class SpatialAttention3d(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention3d, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv3d(2,1,kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class ChannelAttention3d(nn.Module):
    def __init__(self, in_planes, rotio=16):
        super(ChannelAttention3d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.sharedMLP = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // rotio, 1, bias=False), nn.ReLU(),
            nn.Conv3d(in_planes // rotio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class CBAM3D(nn.Module):
    def __init__(self, planes):
        super(CBAM3D,self).__init__()
        self.ca = ChannelAttention3d(planes)
        self.sa = SpatialAttention3d()
    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

def GateNet3D(in_channel = 64,nf=64):
    return nn.Sequential(
        nn.Conv3d(in_channel,nf,3,1,1),
        *([ResidualBlock3D(nf)] * 2),
        nn.LeakyReLU(),
        nn.Conv3d(nf,nf,3,1,1),
        nn.LeakyReLU(),
        CBAM3D(64)
    )

class PyrFusionNet(nn.Module):
    def __init__(self,nlevel=3,nf=64):
        super(PyrFusionNet,self).__init__()
        # self.centNets =nn.ModuleList(([CentNet()]*nlevel))
        # self.merge = nn.ModuleList([last_conv()] +([upSample(128)]*(nlevel-2)) + [upSample(64)])
        self.gateNets =  nn.ModuleList(([CCBGRU(input_dim=64, hidden_dim=64, \
            kernel_size=(3,3), num_layers=1, batch_first=True)]*nlevel))
        self.merges = nn.ModuleList(([nn.Sequential(nn.Conv2d(nf*3,nf,3,1,1),nn.LeakyReLU())]*nlevel))
    def forward(self,pyrFeat):
        """
        pyrFeat nlevel feat ,each feat view: b,v,-1,h,w
        """
        device = torch.cuda.current_device()

        feats = []

        for level in range(len(pyrFeat)):
            b,n,c,h,w = pyrFeat[level].shape
            pyrFeat[level]  = pyrFeat[level]
            front = pyrFeat[level][:,:n//2+1]
            back = torch.flip(pyrFeat[level][:,n//2:],[1])
            feat,_ = self.gateNets[level](front)
            rfeat,_ = self.gateNets[level](back)
            # print(feat.shape,rfeat.shape,pyrFeat[level][:,n//2].shape)
            feat = self.merges[level](torch.cat([feat.squeeze(1),rfeat.squeeze(1),pyrFeat[level][:,n//2]],1))

            feats.append(feat)
        return feats # [b,64,h,w]*level

class ReconNet(nn.Module):
    def __init__(self,nlevel=3,nUp=2,nf=64):
        super(ReconNet,self).__init__()
        # self.merge = nn.ModuleList([last_conv()] +([upSample(128)]*(nlevel-2)) + [upSample(64)])
        self.ups =  nn.ModuleList(([upSample()]*(nlevel-1) ) +[upSample(64)])
        # self.ups = nn.ModuleList( [ups]*nUp)
        self.nUp = nUp
        self.nLevel = nlevel
        self.HRCon = nn.Sequential(
            upSample(64),
            nn.Conv2d(nf,3,3,1,1)
        )
    def forward(self,feats):
        # up
        # for up in range(self.nUp):
        feats[- 1] = self.ups[-1](feats[-1])
        for level in range(self.nLevel-2,-1,-1):
            # print(pyrs[level].shape,lap_pyrs[level+1].shape)
            feats[level] = self.ups[level](torch.cat([feats[level],feats[level+1]],1))
        return self.HRCon(feats[0])

