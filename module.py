import torch
import torch.nn as nn
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
            nn.ReLU()
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
            nn.ReLU()
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
        nn.ReLU()
    )


class TripleGuidedFiter(nn.Module):
    def __init__(self,kernel_size,epsilon):
        super(TripleGuidedFiter, self).__init__()
        self.epsilon = epsilon#torch.autograd.Variable(torch.Tensor([1.0]),requires_grad = True)
        self.kernel_size = kernel_size
    def forward(self,images):
        device = torch.cuda.current_device()
        # self.epsilon = self.epsilon.to(device)
        """
        @param images: b,c,3,h,w

        @return b,c,h,w
        """

        mean = F.avg_pool3d(images,self.kernel_size,stride=(1,1,1),padding=(0,self.kernel_size[1]//2,self.kernel_size[1]//2)) #b,c,1,h,w
        var = F.avg_pool3d(images**2,3,stride=(1,1,1),padding=(0,self.kernel_size[1]//2,self.kernel_size[1]//2)) - mean**2 #b,c,1,h,w

        mean.squeeze_(2)
        var.squeeze_(2)
        guids = []
        for ep in self.epsilon:
            a = var/(var+ep**2)
            b = (1-a)*mean

            a = F.avg_pool2d(a,3,stride=(1,1),padding=(1,1)) #b,c,h,w
            b  = F.avg_pool2d(b,3,stride=(1,1),padding=(1,1)) #b,c,h,w
            guids.append(a*images[:,:,1]+b)
        return 2*images[:,:,1]-torch.cat(guids,1)#b,c,h,w
