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