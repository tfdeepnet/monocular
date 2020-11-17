
import torch
import torch.nn as nn
import torch.nn.functional as F

def ImageConv(in_planes, planes, stride=2):

    return (
            DepthwiseConv(in_planes, planes),
            DepthwiseConv(planes, planes),
            nn.MaxPool2d(2, 2))



'''
class DepthwiseConv(nn.Module):
    def __init__(self, in_planes, planes, dilation=1):
        super(DepthwiseConv, self).__init__()
        self.depthwise = nn.Conv2d(in_planes, in_planes, kernel_size=3,
                                   padding=dilation, groups=in_planes ,
                                   dilation=dilation , bias=False)
        self.pointwise = nn.Conv2d(in_planes, planes, kernel_size=1)
        self.bn =   nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu (self.pointwise(self.depthwise(x)))
        return out
'''

def DepthwiseConv(in_planes, planes, dilation=1):
        return nn.sequential (
            nn.Conv2d(in_planes, in_planes, kernel_size=3,
                                   padding=dilation, groups=in_planes ,
                                   dilation=dilation , bias=False),
            nn.Conv2d(in_planes, planes, kernel_size=1) ,
            nn.BatchNorm2d(planes)
        )

def PointwiseConv(in_planes, planes):
        return nn.sequential (
            nn.Conv2d(in_planes, planes, kernel_size=1) ,
            nn.BatchNorm2d(planes)
        )



def NearestNeighborConv(in_planes, planes):
        return (
            nn.functional.interpolate(x, scale_factor=2, mode="nearest"),
            BasicBlock(in_planes, planes)
        )


def PixelShuffleConv(in_planes, planes):

    return (
        BasicBlock(in_planes, planes),
        nn.PixelShuffle(2)
    )

