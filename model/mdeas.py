import torch
import torch.nn as nn
import torch.nn.functional as F
from model.buildingblock import ImageConv, PointwiseConv, PixelShuffleConv, DepthwiseConv, NearestNeighborConv
# from Decoder import MaskDecoderBlock, DepthDecoderBlock

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.stride = stride

        self.basicconv1 = DepthwiseConv(in_planes, in_planes)
        self.basicconv2 = DepthwiseConv(in_planes, in_planes)
        self.basicconv3 = DepthwiseConv(in_planes, in_planes)
        self.pointconv = PointwiseConv(in_planes, planes)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):

        out1 = self.basicconv1(x)
        out2 = self.basicconv2(x+out1)
        out3 = self.basicconv3(x+out1+out2)
        out = self.pointconv(x+out1+out2+out3)

        if (self.stride == 2):
            out  = self.pool(out)

        return out

class EncoderBlock(nn.Module):

    def __init__(self, planes, stride=2):
        super(EncoderBlock, self).__init__()
        self.stride = stride

        self.encodeconv1 = BasicBlock(planes[0], planes[1],stride)
        self.encodeconv2 = BasicBlock(planes[1], planes[2],stride)
        self.encodeconv3 = BasicBlock(planes[2], planes[2],stride)

    def forward(self, x):

        out1 = self.encodeconv1(x)
        out2 = self.encodeconv2(out1)
        out3 = self.encodeconv3(out2)

        return out1, out2, out3

class DepthDecoderBlock(nn.Module):

    def __init__(self, planes):
        super(DepthDecoderBlock, self).__init__()

        self.depthdecodeconv1 = NearestNeighborConv(planes[0], planes[0])
        self.depthdecodeconv2 = NearestNeighborConv(planes[0], planes[1])
        self.depthdecodeconv3 = PixelShuffleConv(planes[1], planes[2])
        self.depthdecodeconv4 = PixelShuffleConv(planes[2], planes[3])
        self.pointwiseconv = nn.Conv2d(planes[3], 1, kernel_size=1)

    def forward(self, dilation_depth_out , encdr_out1, encdr_out2, image_conv_out  ):

        decode_out1 = self.depthdecodeconv1(dilation_depth_out)
        decode_out1 += encdr_out2
        decode_out2 = self.depthdecodeconv2(decode_out1)
        decode_out2 += encdr_out1
        decode_out3 = self.depthdecodeconv3(decode_out2)
        decode_out3 += image_conv_out
        decode_out4 = self.depthdecodeconv4(decode_out3)
        out = self.pointwiseconv(decode_out4)

        return out

class MaskDecoderBlock(nn.Module):

    def __init__(self, planes):
        super(MaskDecoderBlock, self).__init__()

        self.maskdecodeconv1 = NearestNeighborConv(planes[0], planes[0])
        self.maskdecodeconv2 = NearestNeighborConv(planes[0], planes[0])
        self.maskdecodeconv3 = NearestNeighborConv(planes[0], planes[1])
        self.maskdecodeconv4 = PixelShuffleConv(planes[1], planes[2])
        self.pointwiseconv = nn.Conv2d(planes[2], 1, kernel_size=1)

    def forward(self, dilation_mask_out , encdr_out1, image_conv_out  ):

        decode_out1 = self.maskdecodeconv1(dilation_mask_out)
        decode_out2 = self.maskdecodeconv2(decode_out1)
        decode_out2 += encdr_out1
        decode_out3 = self.maskdecodeconv3(decode_out2)
        decode_out3 += image_conv_out
        decode_out4 = self.maskdecodeconv4(decode_out3)
        out = self.pointwiseconv(decode_out4)

        out = torch.sigmoid(out)

        return out


class DilationBlock(nn.Module):


    def __init__(self, in_planes, planes,mask_planes,depth_planes, stride=1):
        super(DilationBlock, self).__init__()
        self.stride = stride
        #number of dilation layers
        expansion = 3
        self.pointconv = PointwiseConv(in_planes, planes)

        self.dilationconv1 = DepthwiseConv(planes, planes, dilation=3)
        self.dilationconv2 = DepthwiseConv(planes, planes, dilation=7)
        self.dilationconv3 = DepthwiseConv(planes, planes, dilation=11)

        self.pointconv_mask = PointwiseConv(planes * expansion, mask_planes)
        self.pointconv_depth = PointwiseConv(planes * expansion, depth_planes)

    def forward(self, x):

        #squeeze the channels
        squeeze = self.pointconv(x)

        dilatedout1 = self.dilationconv1(squeeze)
        dilatedout2 = self.dilationconv2(squeeze)
        dilatedout3 = self.dilationconv3(squeeze)

        concatout = torch.cat((dilatedout1,dilatedout2,dilatedout3),1)

        out_mask = self.pointconv_mask(concatout)
        out_depth = self.pointconv_depth(concatout)

        return out_mask , out_depth


#############################################MAIN MODEL##################################################
class MDEASModel(nn.Module):
    def __init__(self):
        super(MDEASModel, self).__init__()
        #################### Initial Block #################
        self.bgimageconv  = ImageConv( 3, 32)
        self.overlayimageconv  = ImageConv( 3, 32)

        ################## Encoder Blocks ##################

        self.encoder = EncoderBlock([64,128,256])


        ################# Bottle-Neck Block #################
        self.dilation   = DilationBlock(256, 128, 128 , 256)

        ################## Depth Decoder ####################
        self.depthdecoder = DepthDecoderBlock([256, 128, 64, 32])

        ##################### Mask Decoder ##################
        self.maskdecoder  = MaskDecoderBlock([128, 64, 32])


    def forward(self, bg, fg_bg):
        ### Initial Block
        bg_out = self.bgimageconv(bg)
        overlay_out = self.overlayimageconv(fg_bg)
        imagesout   = torch.cat([bg_out, overlay_out], 1)

        ### Encoder Block
        encdr_out1, encdr_out2, encdr_out3 = self.encoder(imagesout)

        ### Bottleneck Block --(Dilation block)
        dilation_mask_out, dilation_depth_out   = self.dilation(encdr_out3)
        ### Depth Decoder Block
        depth_out  = self.depthdecoder( dilation_depth_out, encdr_out1, encdr_out2, imagesout)

        ### Mask Decoder Block
        mask_out   = self.maskdecoder( dilation_mask_out, encdr_out1, imagesout)

        return depth_out, mask_out

###############################################################################################################






#########################nitialize kernel weights with Gaussian distributions##################################
import math
def weights_init(m):

    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
################################################################################################################