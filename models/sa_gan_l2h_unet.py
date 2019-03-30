import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from .spectral import SpectralNorm
from .networks import GatedConv2dWithActivation, GatedDeConv2dWithActivation, SNConvWithActivation, get_pad

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation,with_attn=False):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.with_attn = with_attn
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        if self.with_attn:
            return out ,attention
        else:
            return out

class InpaintRUNNet(torch.nn.Module):
    """
    Inpaint generator, input should be 5*256*256, where 3*256*256 is the masked image, 1*256*256 for mask, 1*256*256 is the guidence
    """
    def __init__(self, device, n_in_channel=7, ):
        super(InpaintRUNNet, self).__init__()
        cnum = 32
        self.cnum = cnum
        self.device = device
        self.input_block = nn.Sequential(
            GatedConv2dWithActivation(n_in_channel, cnum, 7, 1, dilation=1, padding=get_pad(256, 7, 1, 1)),
            GatedConv2dWithActivation(cnum, cnum, 5, 1, dilation=1, padding=get_pad(256, 5, 1, 1)))

        self.downsample_layer1 = GatedConv2dWithActivation(cnum, cnum, 4, 2, padding=get_pad(256, 4, 2))
        self.eblock1 = nn.Sequential(
                            GatedConv2dWithActivation(cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
                            GatedConv2dWithActivation(2*cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)))

        self.downsample_layer2 = GatedConv2dWithActivation(2*cnum, 2*cnum, 4, 2, padding=get_pad(128, 4, 2))
        self.eblock2 = nn.Sequential(
            GatedConv2dWithActivation(2*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),)

        self.downsample_layer3 = GatedConv2dWithActivation(4*cnum, 4*cnum, 4, 2, padding=get_pad(64, 4, 2))
        self.eblock3 = nn.Sequential(
            GatedConv2dWithActivation(4*cnum, 8*cnum, 3, 1, padding=get_pad(32, 3, 1)),
            GatedConv2dWithActivation(8*cnum, 8*cnum, 3, 1, padding=get_pad(32, 3, 1)),)

        self.downsample_layer4 = GatedConv2dWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(32, 4, 2))
        self.eblock4 = nn.Sequential(
            GatedConv2dWithActivation(8*cnum, 8*cnum, 3, 1, dilation=2, padding=get_pad(16, 3, 1, 2)),
            GatedConv2dWithActivation(8*cnum, 8*cnum, 3, 1, dilation=4, padding=get_pad(16, 3, 1, 4)),)

        self.dblock4 = nn.Sequential(
            GatedConv2dWithActivation(8*cnum, 16*cnum, 3, 1, dilation=8, padding=get_pad(16, 3, 1, 8)),
            GatedConv2dWithActivation(16*cnum, 16*cnum, 3, 1, dilation=16, padding=get_pad(16, 3, 1, 16)),)

        self.upsample_layer4 = GatedDeConv2dWithActivation(2, 16*cnum, 8*cnum, 3, 1, padding=get_pad(32, 3, 1))
        #concate 8*cnum-> 16*cnum
        self.dblock3 = nn.Sequential(
            GatedConv2dWithActivation(16*cnum, 8*cnum, 3, 1, padding=get_pad(32, 3, 1)),
            GatedConv2dWithActivation(8*cnum, 8*cnum, 3, 1, padding=get_pad(32, 3, 1)),)

        self.upsample_layer3 = GatedDeConv2dWithActivation(2, 8*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1))
        #concate 4*cnum-> 8*cnum
        self.dblock2 = nn.Sequential(
            GatedConv2dWithActivation(8*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4*cnum, 4*cnum, 3, 1, padding=get_pad(64, 3, 1)),)

        self.upsample_layer2 = GatedDeConv2dWithActivation(2, 4*cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1))
        #concate 2*cnum-> 4*cnum
        self.dblock1 = nn.Sequential(
            GatedConv2dWithActivation(4*cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),
            GatedConv2dWithActivation(2*cnum, 2*cnum, 3, 1, padding=get_pad(128, 3, 1)),)

        self.upsample_layer1 = GatedDeConv2dWithActivation(2, 2*cnum, cnum, 3, 1, padding=get_pad(256, 3, 1))
        #concate 1*cnum-> 2*cnum
        self.output_block = nn.Sequential(
            GatedConv2dWithActivation(2*cnum, cnum, 3, 1, padding=get_pad(256, 3, 1)),
            GatedConv2dWithActivation(cnum, cnum//2, 3, 1, padding=get_pad(256, 3, 1)),
            GatedConv2dWithActivation(cnum//2, 3, 3, 1, padding=get_pad(256, 3, 1), activation=None)
        )

        self.attn = Self_Attn(8*cnum, 'relu')


    def forward(self, imgs, masks, pre_imgs, pre_inter_imgs, size):
        # Coarse
        masked_imgs =  imgs * (1 - masks) + masks*torch.Tensor([2*117/255.0-1.0,2*104/255.0-1.0,2*123/255.0-1.0]).view(1,3,1,1).to(self.device)
        pre_inter_imgs = imgs * (1 - masks) + masks * pre_inter_imgs
        input_imgs = torch.cat([masked_imgs, pre_inter_imgs, masks], dim=1)
        #input_imgs = torch.cat([masked_imgs, pre_inter_imgs, masks], dim=1)
        #print(input_imgs.size())
        x = self.input_block(input_imgs)
        ex0 = x
        x = self.downsample_layer1(x)
        x = self.eblock1(x)
        ex1 = x
        x = self.downsample_layer2(x)
        x = self.eblock2(x)
        ex2 = x
        x = self.downsample_layer3(x)
        x = self.eblock3(x)
        ex3 = x
        x = self.downsample_layer4(x)
        x = self.eblock4(x)
        x = self.attn(x)
        x = self.dblock4(x)
        x = self.upsample_layer4(x)
        x = torch.cat([x, ex3], dim=1)
        x = self.dblock3(x)
        x = self.upsample_layer3(x)
        x = torch.cat([x, ex2], dim=1)
        x = self.dblock2(x)
        x = self.upsample_layer2(x)
        x = torch.cat([x, ex1], dim=1)
        x = self.dblock1(x)
        x = self.upsample_layer1(x)
        x = torch.cat([x, ex0], dim=1)
        x = self.output_block(x)
        #concate 2cnum
        # if size == (256, 256) or size == (128,128):
        #     #print(pre_imgs.size(), x.size())
        #     pre_feat = self.conv_feat(pre_imgs)
        #     #print(pre_feat.size())
        #     x = torch.cat([x, pre_feat], dim=1)
        # else:
        #     x = torch.cat([x, ux1], dim=1)
        # x = self.upsample_layer2(x)
        # x = self.output_block(x)
        x = torch.clamp(x, -1., 1.)
        return x

class InpaintSADirciminator(nn.Module):
    def __init__(self):
        super(InpaintSADirciminator, self).__init__()
        cnum = 32
        self.discriminator_net = nn.Sequential(
            SNConvWithActivation(5, 2*cnum, 4, 2, padding=get_pad(256, 5, 2)),
            SNConvWithActivation(2*cnum, 4*cnum, 4, 2, padding=get_pad(128, 5, 2)),
            SNConvWithActivation(4*cnum, 8*cnum, 4, 2, padding=get_pad(64, 5, 2)),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(32, 5, 2)),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(16, 5, 2)),
            SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(8, 5, 2)),
            Self_Attn(8*cnum, 'relu'),
            #SNConvWithActivation(8*cnum, 8*cnum, 4, 2, padding=get_pad(16, 5, 2)),
        )
        self.linear = nn.Linear(8*cnum*2*2, 1)

    def forward(self, input):
        x = self.discriminator_net(input)
        x = x.view((x.size(0),-1))
        #x = self.linear(x)
        return x


class SADiscriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, batch_size=64, image_size=64, conv_dim=64):
        super(Discriminator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv2d(3, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        if self.imsize == 64:
            layer4 = []
            layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim*2
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.Conv2d(curr_dim, 1, 4))
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(256, 'relu')
        self.attn2 = Self_Attn(512, 'relu')

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out,p1 = self.attn1(out)
        out=self.l4(out)
        out,p2 = self.attn2(out)
        out=self.last(out)

        return out.squeeze(), p1, p2
