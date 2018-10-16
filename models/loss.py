import torch
import numpy as np
import torch.nn.functional as F
from .cx_loss import CX_loss, symetric_CX_loss


class CXReconLoss(torch.nn.Module):

    """
    contexutal loss with vgg network
    """

    def __init__(self, feat_extractor, device=None, weight=1):
        super(CXReconLoss, self).__init__()
        self.feat_extractor = feat_extractor
        self.device = device
        if device is not None:
            self.feat_extractor = self.feat_extractor.to(device)
        #self.feat_extractor = self.feat_extractor.cuda()
        self.weight = weight

    def forward(self, imgs, recon_imgs):
        if self.device is not None:
            imgs = imgs.to(self.device)
            recon_imgs = recon_imgs.to(self.device)
        imgs = F.interpolate(imgs, (224,224))
        recon_imgs = F.interpolate(recon_imgs, (224,224))
        ori_feats, _ = self.feat_extractor(imgs)
        recon_feats, _ = self.feat_extractor(recon_imgs)
        return self.weight * symetric_CX_loss(ori_feats, recon_feats)


class MaskDisLoss(torch.nn.Module):
    """
    The loss for mask discriminator
    """
    def __init__(self, weight=1):
        super(MaskDisLoss, self).__init__()
        self.weight = weight
        self.leakyrelu = torch.nn.LeakyReLU()
    def forward(self, pos, neg):
        #return self.weight * (torch.sum(F.relu(-1+pos)) + torch.sum(F.relu(-1-neg)))/pos.size(0)
        return self.weight * (torch.mean(self.leakyrelu(1.-pos)) + torch.mean(self.leakyrelu(1.+neg)))


class SNDisLoss(torch.nn.Module):
    """
    The loss for sngan discriminator
    """
    def __init__(self, weight=1):
        super(SNDisLoss, self).__init__()
        self.weight = weight

    def forward(self, pos, neg):
        #return self.weight * (torch.sum(F.relu(-1+pos)) + torch.sum(F.relu(-1-neg)))/pos.size(0)
        return self.weight * (torch.mean(F.relu(1.-pos)) + torch.mean(F.relu(1.+neg)))


class SNGenLoss(torch.nn.Module):
    """
    The loss for sngan generator
    """
    def __init__(self, weight=1):
        super(SNGenLoss, self).__init__()
        self.weight = weight

    def forward(self, neg):
        return - self.weight * torch.mean(neg)

class L1ReconLoss(torch.nn.Module):
    """
    L1 Reconstruction loss for two imgae
    """
    def __init__(self, weight=1):
        super(L1ReconLoss, self).__init__()
        self.weight = weight

    def forward(self, imgs, recon_imgs, masks=None):
        if masks is None:
            return self.weight * torch.mean(torch.abs(imgs - recon_imgs))
        else:
            #print(masks.view(masks.size(0), -1).mean(1).size(), imgs.size())
            return self.weight * torch.mean(torch.abs(imgs - recon_imgs) / masks.view(masks.size(0), -1).mean(1).view(-1,1,1,1))

class PerceptualLoss(torch.nn.Module):
    """
    Use vgg or inception for perceptual loss, compute the feature distance, (todo)
    """
    def __init__(self, weight=1):
        super(PerceptualLoss, self).__init__()
        self.weight = weight

    def forward(self, img, recon_imgs):
        pass

class ReconLoss(torch.nn.Module):
    """
    Reconstruction loss contain l1 loss, may contain perceptual loss

    """
    def __init__(self, chole_alpha, cunhole_alpha, rhole_alpha, runhole_alpha):
        super(ReconLoss, self).__init__()
        self.chole_alpha = chole_alpha
        self.cunhole_alpha = cunhole_alpha
        self.rhole_alpha = rhole_alpha
        self.runhole_alpha = runhole_alpha

    def forward(self, imgs, coarse_imgs, recon_imgs, masks):
        masks_viewed = masks.view(masks.size(0), -1)
        return self.rhole_alpha*torch.mean(torch.abs(imgs - recon_imgs) * masks / masks_viewed.mean(1).view(-1,1,1,1))  + \
                self.runhole_alpha*torch.mean(torch.abs(imgs - recon_imgs) * (1. - masks) / (1. - masks_viewed.mean(1).view(-1,1,1,1)))  + \
                self.chole_alpha*torch.mean(torch.abs(imgs - coarse_imgs) * masks / masks_viewed.mean(1).view(-1,1,1,1))   + \
                self.cunhole_alpha*torch.mean(torch.abs(imgs - coarse_imgs) * (1. - masks) / (1. - masks_viewed.mean(1).view(-1,1,1,1)))
