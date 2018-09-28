from .inception_score.inception_score import inception_score
from .fid.fid import calculate_fid_given_paths
from .ssim.ssim import ssim
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
"""
def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
def calculate_fid_given_paths(paths, batch_size, cuda, dims):
def ssim(img1, img2, window_size = 11, size_average = True):
"""

_transforms_fun=transforms.Compose([transforms.Resize((299,299)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
def _inception_score(path, cuda=True, batch_size=1, resize=False, splits=1):
    imgs = []
    for file in os.listdir(path):
        if file.endswith("png"):
            img = Image.open(os.path.join(path, file)).convert("RGB")
            #print(np.array(img).shape)
            imgs.append(_transforms_fun(img))
    imgs = torch.stack(imgs)
    #print(imgs.size())
    return inception_score(imgs, cuda, batch_size, resize, splits)

def _fid(paths, batch_size=1, cuda=True, dims=2048):
    return calculate_fid_given_paths(paths, batch_size, cuda, dims)

def _ssim(paths, window_size=11, size_average=True):
    path1, path2 = paths
    imgs1, imgs2 = [], []
    for file in os.listdir(path1):
        if file.endswith("png"):
            img1 = Image.open(os.path.join(path1, file)).convert("RGB")
            img2 = Image.open(os.path.join(path2, file)).convert("RGB")

            imgs1.append(_transforms_fun(img1))
            imgs2.append(_transforms_fun(img2))
    imgs1 = torch.stack(imgs1)
    imgs2 = torch.stack(imgs2)

    return ssim(imgs1, imgs2, window_size = 11, size_average = True)


metrics = {"is":_inception_score, "fid":_fid, "ssim":_ssim}
