import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torchvision.models as models
from preprocess.vgg import *
from util.config import Config
from data.inpaint_dataset import InpaintWithFileDataset

import pickle as pkl
import sys
cuda0 = torch.device('cuda:{}'.format(3))

def main():
    config = Config(sys.argv[1])
    dataset_type = config.DATASET
    batch_size = 16

    train_dataset = InpaintWithFileDataset(config.DATA_FLIST[dataset_type][0],\
                                      {mask_type:config.DATA_FLIST[config.MASKDATASET][mask_type][0] for mask_type in config.MASK_TYPES}, \
                                      resize_shape=(224,224), transforms_oprs=['resize', 'to_tensor', 'norm'], random_bbox_shape=config.RANDOM_BBOX_SHAPE, \
                                      random_bbox_margin=config.RANDOM_BBOX_MARGIN,
                                      random_ff_setting=config.RANDOM_FF_SETTING)
    train_loader = train_dataset.loader(batch_size=batch_size, shuffle=False,
                                            num_workers=0)

    # val_dataset = InpaintWithFileDataset(config.DATA_FLIST[dataset_type][1],\
    #                                 {mask_type:config.DATA_FLIST[config.MASKDATASET][mask_type][1] for mask_type in config.MASK_TYPES}, \
    #                                 resize_shape=tuple(config.IMG_SHAPES), transforms_oprs=['random_crop', 'to_tensor', 'norm'], random_bbox_shape=config.RANDOM_BBOX_SHAPE, \
    #                                 random_bbox_margin=config.RANDOM_BBOX_MARGIN,
    #                                 random_ff_setting=config.RANDOM_FF_SETTING)
    #
    # val_loader = val_dataset.loader(batch_size=batch_size, shuffle=False,
    #                                     num_workers=1)

    vgg_model = vgg19_bn(pretrained=True, num_classes=1000)

    vgg_model = vgg_model.to(cuda0)
    start = time.time()
    for i, data in enumerate(train_loader):
        imgs, masks, imgs_path  = data

        masks = masks['random_free_form']
        #print(masks.size(), imgs.size())
        imgs, masks = imgs.to(cuda0), masks.to(cuda0)

        out_feat, out = vgg_model(imgs)

        #out_mask_feat, out_mask = vgg_model(imgs * (1 - masks))

        for j, img_path in enumerate(imgs_path):
            #print(img_path)
            #print(out_feat.size())
            save_terms = {
                "out_feat":out_feat[j].detach().cpu().numpy(),
                "out":out[j].detach().cpu().numpy(),
                #"out_mask_feat":out_mask_feat[j].detach().cpu().numpy(),
                #"out_mask":out_mask[j].detach().cpu().numpy(),
                "img_path":img_path,
                #"img":imgs[j],
                #"mask":masks[j]
            }
            pkl_path = img_path.replace("data_256", "feat/data_256")
            pkl_path, file_name = pkl_path[:pkl_path.rfind('/')], pkl_path[pkl_path.rfind('/'):]
            if not os.path.exists(pkl_path):
                os.makedirs(pkl_path)
            pkl_path = pkl_path + file_name.replace("jpg", "pkl")
            #print(img_path, pkl_path)
            pkl.dump(save_terms, open(pkl_path, "wb"))
        end = time.time()
        if (i+1) % 10 == 0:
            print("Batch {}/{}, Batch Time:{}...".format(i, len(train_loader), end-start))
            start = time.time()




if __name__ == '__main__':
    main()
