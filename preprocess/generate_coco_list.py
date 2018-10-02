import os
import sys
import random
import numpy as np
from pycocotools.coco import COCO
import pickle as pkl

mode = "train"
def generate(mode):
    ann_file = "/home/lhy/datasets/coco2017/annotations/instances_{}2017.json".format(mode)

    coco = COCO(ann_file)
    #coco.info()

    cats = coco.loadCats(coco.getCatIds())
    #print(cats)
    nms=[cat['name'] for cat in cats]

    # Only load one
    bbox_dir = "/home/lhy/datasets/coco2017/bbox/{}2017".format(mode)
    train_dir = "/home/lhy/datasets/coco2017/{}2017".format(mode)
    with open("/home/lhy/datasets/coco2017/whole_{}_inpaint_flist.txt".format(mode), "w") as wf:
        for cat_nm in nms:
            catIds = coco.getCatIds(catNms=[cat_nm])
            imgIds = coco.getImgIds(catIds=catIds)
            imgs = coco.loadImgs(imgIds)
            for img in imgs:
                annId = coco.getAnnIds(imgIds=img['id'])
                anns = coco.loadAnns(annId)
                wf.write("{}\n".format(os.path.join(train_dir, img['file_name'])))
                pkl.dump({'bbox':anns[0]['bbox'], 'shape':(img['height'], img['width'])}, open(os.path.join(bbox_dir, img['file_name'][:-3]+"pkl"), 'wb'))

generate('train')
generate('val')
