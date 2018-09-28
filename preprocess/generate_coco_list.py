import os
import sys
import random
import numpy as np
from pycocotools.coco import COCO
import pickle as pkl

ann_file = "/home/lhy/datasets/coco2017/annotations/instances_val2017.json"

coco = COCO(ann_file)
#coco.info()

cats = coco.loadCats(coco.getCatIds())
print(cats)
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))
print(nms)
# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['person'])
#print(catIds)
imgIds = coco.getImgIds(catIds=catIds )
#print(imgIds)
imgIds = coco.getImgIds(imgIds = [324158])
#print(imgIds)
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
#print(img)
#print(img['id'])
annId = coco.getAnnIds(imgIds=img['id'])
#print(annId)
ann = coco.loadAnns(annId)
#print(ann)
# Only load one
bbox_dir = "/home/lhy/datasets/coco2017/bbox/train2017"
train_dir = "/home/lhy/datasets/coco2017/train2017"
with open("/home/lhy/datasets/coco2017/whole_train_inpaint_flist.txt", "w") as wf:
    for cat_nm in nms:
        catIds = coco.getCatIds(catNms=[cat_nm])
        imgIds = coco.getImgIds(catIds=catIds)
        imgs = coco.loadImgs(imgIds)
        for img in imgs:
            annIds = coco.getAnnIds(imgIds=img['id'])
            anns = coco.loadAnns(annId)
            wf.write("{}\n".format(os.path.join(train_dir, img['file_name'])))
            pkl.dump({'bbox':anns[0]['bbox'], 'shape':(img['height'], img['width'])}, open(os.path.join(bbox_dir, img['file_name'][:-3]+"pkl"), 'wb'))
