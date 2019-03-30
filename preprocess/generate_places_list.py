import os
import sys
import random

train_prefix = '/home/lhy/datasets/Places2/data_256'
train_path = '/home/lhy/datasets/Places2/places365_train_standard.txt'
data_dict = {}
with open(train_path, 'r') as rf:
    for line in rf:
        fpath, cate = line.strip().split()
        if cate in data_dict:
            data_dict[cate].append(fpath)
        else:
            data_dict[cate] = []
cate_num = 3000
with open(train_path, 'r') as rf:
    with open('/home/lhy/datasets/Places2/trainsmall_flist.txt', 'w') as wf:
        for cate in data_dict:
            for i, fpath in enumerate(data_dict[cate]):
                if i > cate_num:
                    break
                wf.write(train_prefix+fpath+"\n")
        # for line in rf:
        #     fpath, cate = line.strip().split()
        #     wf.write(train_prefix+fpath+'\n')

# val_prefix = '/home/lhy/datasets/Places2/val_256/'
# test_path = '/home/lhy/datasets/Places2/places365_val.txt'
# with open(test_path, 'r') as rf:
#     with open('/home/lhy/datasets/Places2/val_flist.txt', 'w') as wf:
#         for line in rf:
#             fpath, cate = line.strip().split()
#             wf.write(val_prefix+fpath+'\n')

# train_prefix = '/home/lhy/datasets/Places2/data_256'
# train_path = '/home/lhy/datasets/Places2/places365_train_standard.txt'
# data_dict = {}
# with open(train_path, 'r') as rf:
#     for line in rf:
#         fpath, cate = line.strip().split()
#         if cate in data_dict:
#             data_dict[cate].append(fpath)
#         else:
#             data_dict[cate] = []
#
# with open('/home/lhy/datasets/Places2/train_pair_flist.txt', 'w') as wf:
#     for cate, fpaths in data_dict.items():
#         l = len(fpaths)
#         random.shuffle(fpaths)
#         for i, fpath in enumerate(fpaths):
#             wf.write("{} {}\n".format(train_prefix+fpath, train_prefix+fpaths[(i+1)%l]))
#
#
#

train_prefix = '/home/lhy/datasets/Places2/val_256/'
train_path = '/home/lhy/datasets/Places2/places365_val.txt'
data_dict = {}
with open(train_path, 'r') as rf:
    for line in rf:
        fpath, cate = line.strip().split()
        if cate in data_dict:
            data_dict[cate].append(fpath)
        else:
            data_dict[cate] = []

# with open('/home/lhy/datasets/Places2/val_pair_flist.txt', 'w') as wf:
#     for cate, fpaths in data_dict.items():
#         l = len(fpaths)
#         random.shuffle(fpaths)
#         for i, fpath in enumerate(fpaths):
#             wf.write("{} {}\n".format(train_prefix+fpath, train_prefix+fpaths[(i+1)%l]))

val_path = '/home/lhy/datasets/InpaintBenchmark/ImageData'

with open(os.path.join(val_path, "val_large_img_flist.txt"), 'w') as wf:
    for i in range(1000):
        wf.write("{}/{}.png\n".format(val_path, i))
