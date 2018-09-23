import os
import sys


train_prefix = '/home/lhy/datasets/Places2/data_256'
train_path = '/home/lhy/datasets/Places2/places365_train_standard.txt'
with open(train_path, 'r') as rf:
    with open('/home/lhy/datasets/Places2/train_flist.txt', 'w') as wf:
        for line in rf:
            fpath, cate = line.strip().split()
            wf.write(train_prefix+fpath+'\n')

val_prefix = '/home/lhy/datasets/Places2/val_256/'
test_path = '/home/lhy/datasets/Places2/places365_val.txt'
with open(test_path, 'r') as rf:
    with open('/home/lhy/datasets/Places2/val_flist.txt', 'w') as wf:
        for line in rf:
            fpath, cate = line.strip().split()
            wf.write(val_prefix+fpath+'\n')
