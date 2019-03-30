import os
import sys
import random
import numpy as np
path = '/home/lhy/datasets/coco2017/whole_val_inpaint_flist.txt'

with open(path, 'r') as rf:
    for i, line in enumerate(rf):
        if i >= 1000:
            break
        else:
            os.system("cp {} /home/lhy/datasets/InpaintBenchmark/CocoData/{}.png".format(line.strip(), i))
