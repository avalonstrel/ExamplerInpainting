import os
import sys

mode = sys.argv[1]
path = '~/ILSVRC2012/{}'.format(mode)
with open('imagenet_{}_flist.txt'.format(mode),'w') as f:
    for dirname in os.listdir(path):
        dir_path = os.path.join(path, dirname)
        if os.path.isdir(dir_path):
            for filename in os.listdir(dir_path):
                file_path = os.path.join(dir_path, filename)
                if os.path.isfile(file_path):
                    f.write(file_path + '\n')
