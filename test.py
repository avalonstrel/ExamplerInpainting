import torch
from evaluation import metrics
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('path', type=str, nargs=2,
                    help=('Path to the generated images'))

args = parser.parse_args()
path1, path2 = args.path
#path1:real data, path2:generated data
inception_score, std_is = None, None
inception_score, std_is = metrics['is'](path1, )
inception_score2, std_is2 = metrics['is'](path2, )
fid_score = None
ssim_score = None
fid_score = metrics['fid']([path1, path2])
ssim_score = metrics['ssim']([path1, path2])

print("IS Mean:{}, IS STD:{}, FID:{}, SSIM:{}".format(inception_score2, std_is, fid_score, ssim_score))
