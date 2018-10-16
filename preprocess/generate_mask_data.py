import numpy as np
import sys
import os
from util.config import Config
import pickle as pkl
import cv2
from PIL import Image
def random_ff_mask(config):
    """Generate a random free form mask with configuration.

    Args:
        config: Config should have configuration including IMG_SHAPES,
            VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.

    Returns:
        tuple: (top, left, height, width)

    """

    h,w = config['img_shape']
    mask = np.zeros((h,w))
    num_v = 8+np.random.randint(config['mv'])#tf.random_uniform([], minval=0, maxval=config.MAXVERTEX, dtype=tf.int32)

    for i in range(num_v):
        start_x = np.random.randint(w)
        start_y = np.random.randint(h)
        for j in range(1+np.random.randint(5)):
            angle = 0.01+np.random.randint(config['ma'])
            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle
            length = 10+np.random.randint(config['ml'])
            brush_w = 10+np.random.randint(config['mbw'])
            end_x = (start_x + length * np.sin(angle)).astype(np.int32)
            end_y = (start_y + length * np.cos(angle)).astype(np.int32)

            cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
            start_x, start_y = end_x, end_y
    return mask.reshape(mask.shape+(1,)).astype(np.float32)

def random_bbox(config):
    """Generate a random tlhw with configuration.

    Args:
        config: Config should have configuration including IMG_SHAPES,
            VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.

    Returns:
        tuple: (top, left, height, width)

    """

    shape, margin, bbox_shape = config.IMG_SHAPES, config.RANDOM_BBOX_SHAPE, config.RANDOM_BBOX_MARGIN
    img_height = shape[0]
    img_width = shape[1]
    height, width = bbox_shape
    ver_margin, hor_margin = margin
    maxt = img_height - ver_margin - height
    maxl = img_width - hor_margin - width
    t = np.random.randint(low=ver_margin, high=maxt)
    l = np.random.randint(low=hor_margin, high=maxl)
    h = height
    w = width

    mask = np.zeros(shape)
    mask[t:(t+h), l:(l+w)] = 1.

    return mask.reshape(mask.shape+(1,)).astype(np.float32)

def main():
    num = 500
    config = Config(sys.argv[1])
    val_ff_path = "/home/lhy/datasets/InpaintBenchmark/MaskData/val_ff"
    if not os.path.exists(val_ff_path):
        os.makedirs(val_ff_path)
    val_ff_fn = os.path.join(val_ff_path, 'val_ff_mask_flist.txt')
    val_ff_img_fn = os.path.join(val_ff_path, 'val_ff_mask_img_flist.txt')
    with open(val_ff_fn, 'w') as wf:
        with open(val_ff_img_fn, 'w') as wif:
            for i in range(num):
                mask_ff = random_ff_mask({'img_shape':[256,256],'mv':5, 'ma':4.0, 'ml':40, 'mbw':10})

                mask_path = os.path.join(val_ff_path, '{}.pkl'.format(i))
                pkl.dump(mask_ff, open(mask_path,'wb'))
                wf.write(mask_path+'\n')
                mask_img = Image.fromarray(np.tile(mask_ff * 255, (1,1,3)).astype(np.uint8))
                mask_img_path = os.path.join(val_ff_path, '{}.png'.format(i))
                mask_img.save(mask_img_path)
                wif.write(mask_img_path+'\n')


    val_rect_path = "/home/lhy/datasets/InpaintBenchmark/MaskData/val_rect"
    if not os.path.exists(val_rect_path):
        os.makedirs(val_rect_path)
    val_rect_fn = os.path.join(val_rect_path, 'val_rect_mask_flist.txt')
    val_rect_img_fn = os.path.join(val_rect_path, 'val_rect_mask_img_flist.txt')
    with open(val_rect_fn, 'w') as wf:
        with open(val_rect_img_fn, 'w') as wif:
            for i in range(num):
                mask_rect = random_bbox(config)
                mask_path = os.path.join(val_rect_path, '{}.pkl'.format(i))
                pkl.dump(mask_rect, open(mask_path, 'wb'))
                wf.write(mask_path+'\n')
                mask_img = Image.fromarray(np.tile(mask_rect * 255, (1,1,3)).astype(np.uint8))
                mask_img_path = os.path.join(val_rect_path, '{}.png'.format(i))
                mask_img.save(mask_img_path)
                wif.write(mask_img_path+'\n')

if __name__ == '__main__':
    main()
