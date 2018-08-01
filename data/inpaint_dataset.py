import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from .base_dataset import BaseDataset, NoriBaseDataset

import numpy as np
import cv2

ALLMASKTYPES = ['bbox', 'seg', 'random']




class InpaintDataset(BaseDataset):
    """
    Dataset for Inpainting task
    Params:
        img_flist_path(str): The file which contains img file path list (e.g. test.flist)
        mask_flist_paths_dict(dict): The dict contain the files which contains the pkl or xml file path for
                                generate mask. And the key represent the mask type (e.g. {"bbox":"bbox_flist.txt", "seg":..., "random":None})
        resize_shape(tuple): The shape of the final image (default:(256,256))
        transforms_oprs(list) : Determine which transformation used on the imgae (default:['random_crop', 'to_tensor'])
        random_bbox_shape(tuple): if use random bbox mask, it define the shape of the mask (default:(32,32))
        random_bbox_margin(tuple): if use random bbox, it define the margin of the bbox which means the distance between the mask and the margin of the image
                                    (default:(64,64))
    Return:
        img, *mask
    """
    def __init__(self, img_flist_path, mask_flist_paths_dict,
                resize_shape=(256, 256), transforms_oprs=['random_crop', 'to_tensor'],
                random_bbox_shape=(32, 32), random_bbox_margin=(64, 64),
                random_ff_setting={'mv':5, 'ma':4.0, 'ml':40, 'mbw':10}):

        with open(img_flist_path, 'r') as f:
            self.img_paths = f.read().splitlines()

        self.mask_paths = {}
        for mask_type in mask_flist_paths_dict:
            assert mask_type in ALLMASKTYPES
            if 'random' in mask_type:
                self.mask_paths[mask_type] = ['' for i in self.img_paths]
            else:
                with open(mask_flist_paths_dict[mask_type]) as f:
                    self.mask_paths[mask_type] = f.read().splitlines()

        self.resize_shape = resize_shape
        self.random_bbox_shape = random_bbox_shape
        self.random_bbox_margin = random_bbox_margin
        self.random_ff_setting = random_ff_setting
        self.transform_initialize(resize_shape, transforms_oprs)




    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        # create the paths for images and masks
        img_path = self.img_paths[index]
        mask_paths = {}
        for mask_type in self.mask_paths:
            mask_paths[mask_type] = self.mask_paths[mask_type][index]

        img = self.transforms_fun(self.read_img(img_path))
        masks = {mask_type:self.transforms_fun(self.read_mask(mask_paths[mask_type], mask_type)) for mask_type in mask_paths}

        return img*255, masks*255

    def read_img(self, path):
        """
        Read Image
        """
        img = Image.open(path)
        return img


    def read_mask(self, path, mask_type):
        """
        Read Masks now only support bbox
        """
        if mask_type == 'random':
            bbox = InpaintDataset.random_bbox(self.resize_shape, self.random_bbox_margin, self.random_bbox_shape)
        elif mask_type == 'random_free_form':
            mask = InpaintDataset.random_ff_mask(self.random_ff_setting)
        else:
            bbox = InpaintDataset.read_bbox(path)
        return bbox2mask(bbox, self.resize_shape)

    @staticmethod
    def read_bbox(path):
        """
        The general method for read bbox file by juding the file type
        Return:
            bbox:[y,x,height, width], shape: (height, width)
        """
        if filename[-3:] == 'pkl' and 'Human' in filename:
            return InpaintDataset.read_bbox_ch(filename)
        elif filename[-3:] == 'pkl' and 'COCO' in filename:
            return InpaintDataset.read_bbox_pkl(filename)
        else:
            return InpaintDataset.read_bbox_xml(path)

    @staticmethod
    def read_bbox_xml(path):
        """
        Read bbox for voc xml
        Return:
            bbox:[y,x,height, width], shape: (height, width)
        """
        with open(filename, 'r') as reader:
            xml = reader.read()
        soup = BeautifulSoup(xml, 'xml')
        size = {}
        for tag in soup.size:
            if tag.string != "\n":
                size[tag.name] = int(tag.string)
        objects = soup.find_all('object')
        bndboxs = []
        for obj in objects:
            bndbox = {}
            for tag in obj.bndbox:
                if tag.string != '\n':
                    bndbox[tag.name] = int(tag.string)

            bbox = [bndbox['ymin'], bndbox['xmin'], bndbox['ymax']-bndbox['ymin'], bndbox['xmax']-bndbox['xmin']]
            bndboxs.append(bbox)
        #print(bndboxs, size)
        return bndboxs, (size['height'], size['width'])

    @staticmethod
    def read_bbox_pkl(path):
        """
        Read bbox from coco pkl
        Return:
            bbox:[y,x,height, width], shape: (height, width)
        """
        aux_dict = pkl.load(open(path, 'rb'))
        bbox = aux_dict["bbox"]
        shape = aux_dict["shape"]
        #bbox = random.choice(bbox)
        #fbox = bbox['fbox']
        return [[int(bbox[1]), int(bbox[0]), int(bbox[3]), int(bbox[2])]], (shape[1], shape[0])

    @staticmethod
    def read_bbox_ch(path):
        """
        Read bbox from crowd human pkl
        Return:
            bbox:[y,x,height, width], shape: (height, width)
        """
        aux_dict = pkl.load(open(path, 'rb'))
        bboxs = aux_dict["bbox"]
        bbox = random.choice(bboxs)
        extra = bbox['extra']
        shape = aux_dict["shape"]
        while 'ignore' in extra and extra['ignore'] == 1 and bbox['fbox'][0] < 0 and bbox['fbox'][1] < 0:
            bbox = random.choice(bboxs)
            extra = bbox['extra']
        fbox = bbox['fbox']
        return [[fbox[1],fbox[0],fbox[3],fbox[2]]], (shape[1], shape[0])

    @staticmethod
    def read_seg_img(path):
        pass

    @staticmethod
    def random_bbox(shape, margin, bbox_shape):
        """Generate a random tlhw with configuration.

        Args:
            config: Config should have configuration including IMG_SHAPES,
                VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.

        Returns:
            tuple: (top, left, height, width)

        """
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
        return (t, l, h, w)

    @staticmethod
    def random_ff_mask(config):
        """Generate a random free form mask with configuration.

        Args:
            config: Config should have configuration including IMG_SHAPES,
                VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.

        Returns:
            tuple: (top, left, height, width)

        """

        h,w = img_shape
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


    @staticmethod
    def bbox2mask(bbox, shape):
        """Generate mask tensor from bbox.

        Args:
            bbox: configuration tuple, (top, left, height, width)
            config: Config should have configuration including IMG_SHAPES,
                MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.

        Returns:
            tf.Tensor: output with shape [1, H, W, 1]

        """
        height, width = shape
        mask = np.zeros(( height, width, 1), np.float32)
        h = int(0.1*bbox[2])+np.random.randint(int(bbox[2]*0.2+1))
        w = int(0.1*bbox[3])+np.random.randint(int(bbox[3]*0.2)+1)
        mask[bbox[0]+h:bbox[0]+bbox[2]-h,
             bbox[1]+w:bbox[1]+bbox[3]-w, :] = 1.
        return mask

class NoriInpaintDataset(NoriBaseDataset):
    """
    Dataset for Inpainting task but use nori instead images
    Params:
        img_nori_list_path(str): The file which contains img file path list (e.g. test.flist)
        mask_nori_list_paths_dict(dict): The dict contain the files which contains the pkl or xml file path for
                                generate mask. And the key represent the mask type (e.g. {"bbox":"bbox_flist.txt", "seg":..., "random":None})
        resize_shape(tuple): The shape of the final image (default:(256,256))
        transforms_oprs(list) : Determine which transformation used on the imgae (default:['random_crop', 'to_tensor'])
        random_bbox_shape(tuple): if use random bbox mask, it define the shape of the mask (default:(32,32))
        random_bbox_margin(tuple): if use random bbox, it define the margin of the bbox which means the distance between the mask and the margin of the image
                                    (default:(64,64))
    Return:
        img, *mask
    """
    def __init__(self, img_nori_list_path, img_nori_path, mask_flist_paths_dict,
                resize_shape=(256, 256), transforms_oprs=['random_crop', 'to_tensor', 'norm'], random_bbox_shape=(32, 32), random_bbox_margin=(64, 64)):

        self.img_nori_list, self.img_cls_ids, self.img_nr = self.initialize_nori(img_nori_list_path, img_nori_path)

        # self.mask_nori_lists = {}
        # self.mask_nrs = {}
        # for mask_type in mask_nori_list_paths_dict:
        #     assert mask_type in ALLMASKTYPES
        #     if mask_type == 'random':
        #         self.mask_paths[mask_type] = ['' for i in self.img_nori_list]
        #     self.mask_nori_lists[mask_type], _, self.mask_nrs[mask_type] = self.initialize_nori(mask_nori_list_paths_dict[mask_type])
        self.mask_paths = {}
        for mask_type in mask_flist_paths_dict:
            assert mask_type in ALLMASKTYPES
            if mask_type == 'random':
                self.mask_paths[mask_type] = ['' for i in self.img_nori_list]
            else:
                with open(mask_flist_paths_dict[mask_type]) as f:
                    self.mask_paths[mask_type] = f.read().splitlines()

        self.resize_shape = resize_shape
        self.random_bbox_shape = random_bbox_shape
        self.random_bbox_margin = random_bbox_margin
        self.transform_initialize(resize_shape, transforms_oprs)

    def __len__(self):
        return len(self.img_nori_list)

    def read_mask(self, path, mask_type):
        """
        Read Masks now only support bbox
        """
        if mask_type == 'random':
            bbox = InpaintDataset.random_bbox(self.resize_shape, self.random_bbox_margin, self.random_bbox_shape)
        else:
            bbox = InpaintDataset.read_bbox(path)
        mask = np.tile((InpaintDataset.bbox2mask(bbox, self.resize_shape)*255).astype(np.uint8), (1,1,3))
        #print(mask.shape)
        #print(mask)
        return Image.fromarray(mask)

    def read_img(self, nori_id):
        """
        Read Images
        """

        img_bytes = self.img_nr.get(nori_id)
        nparr = np.fromstring(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        return Image.fromarray(np.array(img)[:,:,::-1])

    def __getitem__(self, index):
        # create the paths for images and masks
        try:
            img_path = self.img_nori_list[index]
            mask_paths = {}
            for mask_type in self.mask_paths:
                mask_paths[mask_type] = self.mask_paths[mask_type][index]

            img = self.transforms_fun(self.read_img(img_path))
            masks = {mask_type:self.transforms_fun(self.read_mask(mask_paths[mask_type], mask_type)) for mask_type in mask_paths}

            return img, masks, self.img_cls_ids[index]
        except:
            #print("error")
            return self.__getitem__(index-1)
