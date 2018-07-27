import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
ALLMASKTYPES = ['bbox', 'seg', 'random']


def InpaintDataset(Dataset):
    """
    Dataset for Inpainting task
    Params:
        img_flist_path(str): The file which contains img file path list (e.g. test.flist)
        mask_flist_paths_dict(dict): The dict contain the files which contains the pkl or xml file path for
                                generate mask. And the key represent the mask type (e.g. {"bbox":"bbox_flist.txt", "seg":..., "random":None})
        resize_shape(tuple): The shape of the final image
        random_crop(bool) : Determine whether use random crop
        random_bbox_shape(tuple): if use random bbox, it define the shape of the bbox
    Return:
        img, *mask
    """
    def __init__(self, img_flist_path, mask_flist_paths_dict,
                resize_shape=(256, 256), random_crop=True, random_bbox_shape=(32, 32), random_bbox_margin=(64, 64)):

        with open(img_flist_path, 'r') as f:
            self.img_paths = f.read().splitlines()

        self.mask_paths = {}
        for mask_type in mask_flist_paths_dict:
            assert mask_type in ALLMASKTYPES
            if mask_type == 'random':
                self.mask_paths[mask_type] = ['' for i in self.img_paths]
            with open(mask_flist_paths_dict[mask_type]) as f:
                self.mask_paths[mask_type] = f.read().splitlines()

        self.resize_shape = resize_shape
        self.random_bbox_shape = random_bbox_shape
        self.transform_initialize(resize_shape)




    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        # create the paths for images and masks
        img_path = self.img_paths[index]
        mask_paths = {}
        for mask_type in self.mask_paths:
            mask_paths[mask_type] = self.mask_paths[mask_type][index]

        img = self.transforms_fun(read_img(img_path))
        masks = {mask_type:self.transforms_fun(bbox2mask(read_mask(mask_paths[mask_type], mask_type))) for mask_type in mask_paths}

        return img, masks

    def transform_initialize(self, crop_size, config=['random_crop', 'to_tensor']):
        """
        Initialize the transformation oprs and create transform function for img
        """
        self.transforms_oprs = {}
        self.transforms_oprs["hflip "]= transforms.RandomHorizontalFlip(0.5)
        self.transforms_oprs["vflip"] = transforms.RandomVerticalFlip(0.5)
        self.transforms_oprs["random_crop"] = transforms.RandomCrop(crop_size)
        self.transforms_oprs["to_tensor"] = transforms.ToTensor()
        self.transforms_oprs["norm"] = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        self.transforms_oprs["resize"] = transforms.Resize(crop_size)
        self.transforms_oprs["center_crop"] = transforms.CenterCrop(crop_size)
        self.transforms_oprs["rdresizecrop"] = transforms.RandomResizedCrop(crop_size, scale=(0.7, 1.0), ratio=(1,1), interpolation=2)
        self.transforms_fun = transforms.Compose([self.transforms_oprs[name] for name in config])

    @staticmethod
    def read_img(path):
        """
        Read Images
        """
        img = Image.open(path)

        return np.array(img)

    @staticmethod
    def read_mask(path, mask_type):
        """
        Read Masks now only support bbox
        """
        if mask_type == 'random':
            bbox = random_bbox(self.resize_shape, self.random_bbox_shape)
    @staticmethod
    def read_bbox(path):
        """
        The general method for read bbox file by juding the file type
        Return:
            bbox:[y,x,height, width], shape: (height, width)
        """
        if filename[-3:] == 'pkl' and 'Human' in filename:
            return read_bbox_ch(filename)
        elif filename[-3:] == 'pkl' and 'COCO' in filename:
            return read_bbox_pkl(filename)
        else:
            return read_bbox_xml(path)

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
        t = tf.random_uniform(
            [], minval=ver_margin, maxval=maxt, dtype=tf.int32)
        l = tf.random_uniform(
            [], minval=hor_margin, maxval=maxl, dtype=tf.int32)
        h = tf.constant(height)
        w = tf.constant(width)
        return (t, l, h, w)

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
