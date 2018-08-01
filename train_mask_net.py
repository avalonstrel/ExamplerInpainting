import torch
import torchvision.models as models
import torch.backends.cudnn as cudnn
from data.inpaint_dataset import InpaintDataset, NoriInpaintDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from util.util import adjust_learning_rate
from util.evaluation import accuracy, AverageMeter
from PIL import Image
import numpy as np
import time
import logging
import sys
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logfile = 'logs/{}.log'.format(time.strftime('%Y%m%d%H%M', time.localtime(time.time())))
fh = logging.FileHandler(logfile, mode='w')
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)
print_freq = 100
mask_rate = 0.3
device = torch.device('cuda:{}'.format(sys.argv[2]))
train_img_nori_path = '/unsullied/sharefs/g:brain/imagenet/ILSVRC2012/imagenet.train.nori'
train_img_nori_list_path = "/unsullied/sharefs/maningning/wh/Dataset/ImageNet2012/imagenet.train.nori.list"
val_img_nori_path = '/unsullied/sharefs/g:brain/imagenet/ILSVRC2012/imagenet.val.nori'
val_img_nori_list_path = "/unsullied/sharefs/maningning/wh/Dataset/ImageNet2012/imagenet.val.nori.list"
mask_flist_paths_dict = {'random':None}
resize_shape = (224,224)
margin = (32,32)
bbox_shape = (32,32)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

transform_fun = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
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
def generate_mask():
    mask = bbox2mask(random_bbox((256,256), margin, bbox_shape), (256,256))
    mask = np.tile(mask.reshape((256,256, 1)),(1,1,3))*255
    mask = Image.fromarray(mask.astype(np.uint8))
    return transform_fun(mask)

def validate(net, dataloader, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    net.eval()
    with torch.no_grad():
        end = time.time()
        for i, (imgs,  cls_ids) in enumerate(dataloader):
            imgs, cls_ids = imgs.to(device), cls_ids.to(device)
            masks = generate_mask()
            masks = masks.to(device)
            pred = net(imgs*(1-masks))
            loss = criterion(pred, cls_ids)
            #measure
            prec1, prec5 = accuracy(pred, cls_ids, topk=(1, 5))
            losses.update(loss.item(), imgs.size(0))
            top1.update(prec1[0], imgs.size(0))
            top5.update(prec5[0], imgs.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % print_freq == 0:
                logger.info('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(dataloader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

    return top1.avg, top5.avg
def train(net, dataloader, epoch, opt, criterion):
    net.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for i, (imgs, cls_ids) in enumerate(dataloader):
        data_time.update(time.time() - end)

        imgs, cls_ids = imgs.to(device), cls_ids.to(device)
        opt.zero_grad()
        masks = generate_mask()
        masks = masks.to(device)
        if np.random.rand() < mask_rate:
            pred = net(imgs*(1-masks))
        else:
            pred = net(imgs)
        loss = criterion(pred, cls_ids)

        loss.backward()
        opt.step()

        #measure
        prec1, prec5 = accuracy(pred, cls_ids, topk=(1, 5))
        losses.update(loss.item(), imgs.size(0))
        top1.update(prec1[0], imgs.size(0))
        top5.update(prec5[0], imgs.size(0))


        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq  == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(dataloader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

def main(net_type='vgg'):

    epoch = 100
    batch_size = 64
    n_val = 1000
    lr = 0.0001
    decay = 0.00005
    momentum = 0.9
    cudnn.benchmark = True
    logger.info("Define Network and Loss...")
    if net_type == 'vgg':
        net = models.vgg19(pretrained=True)
    elif net_type == 'resnet':
        net = models.resnet50(pretrained=True)
    net.to(device)

    criterion = torch.nn.CrossEntropyLoss()

    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
    logger.info("Finish")
    traindir = '/home/lhy/ILSVRC2012/train'
    valdir = '/home/lhy/ILSVRC2012/val'

    # train_dataset = NoriInpaintDataset(train_img_nori_list_path, train_img_nori_path, mask_flist_paths_dict)
    # train_dataloader = train_dataset.loader(batch_size=batch_size, num_workers=batch_size//4, shuffle=True)
    # val_dataset = NoriInpaintDataset(val_img_nori_list_path, val_img_nori_path, mask_flist_paths_dict)
    # val_dataloader = val_dataset.loader(batch_size=batch_size, num_workers=batch_size//4, shuffle=False)
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=16, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=16, pin_memory=True)


    for i in range(epoch):
        adjust_learning_rate(opt, epoch, lr)

        train(net, train_loader, i, opt, criterion)

        # evaluate on validation set
        prec1, perc5 = validate(net, val_loader, criterion)

        msg = "Epoch {}/{} Val, Perc1:{}, Perc5{}".format(i, epoch, prec1, perc5)
        logger.info(msg)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': net_type,
            'state_dict': net.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : opt.state_dict(),
        }, is_best, filename=sys.argv[1]+"checkpoint.pth.tar")

if __name__ == '__main__':
    main(sys.argv[1])
