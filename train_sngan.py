import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gatedconv import InpaintGCNet, InpaintDirciminator
from models.loss import SNDisLoss, SNGenLoss, ReconLoss
from util.logger import TensorBoardLogger
from util.config import Config
from data.inpaint_dataset import InpaintDataset
from util.evaluation import AverageMeter

import logging
import time
import sys
# python train inpaint.yml
config = Config(sys.argv[1])
logger = logging.getLogger(__name__)
log_dir = 'model_logs/{}_{}'.format(time.strftime('%Y%m%d%H%M', time.localtime(time.time())), config.LOG_DIR)
tensorboardlogger = TensorBoardLogger(log_dir)
cuda0 = torch.device('cuda:{}'.format(config.GPU_ID))
cpu0 = torch.device('cpu')

def logger_init():
    """
    Initialize the logger to some file.
    """
    logging.basicConfig(level=logging.INFO)

    logfile = 'logs/{}_{}.log'.format(time.strftime('%Y%m%d%H%M', time.localtime(time.time())), config.LOG_DIR)
    fh = logging.FileHandler(logfile, mode='w')
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def validate(netG, netD, GANLoss, ReconLoss, DLoss, optG, optD, dataloader, epoch, device=cuda0, val_num=30):
    """
    validate phase
    """
    netG.to(device)
    netD.to(device)
    netG.eval()
    netD.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = {"g_loss":AverageMeter(), "r_loss":AverageMeter(), "whole_loss":AverageMeter(), "d_loss":AverageMeter()}

    netG.train()
    netD.train()
    end = time.time()
    for i, (imgs, masks) in enumerate(dataloader):
        if i > val_num:
            break
        data_time.update(time.time() - end)
        masks = masks['random_free_form']
        #masks = (masks > 0).type(torch.FloatTensor)

        imgs, masks = imgs.to(device), masks.to(device)
        imgs = (imgs / 127.5 - 1)
        # mask is 1 on masked region
        # forward
        coarse_imgs, recon_imgs = netG(imgs, masks)

        complete_imgs = recon_imgs * masks + imgs * (1 - masks)

        pos_imgs = torch.cat([imgs, masks, torch.full_like(masks, 1.)], dim=1)
        neg_imgs = torch.cat([complete_imgs, masks, torch.full_like(masks, 1.)], dim=1)
        pos_neg_imgs = torch.cat([pos_imgs, neg_imgs], dim=0)

        pred_pos_neg = netD(pos_neg_imgs)
        pred_pos, pred_neg = torch.chunk(pred_pos_neg,  2, dim=0)


        g_loss = GANLoss(pred_neg)

        r_loss = ReconLoss(imgs, coarse_imgs, recon_imgs, masks)

        whole_loss = g_loss + r_loss

        # Update the recorder for losses
        losses['g_loss'].update(g_loss.item(), imgs.size(0))
        losses['r_loss'].update(r_loss.item(), imgs.size(0))
        losses['whole_loss'].update(whole_loss.item(), imgs.size(0))

        d_loss = DLoss(pred_pos, pred_neg)
        losses['d_loss'].update(d_loss.item(), imgs.size(0))
        # Update time recorder
        batch_time.update(time.time() - end)


        # Logger logging
        logger.info("Validation Epoch {0}, [{1}/{2}]: Batch Time:{batch_time.val:.4f},\t Data Time:{data_time.val:.4f},\t Whole Gen Loss:{whole_loss.val:.4f}\t,"
                    "Recon Loss:{r_loss.val:.4f},\t GAN Loss:{g_loss.val:.4f},\t D Loss:{d_loss.val:.4f}"
                    .format(epoch, i, len(dataloader), batch_time=batch_time, data_time=data_time, whole_loss=losses['whole_loss'], r_loss=losses['r_loss'] \
                    ,g_loss=losses['g_loss'], d_loss=losses['d_loss']))

        if i*config.BATCH_SIZE < config.STATIC_VIEW_SIZE:
            def img2photo(imgs):
                return ((imgs+1)*127.5).transpose(1,2).transpose(2,3).detach().cpu().numpy()
            # info = { 'val/ori_imgs':img2photo(imgs),
            #          'val/coarse_imgs':img2photo(coarse_imgs),
            #          'val/recon_imgs':img2photo(recon_imgs),
            #          'val/comp_imgs':img2photo(complete_imgs),
            info = {
                     'val/whole_imgs':img2photo(torch.cat([imgs, imgs * (1 - masks), coarse_imgs, recon_imgs, complete_imgs], dim=3))
                     }

            for tag, images in info.items():
                tensorboardlogger.image_summary(tag, images, i)
        end = time.time()


def train(netG, netD, GANLoss, ReconLoss, DLoss, optG, optD, dataloader, epoch, device=cuda0):
    """
    Train Phase, for training and spectral normalization patch gan in
    Free-Form Image Inpainting with Gated Convolution (snpgan)

    """
    netG.to(device)
    netD.to(device)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = {"g_loss":AverageMeter(), "r_loss":AverageMeter(), "whole_loss":AverageMeter(), 'd_loss':AverageMeter()}

    netG.train()
    netD.train()
    end = time.time()
    for i, (imgs, masks) in enumerate(dataloader):
        data_time.update(time.time() - end)
        masks = masks['random_free_form']

        # Optimize Discriminator
        optD.zero_grad(), netD.zero_grad(), netG.zero_grad(), optG.zero_grad()

        imgs, masks = imgs.to(device), masks.to(device)
        imgs = (imgs / 127.5 - 1)
        # mask is 1 on masked region
        coarse_imgs, recon_imgs = netG(imgs, masks)

        complete_imgs = recon_imgs * masks + imgs * (1 - masks)

        pos_imgs = torch.cat([imgs, masks, torch.full_like(masks, 1.)], dim=1)
        neg_imgs = torch.cat([complete_imgs, masks, torch.full_like(masks, 1.)], dim=1)
        pos_neg_imgs = torch.cat([pos_imgs, neg_imgs], dim=0)

        pred_pos_neg = netD(pos_neg_imgs)
        pred_pos, pred_neg = torch.chunk(pred_pos_neg, 2, dim=0)
        d_loss = DLoss(pred_pos, pred_neg)
        losses['d_loss'].update(d_loss.item(), imgs.size(0))
        d_loss.backward(retain_graph=True)

        optD.step()


        # Optimize Generator
        optD.zero_grad(), netD.zero_grad(), optG.zero_grad(), netG.zero_grad()
        pred_neg = netD(neg_imgs)
        #pred_pos, pred_neg = torch.chunk(pred_pos_neg,  2, dim=0)
        g_loss = GANLoss(pred_neg)
        r_loss = ReconLoss(imgs, coarse_imgs, recon_imgs, masks)

        whole_loss = g_loss + r_loss

        # Update the recorder for losses
        losses['g_loss'].update(g_loss.item(), imgs.size(0))
        losses['r_loss'].update(r_loss.item(), imgs.size(0))
        losses['whole_loss'].update(whole_loss.item(), imgs.size(0))
        whole_loss.backward()

        optG.step()


        # Update time recorder
        batch_time.update(time.time() - end)


        if i % config.SUMMARY_FREQ == 0:
            # Logger logging
            logger.info("Epoch {0}, [{1}/{2}]: Batch Time:{batch_time.val:.4f},\t Data Time:{data_time.val:.4f}, Whole Gen Loss:{whole_loss.val:.4f}\t,"
                        "Recon Loss:{r_loss.val:.4f},\t GAN Loss:{g_loss.val:.4f},\t D Loss:{d_loss.val:.4f}" \
                        .format(epoch, i, len(dataloader), batch_time=batch_time, data_time=data_time, whole_loss=losses['whole_loss'], r_loss=losses['r_loss'] \
                        ,g_loss=losses['g_loss'], d_loss=losses['d_loss']))
            # Tensorboard logger for scaler and images
            info_terms = {'WGLoss':whole_loss.item(), 'ReconLoss':r_loss.item(), "GANLoss":g_loss.item(), "DLoss":d_loss.item()}

            for tag, value in info_terms.items():
                tensorboardlogger.scalar_summary(tag, value, epoch*len(dataloader)+i)

            for tag, value in losses.items():
                tensorboardlogger.scalar_summary('avg_'+tag, value.avg, epoch*len(dataloader)+i)

            def img2photo(imgs):
                return ((imgs+1)*127.5).transpose(1,2).transpose(2,3).detach().cpu().numpy()
            # info = { 'train/ori_imgs':img2photo(imgs),
            #          'train/coarse_imgs':img2photo(coarse_imgs),
            #          'train/recon_imgs':img2photo(recon_imgs),
            #          'train/comp_imgs':img2photo(complete_imgs),
            info = {
                     'train/whole_imgs':img2photo(torch.cat([imgs, imgs * (1 - masks), coarse_imgs, recon_imgs, complete_imgs], dim=3))
                     }

            for tag, images in info.items():
                tensorboardlogger.image_summary(tag, images, i)
        end = time.time()

def main():
    logger_init()
    dataset_type = config.DATASET
    batch_size = config.BATCH_SIZE

    # Dataset setting
    logger.info("Initialize the dataset...")
    train_dataset = InpaintDataset(config.DATA_FLIST[dataset_type][0],\
                                      {mask_type:config.DATA_FLIST[config.MASKDATASET][mask_type][0] for mask_type in config.MASK_TYPES}, \
                                      resize_shape=tuple(config.IMG_SHAPES), random_bbox_shape=config.RANDOM_BBOX_SHAPE, \
                                      random_bbox_margin=config.RANDOM_BBOX_MARGIN,
                                      random_ff_setting=config.RANDOM_FF_SETTING)
    train_loader = train_dataset.loader(batch_size=batch_size, shuffle=True,
                                            num_workers=16,pin_memory=True)

    val_dataset = InpaintDataset(config.DATA_FLIST[dataset_type][1],\
                                    {mask_type:config.DATA_FLIST[config.MASKDATASET][mask_type][1] for mask_type in config.MASK_TYPES}, \
                                    resize_shape=tuple(config.IMG_SHAPES), random_bbox_shape=config.RANDOM_BBOX_SHAPE, \
                                    random_bbox_margin=config.RANDOM_BBOX_MARGIN,
                                    random_ff_setting=config.RANDOM_FF_SETTING)
    val_loader = val_dataset.loader(batch_size=1, shuffle=False,
                                        num_workers=1)
    logger.info("Finish the dataset initialization.")

    # Define the Network Structure
    logger.info("Define the Network Structure and Losses")
    netG = InpaintGCNet()
    netD = InpaintDirciminator()

    if config.MODEL_RESTORE != '':
        whole_model_path = 'model_logs/{}'.format( config.MODEL_RESTORE)
        nets = torch.load(whole_model_path)
        netG_state_dict, netD_state_dict = nets['netG_state_dict'], nets['netD_state_dict']
        netG.load_state_dict(netG_state_dict)
        netD.load_state_dict(netD_state_dict)
        logger.info("Loading pretrained models from {} ...".format(config.MODEL_RESTORE))
    #netG, netD = netG.to(cuda0), netD.to(cuda0)

    # Define loss
    recon_loss = ReconLoss(*(config.L1_LOSS_ALPHA))
    gan_loss = SNGenLoss(config.GAN_LOSS_ALPHA)
    dis_loss = SNDisLoss()
    lr, decay = config.LEARNING_RATE, config.WEIGHT_DECAY
    #print(netG.parameters())
    optG = torch.optim.Adam(netG.parameters(), lr=lr, weight_decay=decay)
    optD = torch.optim.Adam(netD.parameters(), lr=4*lr, weight_decay=decay)

    logger.info("Finish Define the Network Structure and Losses")

    # Start Training
    logger.info("Start Training...")
    epoch = 50

    for i in range(epoch):
        #validate(netG, netD, gan_loss, recon_loss, dis_loss, optG, optD, val_loader, i, device=cuda0)

        #train data
        train(netG, netD, gan_loss, recon_loss, dis_loss, optG, optD, train_loader, i, device=cuda0)

        # validate
        validate(netG, netD, gan_loss, recon_loss, dis_loss, optG, optD, val_loader, i, device=cuda0)

        saved_model = {
            'epoch': i + 1,
            'netG_state_dict': netG.to(cpu0).state_dict(),
            'netD_state_dict': netD.to(cpu0).state_dict(),
            # 'optG' : optG.state_dict(),
            # 'optD' : optD.state_dict()
        }
        torch.save(saved_model, '{}/epoch_{}_ckpt.pth.tar'.format(log_dir, i+1))
        torch.save(saved_model, '{}/latest_ckpt.pth.tar'.format(log_dir, i+1))
if __name__ == '__main__':
    main()
