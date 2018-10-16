import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gatedconv_exampler import InpaintGCExamplerNet, InpaintDirciminator, MaskInpaintDiscriminator
from models.loss import SNDisLoss, SNGenLoss, ReconLoss, L1ReconLoss
from util.logger import TensorBoardLogger
from util.config import Config
from data.inpaint_dataset import InpaintDataset, InpaintPairDataset
from util.evaluation import AverageMeter

from PIL import Image
import pickle as pkl
import numpy as np
import logging
import time
import sys
import os

# python train inpaint.yml
config = Config(sys.argv[1])
logger = logging.getLogger(__name__)
time_stamp = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
log_dir = 'model_logs/{}_{}'.format(time_stamp, config.LOG_DIR)
result_dir = 'result_logs/{}_{}'.format(time_stamp, config.LOG_DIR)
tensorboardlogger = TensorBoardLogger(log_dir)
cuda0 = torch.device('cuda:{}'.format(config.GPU_IDS[0]))
cuda1 = torch.device('cuda:{}'.format(config.GPU_IDS[1]))
cpu0 = torch.device('cpu')

def logger_init():
    """
    Initialize the logger to some file.
    """
    logging.basicConfig(level=logging.INFO)

    logfile = 'logs/{}_{}.log'.format(time_stamp, config.LOG_DIR)
    fh = logging.FileHandler(logfile, mode='w')
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def validate(nets, loss_terms, opts, dataloader, epoch, devices=(cuda0, cuda1), batch_n="whole"):
    """
    validate phase
    """
    netG, netD  = nets["netG"], nets["netD"]
    GANLoss, ReconLoss, L1ReconLoss, DLoss = loss_terms["GANLoss"], loss_terms["ReconLoss"], loss_terms["L1ReconLoss"], loss_terms["DLoss"]
    optG, optD = opts["optG"], opts["optD"]
    device0, device1 = devices[0], devices[1]
    netG.to(device0)
    netD.to(device0)
    # maskNetD.to(device1)

    netG.eval()
    netD.eval()
    # maskNetD.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = {"g_loss":AverageMeter(), "r_loss":AverageMeter(), "r_ex_loss":AverageMeter(), "whole_loss":AverageMeter(), 'd_loss':AverageMeter(),
              'mask_d_loss':AverageMeter(), 'mask_rec_loss':AverageMeter(),'mask_whole_loss':AverageMeter()}

    end = time.time()
    val_save_dir = os.path.join(result_dir, "val_{}_{}".format(epoch, batch_n+1))
    val_save_real_dir = os.path.join(val_save_dir, "real")
    val_save_gen_dir = os.path.join(val_save_dir, "gen")
    val_save_inf_dir = os.path.join(val_save_dir, "inf")
    if not os.path.exists(val_save_real_dir):
        os.makedirs(val_save_real_dir)
        os.makedirs(val_save_gen_dir)
        os.makedirs(val_save_inf_dir)
    info = {}

    for i, data in enumerate(dataloader):

        data_time.update(time.time() - end, 1)
        imgs, img_exs, masks = data
        masks = masks['val']
        #masks = (masks > 0).type(torch.FloatTensor)

        imgs, img_exs, masks = imgs.to(device0), img_exs.to(device0), masks.to(device0)
        imgs = (imgs / 127.5 - 1)
        img_exs = (img_exs / 127.5 - 1)
        # mask is 1 on masked region
        # forward
        coarse_imgs, recon_imgs, recon_ex_imgs = netG(imgs, img_exs, masks)

        complete_imgs = recon_imgs * masks + imgs * (1 - masks)

        pos_imgs = torch.cat([imgs, masks, torch.full_like(masks, 1.)], dim=1)
        neg_imgs = torch.cat([complete_imgs, masks, torch.full_like(masks, 1.)], dim=1)
        pos_neg_imgs = torch.cat([pos_imgs, neg_imgs], dim=0)
        #mask_pos_neg_imgs = torch.cat([imgs, complete_imgs], dim=0)

        pred_pos_neg = netD(pos_neg_imgs)
        pred_pos, pred_neg = torch.chunk(pred_pos_neg,  2, dim=0)

        # # Mask Gan
        # mask_pos_neg_imgs = mask_pos_neg_imgs.to(device1)
        # mask_pred_pos_neg = maskNetD(mask_pos_neg_imgs)
        # mask_pred_pos, mask_pred_neg = torch.chunk(mask_pred_pos_neg, 2, dim=0)

        g_loss = GANLoss(pred_neg)

        r_loss = ReconLoss(imgs, coarse_imgs, recon_imgs, masks)

        r_ex_loss = L1ReconLoss(img_exs, recon_ex_imgs)

        whole_loss = g_loss + r_loss + r_ex_loss

        # Update the recorder for losses
        losses['g_loss'].update(g_loss.item(), imgs.size(0))
        losses['r_loss'].update(r_loss.item(), imgs.size(0))
        losses['r_ex_loss'].update(r_ex_loss.item(), imgs.size(0))
        losses['whole_loss'].update(whole_loss.item(), imgs.size(0))

        d_loss = DLoss(pred_pos, pred_neg)
        losses['d_loss'].update(d_loss.item(), imgs.size(0))

        # masks = masks.to(device1)
        # mask_d_loss = DLoss(mask_pred_pos*masks + (1-masks), mask_pred_neg*masks + (1-masks))
        # mask_rec_loss = L1ReconLoss(mask_pred_neg, masks)
        # mask_whole_loss = mask_rec_loss

        # masks = masks.to(device0)
        # losses['mask_d_loss'].update(mask_d_loss.item(), imgs.size(0))
        # losses['mask_rec_loss'].update(mask_rec_loss.item(), imgs.size(0))
        # losses['mask_whole_loss'].update(mask_whole_loss.item(), imgs.size(0))

        # Update time recorder
        batch_time.update(time.time() - end, 1)


        # Logger logging

        if (i+1) < config.STATIC_VIEW_SIZE:

            def img2photo(imgs):
                return ((imgs+1)*127.5).transpose(1,2).transpose(2,3).detach().cpu().numpy()
            # info = { 'val/ori_imgs':img2photo(imgs),
            #          'val/coarse_imgs':img2photo(coarse_imgs),
            #          'val/recon_imgs':img2photo(recon_imgs),
            #          'val/comp_imgs':img2photo(complete_imgs),
            info['val/whole_imgs/{}'.format(i)] = {"img":img2photo(torch.cat([imgs * (1 - masks), coarse_imgs, recon_imgs, imgs, complete_imgs], dim=3)),
                                                   }

        else:
            logger.info("Validation Epoch {0}, [{1}/{2}]: Batch Time:{batch_time.val:.4f},\t Data Time:{data_time.val:.4f},\t Whole Gen Loss:{whole_loss.val:.4f}\t,"
                        "Recon Loss:{r_loss.val:.4f},\t Ex Recon Loss:{r_ex_loss.val:.4f},\t GAN Loss:{g_loss.val:.4f},\t D Loss:{d_loss.val:.4f}"
                        .format(epoch, i+1, len(dataloader), batch_time=batch_time, data_time=data_time, whole_loss=losses['whole_loss'], \
                                r_loss=losses['r_loss'], r_ex_loss=losses['r_ex_loss'], g_loss=losses['g_loss'], d_loss=losses['d_loss']))

            for tag, value in losses.items():
                tensorboardlogger.scalar_summary('val/avg_'+tag, value.avg, epoch*len(dataloader)+i)
            j = 0
            for tag, datas in info.items():
                images = datas["img"]
                h, w = images.shape[1], images.shape[2] // 5
                for kv, val_img in enumerate(images):
                    real_img = val_img[:,(3*w):(4*w),:]
                    gen_img = val_img[:,(4*w):(5*w),:]
                    real_img = Image.fromarray(real_img.astype(np.uint8))
                    gen_img = Image.fromarray(gen_img.astype(np.uint8))
                    #pkl.dump({datas[term][kv] for term in datas if term != "img"}, open(os.path.join(val_save_inf_dir, "{}.png".format(j)), 'wb'))
                    real_img.save(os.path.join(val_save_real_dir, "{}.png".format(j)))
                    gen_img.save(os.path.join(val_save_gen_dir, "{}.png".format(j)))
                    j += 1
                tensorboardlogger.image_summary(tag, images, epoch)
            path1, path2 = val_save_real_dir, val_save_gen_dir
            fid_score = metrics['fid']([path1, path2], cuda=False)
            ssim_score = metrics['ssim']([path1, path2])
            tensorboardlogger.scalar_summary('val/fid', fid_score.item(), epoch*len(dataloader)+i)
            tensorboardlogger.scalar_summary('val/ssim', ssim_score.item(), epoch*len(dataloader)+i)
            break
            
        end = time.time()


def train(nets, loss_terms, opts, dataloader, epoch, devices=(cuda0, cuda1), val_datas=None):
    """
    Train Phase, for training and spectral normalization patch gan in
    Free-Form Image Inpainting with Gated Convolution (snpgan)

    """
    netG, netD = nets["netG"], nets["netD"]
    GANLoss, ReconLoss, L1ReconLoss, DLoss = loss_terms["GANLoss"], loss_terms["ReconLoss"], loss_terms["L1ReconLoss"], loss_terms["DLoss"]
    optG, optD = opts["optG"], opts["optD"]
    device0, device1 = devices[0], devices[1]
    netG.to(device0)
    netD.to(device0)
    # maskNetD.to(device1)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = {"g_loss":AverageMeter(), "r_loss":AverageMeter(), "r_ex_loss":AverageMeter(), "whole_loss":AverageMeter(), 'd_loss':AverageMeter(),}
              # 'mask_d_loss':AverageMeter(), 'mask_rec_loss':AverageMeter(),'mask_whole_loss':AverageMeter()}

    netG.train()
    netD.train()
    # maskNetD.train()
    end = time.time()
    for i, data in enumerate(dataloader):
        data_time.update(time.time() - end)
        imgs, img_exs, masks = data
        masks = masks['random_free_form']

        # Optimize Discriminator
        optD.zero_grad(), netD.zero_grad(), netG.zero_grad(), optG.zero_grad()

        imgs, img_exs, masks = imgs.to(device0), img_exs.to(device0), masks.to(device0)
        imgs = (imgs / 127.5 - 1)
        img_exs = (img_exs / 127.5 - 1)
        # mask is 1 on masked region
        coarse_imgs, recon_imgs, recon_ex_imgs = netG(imgs, img_exs, masks)

        complete_imgs = recon_imgs * masks + imgs * (1 - masks)

        pos_imgs = torch.cat([imgs, masks, torch.full_like(masks, 1.)], dim=1)
        neg_imgs = torch.cat([complete_imgs, masks, torch.full_like(masks, 1.)], dim=1)
        pos_neg_imgs = torch.cat([pos_imgs, neg_imgs], dim=0)
        #mask_pos_neg_imgs = torch.cat([imgs, complete_imgs], dim=0)

        # Discriminator Loss
        pred_pos_neg = netD(pos_neg_imgs)
        pred_pos, pred_neg = torch.chunk(pred_pos_neg, 2, dim=0)
        d_loss = DLoss(pred_pos, pred_neg)
        d_loss.backward(retain_graph=True)
        optD.step()


        # Mask Discriminator Loss
        # mask_pos_neg_imgs = mask_pos_neg_imgs.to(device1)
        # masks = masks.to(device1)
        # mask_pred_pos_neg = maskNetD(mask_pos_neg_imgs)
        # mask_pred_pos, mask_pred_neg = torch.chunk(mask_pred_pos_neg, 2, dim=0)
        # mask_d_loss = DLoss(mask_pred_pos*masks , mask_pred_neg*masks )
        # mask_rec_loss = L1ReconLoss(mask_pred_neg, masks, masks=masks)

        losses['d_loss'].update(d_loss.item(), imgs.size(0))
        # losses['mask_d_loss'].update(mask_d_loss.item(), imgs.size(0))
        # losses['mask_rec_loss'].update(mask_rec_loss.item(), imgs.size(0))
        # mask_whole_loss = mask_rec_loss
        # losses['mask_whole_loss'].update(mask_whole_loss.item(), imgs.size(0))
        # mask_whole_loss.backward(retain_graph=True)
        # optMD.step()


        # Optimize Generator
        # masks = masks.to(device0)
        optD.zero_grad(), netD.zero_grad(), optG.zero_grad(), netG.zero_grad(),# optMD.zero_grad(), maskNetD.zero_grad()
        pred_neg = netD(neg_imgs)
        #pred_pos, pred_neg = torch.chunk(pred_pos_neg,  2, dim=0)
        g_loss = GANLoss(pred_neg)
        r_loss = ReconLoss(imgs, coarse_imgs, recon_imgs, masks)
        r_ex_loss = L1ReconLoss(img_exs, recon_ex_imgs)

        whole_loss = g_loss + r_loss + r_ex_loss

        # Update the recorder for losses
        losses['g_loss'].update(g_loss.item(), imgs.size(0))
        losses['r_loss'].update(r_loss.item(), imgs.size(0))
        losses['r_ex_loss'].update(r_ex_loss.item(), imgs.size(0))
        losses['whole_loss'].update(whole_loss.item(), imgs.size(0))
        whole_loss.backward()

        optG.step()

        # Update time recorder
        batch_time.update(time.time() - end)

        if (i+1) % config.SUMMARY_FREQ == 0:
            # Logger logging
            logger.info("Epoch {0}, [{1}/{2}]: Batch Time:{batch_time.val:.4f},\t Data Time:{data_time.val:.4f}, Whole Gen Loss:{whole_loss.val:.4f}\t,"
                        "Recon Loss:{r_loss.val:.4f}, \t Ex Recon Loss:{r_ex_loss.val:.4f},\t GAN Loss:{g_loss.val:.4f},\t D Loss:{d_loss.val:.4f}, " \
                        .format(epoch, i+1, len(dataloader), batch_time=batch_time, data_time=data_time \
                                ,whole_loss=losses['whole_loss'], r_loss=losses['r_loss'], r_ex_loss=losses['r_ex_loss'] \
                        ,g_loss=losses['g_loss'], d_loss=losses['d_loss']))
            # Tensorboard logger for scaler and images
            info_terms = {'WGLoss':whole_loss.item(), 'ReconLoss':r_loss.item(), "GANLoss":g_loss.item(), "DLoss":d_loss.item(), }

            for tag, value in info_terms.items():
                tensorboardlogger.scalar_summary(tag, value, epoch*len(dataloader)+i)

            for tag, value in losses.items():
                tensorboardlogger.scalar_summary('avg_'+tag, value.avg, epoch*len(dataloader)+i)

            def img2photo(imgs):
                return ((imgs+1)*127.5).transpose(1,2).transpose(2,3).detach().cpu().numpy()

            info = {
                     'train/whole_imgs':img2photo(torch.cat([imgs * (1 - masks), coarse_imgs, recon_imgs, imgs, complete_imgs], dim=3))
                     }

            for tag, images in info.items():
                tensorboardlogger.image_summary(tag, images, epoch*len(dataloader)+i)

        if (i+1) % config.VAL_SUMMARY_FREQ == 0 and val_datas is not None:

            validate(nets, loss_terms, opts, val_datas , epoch, devices, batch_n=i)
            netG.train()
            netD.train()
            #maskNetD.train()
        end = time.time()

def main():
    logger_init()
    dataset_type = config.DATASET
    batch_size = config.BATCH_SIZE

    # Dataset setting
    logger.info("Initialize the dataset...")
    train_dataset = InpaintPairDataset(config.DATA_FLIST[dataset_type][0],\
                                      {mask_type:config.DATA_FLIST[config.MASKDATASET][mask_type][0] for mask_type in config.MASK_TYPES}, \
                                      resize_shape=tuple(config.IMG_SHAPES), random_bbox_shape=config.RANDOM_BBOX_SHAPE, \
                                      random_bbox_margin=config.RANDOM_BBOX_MARGIN,
                                      random_ff_setting=config.RANDOM_FF_SETTING)
    train_loader = train_dataset.loader(batch_size=batch_size, shuffle=True,
                                            num_workers=16,pin_memory=True)

    val_dataset = InpaintPairDataset(config.DATA_FLIST[dataset_type][1],\
                                    {mask_type:config.DATA_FLIST[config.MASKDATASET][mask_type][1] for mask_type in ('val',)}, \
                                    resize_shape=tuple(config.IMG_SHAPES), random_bbox_shape=config.RANDOM_BBOX_SHAPE, \
                                    random_bbox_margin=config.RANDOM_BBOX_MARGIN,
                                    random_ff_setting=config.RANDOM_FF_SETTING)

    val_loader = val_dataset.loader(batch_size=1, shuffle=False,
                                        num_workers=1)

    ### Generate a new val data
    val_datas = []
    j = 0
    for i, data in enumerate(val_loader):
        if j < config.STATIC_VIEW_SIZE:
            imgs = data[0]
            if imgs.size(1) == 3:
                val_datas.append(data)
                j += 1
        else:
            break

    #val_datas = [(imgs, masks) for imgs, masks in val_loader]

    val_loader = val_dataset.loader(batch_size=1, shuffle=False,
                                        num_workers=1)
    logger.info("Finish the dataset initialization.")

    # Define the Network Structure
    logger.info("Define the Network Structure and Losses")
    netG = InpaintGCExamplerNet(n_in_channel=8)
    netD = InpaintDirciminator(n_in_channel=5)
    #maskNetD = MaskInpaintDiscriminator(n_in_channel=3)
    if config.MODEL_RESTORE != '':
        whole_model_path = 'model_logs/{}'.format( config.MODEL_RESTORE)
        nets = torch.load(whole_model_path)
        netG_state_dict, netD_state_dict = nets['netG_state_dict'], nets['netD_state_dict']
        netG.load_state_dict(netG_state_dict)
        netD.load_state_dict(netD_state_dict)
        logger.info("Loading pretrained models from {} ...".format(config.MODEL_RESTORE))

    # Define loss
    recon_loss = ReconLoss(*(config.L1_LOSS_ALPHA))
    l1_recon_loss = L1ReconLoss(config.L1_RECONLOSS_ALPHA)
    gan_loss = SNGenLoss(config.GAN_LOSS_ALPHA)
    dis_loss = SNDisLoss()
    lr, decay = config.LEARNING_RATE, config.WEIGHT_DECAY
    optG = torch.optim.Adam(netG.parameters(), lr=lr, weight_decay=decay)
    optD = torch.optim.Adam([{'params':netD.parameters()} ], lr=4*lr, weight_decay=decay)
    #optMD = torch.optim.Adam([{'params':maskNetD.parameters()} ], lr=4*lr, weight_decay=decay)

    nets = {
        "netG":netG,
        "netD":netD,
        #"maskNetD":maskNetD
    }
    losses = {
        "GANLoss":gan_loss,
        "ReconLoss":recon_loss,
        "L1ReconLoss":l1_recon_loss,
        "DLoss":dis_loss
    }
    opts = {
        "optG":optG,
        "optD":optD,
        #"optMD":optMD
    }

    logger.info("Finish Define the Network Structure and Losses")

    # Start Training
    logger.info("Start Training...")
    epoch = 50

    for i in range(epoch):

        #train data
        train(nets, losses, opts, train_loader, i, devices=(cuda0,cuda1), val_datas=val_datas)

        # validate
        validate(nets, losses, opts, val_datas, i, devices=(cuda0,cuda1))

        saved_model = {
            'epoch': i + 1,
            'netG_state_dict': netG.to(cpu0).state_dict(),
            'netD_state_dict': netD.to(cpu0).state_dict(),
            'netMD_state_dict':maskNetD.to(cpu0).state_dict()
            # 'optG' : optG.state_dict(),
            # 'optD' : optD.state_dict()
        }
        torch.save(saved_model, '{}/epoch_{}_ckpt.pth.tar'.format(log_dir, i+1))
        torch.save(saved_model, '{}/latest_ckpt.pth.tar'.format(log_dir, i+1))
if __name__ == '__main__':
    main()
