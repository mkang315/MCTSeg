# coding=utf-8
import argparse
import os
import time
import logging
import random
import numpy as np
from collections import OrderedDict
import setproctitle

import torch
import torch.optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.transforms import *
from data.datasets_nii import Brats_loadall_nii, Brats_loadall_test_nii
from data.data_utils import init_fn
from model.net import Model
from utils import Parser, criterions
from utils.parser import setup
from utils.lr_scheduler import LR_Scheduler, record_loss, MultiEpochsDataLoader
from predict import AverageMeter, test_softmax

parser = argparse.ArgumentParser()

parser.add_argument('-batch_size', '--batch_size', default=1, type=int, help='Batch size')
parser.add_argument('--datapath', default='./datasets/BraTS20/Train_npy', type=str)
parser.add_argument('--dataname', default='BRATS20', type=str)
parser.add_argument('--user', default='mkang315', type=str)
parser.add_argument('--savepath', default='./runs/BraTS20/output2', type=str)
parser.add_argument('--resume', default='./runs/BraTS20/output2/model_980.pth', type=str)
parser.add_argument('--start_epoch', default=980, type=int)
parser.add_argument('--pretrain', default=None, type=str)
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--iter_per_epoch', default=150, type=int)
parser.add_argument('--region_fusion_start_epoch', default=0, type=int)
parser.add_argument('--seed', default=1024, type=int)
parser.add_argument('--gpu', default='1', type=str)
parser.add_argument('--load', default=True, type=bool)
path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
setup(args, 'training')
args.train_transforms = 'Compose([RandCrop3D((128,128,128)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'

ckpts = args.savepath
os.makedirs(ckpts, exist_ok=True)

###tensorboard writer
writer = SummaryWriter(os.path.join(args.savepath, 'summary'))

###modality missing mask
masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False],
         [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True],
         [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]
masks_torch = torch.from_numpy(np.array(masks))
mask_name = ['t2', 't1c', 't1', 'flair', 't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
             'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2', 'flairt1cet1t2']
print(masks_torch.int())


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    ##########setting seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    ##########setting models
    if args.dataname in ['BRATS2021', 'BRATS2020', 'BRATS2018']:
        num_cls = 4
    elif args.dataname == 'BRATS2015':
        num_cls = 5
    else:
        print('dataset is error')
        exit(0)
    model = Model(num_cls=num_cls)
    # print(model)
    model = torch.nn.DataParallel(model).cuda()

    ##########Setting learning schedule and optimizer
    lr_schedule = LR_Scheduler(args.lr, args.num_epochs)
    train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay}]
    optimizer = torch.optim.Adam(train_params, betas=(0.9, 0.999), eps=1e-08, amsgrad=True)

    if os.path.isfile(args.resume) and args.load:
        logging.info('loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info('successfully loading checkpoint {} and traing from epoch:{}'.format(args.resume, args.start_epoch))

    else:
        logging.info('re-traing!')

    ##########Setting data
    if args.dataname in ['BRATS18', 'BRATS20']:
        train_file = 'train.txt'
        test_file = 'test.txt'
    elif args.dataname == 'BRATS18':
        ####BRATS2018 contains three splits (1,2,3)
        train_file = 'train3.txt'
        test_file = 'test3.txt'

    logging.info(str(args))
    if os.path.isfile(args.resume) and args.load:
        logging.info('loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info('successfully loading checkpoint {} and traing from epoch:{}'.format(args.resume, args.start_epoch))

    else:
        logging.info('re-traing!')
    train_set = Brats_loadall_nii(transforms=args.train_transforms, root=args.datapath, num_cls=num_cls,
                                  train_file=train_file)
    test_set = Brats_loadall_test_nii(transforms=args.test_transforms, root=args.datapath, test_file=test_file)
    train_loader = MultiEpochsDataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=init_fn)
    test_loader = MultiEpochsDataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)

    ##########Evaluate
    # if args.resume is not None:
    #     checkpoint = torch.load(args.resume)
    #     logging.info('best epoch: {}'.format(checkpoint['epoch']))
    #     model.load_state_dict(checkpoint['state_dict'])
        # test_score = AverageMeter()
        # with torch.no_grad():
        #     logging.info('###########test set wi post process###########')
        #     for i, mask in enumerate(masks[::-1]):
        #         logging.info('{}'.format(mask_name[::-1][i]))
        #         dice_score = test_softmax(
        #                         test_loader,
        #                         model,
        #                         dataname = args.dataname,
        #                         feature_mask = mask,
        #                         mask_name = mask_name[::-1][i])
        #         test_score.update(dice_score)
        #     logging.info('Avg scores: {}'.format(test_score.avg))
        #     exit(0)

    ##########Training
    start = time.time()
    torch.set_grad_enabled(True)
    logging.info('#############training############')
    # iter_per_epoch = args.iter_per_epoch
    iter_per_epoch = len(train_loader)
    train_iter = iter(train_loader)
    for epoch in range(args.start_epoch, args.num_epochs):
        # setproctitle.setproctitle('{}: {}/{}'.format(args.user, epoch + 1, args.num_epochs))
        setproctitle.setproctitle('{}'.format(args.user))
        step_lr = lr_schedule(optimizer, epoch)
        writer.add_scalar('lr', step_lr, global_step=(epoch + 1))
        b = time.time()
        for i in range(iter_per_epoch):
            step = (i + 1) + epoch * iter_per_epoch
            ###Data load
            try:
                data = next(train_iter)
            except:
                train_iter = iter(train_loader)
                data = next(train_iter)
            x, target, mask = data[:3]
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)

            model.module.is_training = True
            fuse_pred, pred_all, preds, flair, t1ce, t1, t2, all, aux = model(x, mask)

            ###Loss compute
            fuse_cross_loss = criterions.softmax_weighted_loss(fuse_pred, target, num_cls=num_cls)
            fuse_dice_loss = criterions.dice_loss(fuse_pred, target, num_cls=num_cls)
            fuse_loss = fuse_cross_loss + fuse_dice_loss

            all_cross_loss = criterions.softmax_weighted_loss(pred_all, target, num_cls=num_cls)
            all_dice_loss = criterions.dice_loss(pred_all, target, num_cls=num_cls)
            all_loss = all_cross_loss + all_dice_loss

            sep_cross_loss = torch.zeros(1).cuda().float()
            sep_dice_loss = torch.zeros(1).cuda().float()
            for sep_pred in preds:
                sep_cross_loss += criterions.softmax_weighted_loss(sep_pred, target, num_cls=num_cls)
                sep_dice_loss += criterions.dice_loss(sep_pred, target, num_cls=num_cls)
            sep_loss = sep_cross_loss + sep_dice_loss

            loss_flair = torch.zeros(1).cuda().float()
            loss_t1ce = torch.zeros(1).cuda().float()
            loss_t1 = torch.zeros(1).cuda().float()
            loss_t2 = torch.zeros(1).cuda().float()
            for stage in range(3):
                loss_flair += criterions.l1loss(all[stage+2], flair[stage+2])
                loss_t1ce += criterions.l1loss(all[stage+2], t1ce[stage+2])
                loss_t1 += criterions.l1loss(all[stage+2], t1[stage+2])
                loss_t2 += criterions.l1loss(all[stage+2], t2[stage+2])
            loss_all = loss_flair + loss_t1ce + loss_t1 + loss_t2

            mod_cross_loss = torch.zeros(1).cuda().float()
            mod_dice_loss = torch.zeros(1).cuda().float()
            for mod in aux:
                mod_cross_loss += criterions.softmax_weighted_loss(mod, target, num_cls=num_cls)
                mod_dice_loss += criterions.dice_loss(mod, target, num_cls=num_cls)
            mod_loss = mod_cross_loss + mod_dice_loss

            loss = fuse_loss + all_loss + sep_loss + loss_all + mod_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ###log
            writer.add_scalar('loss', loss.item(), global_step=step)
            writer.add_scalar('fuse_cross_loss', fuse_cross_loss.item(), global_step=step)
            writer.add_scalar('fuse_dice_loss', fuse_dice_loss.item(), global_step=step)
            writer.add_scalar('all_cross_loss', all_cross_loss.item(), global_step=step)
            writer.add_scalar('all_dice_loss', all_dice_loss.item(), global_step=step)
            writer.add_scalar('sep_cross_loss', sep_cross_loss.item(), global_step=step)
            writer.add_scalar('sep_dice_loss', sep_dice_loss.item(), global_step=step)
            writer.add_scalar('loss_l1', loss_all.item(), global_step=step)
            writer.add_scalar('mod_cross_loss', mod_cross_loss.item(), global_step=step)
            writer.add_scalar('mod_dice_loss', mod_dice_loss.item(), global_step=step)

            msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch + 1), args.num_epochs, (i + 1), iter_per_epoch,
                                                                  loss.item())
            msg += 'fusecross:{:.4f}, fusedice:{:.4f},'.format(fuse_cross_loss.item(), fuse_dice_loss.item())
            msg += 'allcross:{:.4f}, alldice:{:.4f},'.format(all_cross_loss.item(), all_dice_loss.item())
            msg += 'sepcross:{:.4f}, sepdice:{:.4f},'.format(sep_cross_loss.item(), sep_dice_loss.item())
            msg += 'modcross:{:.4f}, moddice:{:.4f},'.format(mod_cross_loss.item(), mod_dice_loss.item())
            msg += 'l1:{:.4f},'.format(loss_all.item())
            logging.info(msg)
        logging.info('train time per epoch: {}'.format(time.time() - b))

        ##########model save
        file_name = os.path.join(ckpts, 'model_last.pth')
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
        },
            file_name)

        if (epoch + 1) % 20 == 0 or (epoch >= (args.num_epochs - 10)):
            file_name = os.path.join(ckpts, 'model_{}.pth'.format(epoch + 1))
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
            },
                file_name)

    msg = 'total time: {:.4f} hours'.format((time.time() - start) / 3600)
    logging.info(msg)

    ##########Evaluate the last epoch model
    test_score = AverageMeter()
    with torch.no_grad():
        logging.info('###########test set wi/wo postprocess###########')
        for i, mask in enumerate(masks):
            logging.info('{}'.format(mask_name[i]))
            dice_score = test_softmax(
                test_loader,
                model,
                dataname=args.dataname,
                feature_mask=mask)
            test_score.update(dice_score)
        logging.info('Avg scores: {}'.format(test_score.avg))


if __name__ == '__main__':
    main()
