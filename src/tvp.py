from tqdm import tqdm
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from . import config
from .common.logger import get_logger
from .common.util import str_stats
from .helper_func import train_valid_split_v1
from .dataset import CellerDataset, alb_trn_trnsfms, alb_val_trnsfms
from .model.model import ResNet
from .model.metrics import accuracy, AverageMeter
from .model.loss import FocalLoss
from .model.model_util import load_checkpoint, save_checkpoint


def train_one_epoch(epoch,
                    model,
                    loader,
                    criterion,
                    optimizer):
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    get_logger().info('[Start] epoch: %d' % epoch)
    get_logger().info('lr: %f' %
                      optimizer.state_dict()['param_groups'][0]['lr'])
    loader.dataset.update()
    # train phase
    model.train()
    for i, data in enumerate(tqdm(loader)):
        img, label = data
        img, label = img.cuda(), label.cuda().long()
        with torch.set_grad_enabled(True):
            logit = model(img, label)
            loss = criterion(logit, label.squeeze())

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), img.size(0))
            score_top1, score_top5 = accuracy(
                logit.detach(), label.detach(), topk=(1, 5))
            top1_meter.update(score_top1.item(), img.size(0))
            top5_meter.update(score_top5.item(), img.size(0))

        # print
        if i % config.PRINT_FREQ == 0:
            logit_cpu = logit.view(img.size(0), -1).detach().cpu()
            get_logger().info('\n' + str_stats(logit_cpu[0].numpy()))
            softmaxed = F.softmax(logit_cpu, dim=1)
            get_logger().info('\n' + str_stats(softmaxed[0].numpy()))
            get_logger().info('train: %d loss: %f top1: %f top5: %f (just now)' %
                              (i, loss_meter.val, top1_meter.val, top5_meter.val))
            get_logger().info('train: %d loss: %f top1: %f top5: %f' %
                              (i, loss_meter.avg, top1_meter.avg, top5_meter.avg))

    get_logger().info("Epoch %d/%d train loss %f top1 %f top5 %f"
                      % (epoch, config.EPOCHS, loss_meter.avg, top1_meter.avg, top5_meter.avg))

    return loss_meter.avg, top1_meter.avg, top5_meter.avg


def validate_one_epoch(epoch,
                       model,
                       loader,
                       criterion):
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    # validate phase
    model.eval()
    for i, data in enumerate(tqdm(loader)):
        img, label = data
        img, label = img.cuda(), label.cuda().long()
        with torch.no_grad():
            logit = model(img)

            loss = criterion(logit, label.squeeze())
            loss_meter.update(loss.item(), img.size(0))
            score_top1, score_top5 = accuracy(
                logit.detach(), label, topk=(1, 5))
            top1_meter.update(score_top1.item(), img.size(0))
            top5_meter.update(score_top5.item(), img.size(0))

    get_logger().info("Epoch %d/%d valid loss %f top1 %f top5 %f"
                      % (epoch, config.EPOCHS, loss_meter.avg, top1_meter.avg, top5_meter.avg))

    return loss_meter.avg, top1_meter.avg, top5_meter.avg


def train():
    # Load csv
    df_train = pd.read_csv(config.TRAIN_PATH)
    df_trn, df_val = train_valid_split_v1(df_train, valid_ratio=0.05)
    get_logger().info('train size: %d valid size: %d' % (len(df_trn), len(df_val)))

    train_dataset = CellerDataset(
        df_trn, config.TRAIN_IMG_PATH, alb_trn_trnsfms, mode='train')
    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE_TRAIN,
                              num_workers=config.NUM_WORKERS,
                              pin_memory=True,
                              drop_last=True,
                              shuffle=True
                              )
    valid_dataset = CellerDataset(
        df_val, config.TRAIN_IMG_PATH, alb_val_trnsfms, mode='valid')
    valid_loader = DataLoader(valid_dataset,
                              batch_size=config.BATCH_SIZE_TRAIN,
                              num_workers=config.NUM_WORKERS,
                              pin_memory=True,
                              drop_last=False,
                              shuffle=False
                              )
    model, criterion, optimizer, scheduler = init_model()
    start_epoch = 0
    if config.USE_PRETRAINED:
        start_epoch, model, optimizer, scheduler, _ = load_checkpoint(
            model, optimizer, scheduler, config.PRETRAIN_PATH)

    get_logger().info('[Start] Recursion Celler Training')
    best_score = 0
    train_history = {'loss': [], 'top1': [], 'top5': []}
    valid_history = {'loss': [], 'top1': [], 'top5': []}
    for epoch in range(start_epoch + 1, config.EPOCHS + 1):
        train_loss, train_top1, train_top5 = train_one_epoch(
            epoch, model, train_loader, criterion, optimizer)
        valid_loss, valid_top1, valid_top5 = validate_one_epoch(
            epoch, model, valid_loader, criterion)

        train_history['loss'].append(train_loss)
        train_history['top1'].append(train_top1)
        train_history['top5'].append(train_top1)
        valid_history['loss'].append(valid_loss)
        valid_history['top1'].append(valid_top1)
        valid_history['top5'].append(valid_top5)

        valid_score = valid_top1
        is_best = valid_score > best_score
        if is_best:
            best_score = valid_score
        get_logger().info('best score (%f) at epoch (%d)' % (valid_score, epoch))
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, is_best)
        # move scheduler.step to here
        scheduler.step()

    return train_history, valid_history


def init_model():
    torch.backends.cudnn.benchmark = True
    get_logger().info('Initializing classification model...')
    model = ResNet(dropout_rate=config.DROPOUT_RATE,
                   latent_dim=config.LATENT_DIM,
                   temperature=config.TEMPERATURE,
                   m=config.MARGIN).cuda()
    # model = SEResNet(dropout_rate=0.3, latent_dim=512, temperature=1.0, m=MARGIN).cuda()
    criterion = FocalLoss(gamma=2).cuda()
    '''
    optimizer = torch.optim.SGD([{'params': model.parameters()}],
                                lr=SGD_LR,
                                momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, EPOCHS, MIN_LR)
    '''
    optimizer = optim.Adam([{'params': model.parameters()}], lr=config.ADAM_LR)
    mile_stones = [10, 20, 30, 40]
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, mile_stones, gamma=0.5, last_epoch=-1)

    return model, criterion, optimizer, scheduler
