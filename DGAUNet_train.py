import os
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from alive_progress import alive_bar
from torch import nn

from torch.utils.data import DataLoader
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations import RandomRotate90, Resize, Flip

from src.dataloader.dataset import MedicalDataSets
from src.network.New.DGAUNet import  DGAUNet

from src.network.New.DGAUNet_one_encoder import DGAUNet_one_encoder
from src.network.conv_based.AttU_Net import AttU_Net
from src.network.conv_based.CMUNeXt import cmunext
from src.network.conv_based.CMUNet import CMUNet
from src.network.conv_based.UNeXt import UNext
from src.network.conv_based.UNet3plus import UNet3plus
from src.network.conv_based.UNetplus import ResNet34UnetPlus
from src.network.conv_based.U_Net import U_Net
from src.network.New.NewU_Net_skip1 import NewU_Net_skip1
from src.network.transfomer_based.transformer_based_network import get_transformer_based_model
from src.utils import losses
from src.utils.kd_loss import MSELoss, DistillKL
from src.utils.metrics import iou_score
from src.utils.util import AverageMeter


def seed_torch(seed):
    if seed==None:
        seed= random.randint(1, 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="DGAUNet", help='model')
parser.add_argument('--base_dir', type=str, default="./data", help='dir')
parser.add_argument('--train_file_dir', type=str, default="GCPS_train.txt", help='dir')
parser.add_argument('--val_file_dir', type=str, default="GCPS_val.txt", help='dir')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--epoch', type=int, default=150, help='train epoch')
parser.add_argument('--img_size', type=int, default=256, help='img size of per batch')
parser.add_argument('--num_classes', type=int, default=1, help='seg num_classes')
parser.add_argument('--seed', type=int, default=32,  help='random seed')
args = parser.parse_args()
seed_torch(args.seed)


def get_model(args):
    if args.model == "NewU_Net_skip1":
        model = NewU_Net_skip1(output_ch=args.num_classes).cuda()
    elif args.model == "DGAUNet":
        model = DGAUNet(output_ch=args.num_classes).cuda()
    elif args.model == "DGAUNet_one_encoder":
        model = DGAUNet_one_encoder(output_ch=args.num_classes).cuda()
    return model


def getDataloader(args):
    img_size = args.img_size
    if args.model == "SwinUnet":
        img_size = 224
    train_transform = Compose([
        RandomRotate90(),
        # transforms.Flip(),
        Flip(),
        Resize(img_size, img_size),
        # transforms.Normalize(),
    ])

    val_transform = Compose([
        Resize(img_size, img_size),
        # transforms.Normalize(),
    ])
    db_train = MedicalDataSets(base_dir=args.base_dir, split="train",
                               transform=train_transform, train_file_dir=args.train_file_dir,
                               val_file_dir=args.val_file_dir)
    db_val = MedicalDataSets(base_dir=args.base_dir, split="val", transform=val_transform,
                             train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir)
    print("train num:{}, val num:{}".format(len(db_train), len(db_val)))

    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    valloader = DataLoader(db_val, batch_size=args.batch_size, shuffle=False, num_workers=0)

    return trainloader, valloader


def main(args):
    for folds in [0,1]:
        args.train_file_dir = 'GCPS{}_train.txt'.format(folds)
        args.val_file_dir = 'GCPS{}_val.txt'.format(folds)
        # train_result = open(os.path.join("./train_result", 'total', '{}_train_resultFinal_v9.txt'.format(args.model)), 'w')
        train_result = open(os.path.join("./train_result", 'total', '{}_train_result{}.txt'.format(args.model,folds)), 'w')

    # train_result = open(os.path.join("./train_result",'total', '{}_train_resultFinal.txt'.format(args.model)), 'w')

        base_lr = args.base_lr
        trainloader, valloader = getDataloader(args)
        model = get_model(args)
        print("train file dir:{} val file dir:{}".format(args.train_file_dir, args.val_file_dir))
        optimizer_seg = optim.SGD(model.get_parameters(net="seg_net"), lr=base_lr, momentum=0.9, weight_decay=0.0001)
        optimizer_l = optim.SGD(model.get_parameters(net="l"), lr=base_lr, momentum=0.9, weight_decay=0.0001)
        # optimizer_l = torch.optim.Adam(model.get_parameters(net="l"),lr=0.01, betas=(0.5, 0.999))
        criterion = losses.__dict__['BCEDiceLoss']().cuda()
        criterion1 = MSELoss().cuda()
        criterion2 = DistillKL(T=1).cuda()
        print("{} iterations per epoch".format(len(trainloader)))

        best_iou = 0
        iter_num = 0
        max_epoch = args.epoch

        max_iterations = len(trainloader) * max_epoch

        total_avg_meters = {'loss_label': AverageMeter(),
                            'loss': AverageMeter(),
                            's_loss': AverageMeter(),
                            'iou': AverageMeter(),
                            'val_loss': AverageMeter(),
                            'val_iou': AverageMeter(),
                            'val_SE': AverageMeter(),
                            'val_PC': AverageMeter(),
                            'val_F1': AverageMeter(),
                            'val_Dice': AverageMeter(),
                            'val_ACC': AverageMeter()
                            }
        for epoch_num in range(max_epoch):
            model.train()
            avg_meters = {'loss_label': AverageMeter(),
                            'loss': AverageMeter(),
                          's_loss': AverageMeter(),
                          'iou': AverageMeter(),
                          'val_loss': AverageMeter(),
                          'val_iou': AverageMeter(),
                          'val_SE': AverageMeter(),
                          'val_PC': AverageMeter(),
                          'val_F1': AverageMeter(),
                          'val_Dice': AverageMeter(),
                          'val_ACC': AverageMeter()}
            with alive_bar(len(trainloader) + len(valloader), force_tty=True,
                           title="epoch %d/%d" % (epoch_num + 1, max_epoch)) as bar:
                for i_batch, sampled_batch in enumerate(trainloader):
                    img_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                    img_batch, label_batch = img_batch.cuda(), label_batch.cuda()

                    # output, s_e_out,s_skip, t_e_out, t_skip = model(img_batch, img_batch + label_batch)
                    # output, s_e_out, t_e_out, t_skip ,s_skip= model(img_batch, img_batch + label_batch)
                    #
                    # # skip_loss1=criterion1(s_skip[0],t_skip[0].detach())
                    # # skip_loss2=criterion1(s_skip[1],t_skip[1].detach())
                    # # skip_loss3=criterion1(s_skip[2],t_skip[2].detach())
                    # # skip_loss4=criterion1(s_skip[3],t_skip[3].detach())
                    # # all_skiploss=(skip_loss1+skip_loss2+skip_loss3+skip_loss4)/4
                    # #
                    # # e_studyloss=0.6*criterion1(s_e_out,t_e_out.detach())+0.4*all_skiploss
                    # e_studyloss=criterion1(s_e_out,t_e_out.detach())
                    #
                    # s_seg_loss=criterion(output, label_batch)
                    #
                    # iou, dice, _, _, _, _, _ = iou_score(output, label_batch)
                    # x_seg_loss = 0.85 * s_seg_loss + 0.15  * e_studyloss
                    # optimizer_seg.zero_grad()
                    # x_seg_loss.backward()
                    # optimizer_seg.step()
                    #
                    # pre_x_with_label = model.pre_x_with_label(t_e_out, t_skip)
                    # loss_label = criterion(pre_x_with_label, label_batch)
                    # l_iou, _, _, _, _, _, _ = iou_score(pre_x_with_label, label_batch)
                    #
                    # optimizer_l.zero_grad()
                    # loss_label.backward()
                    # optimizer_l.step()


                    r_out,r_e_out=model.get_R_out(img_batch+label_batch)
                    loss_label=criterion(r_out,label_batch)
                    l_iou, _, _, _, _, _, _ = iou_score(r_out, label_batch)
                    optimizer_l.zero_grad()
                    loss_label.backward()
                    optimizer_l.step()

                    l_out,l_e_out=model.get_L_out(img_batch)
                    e_studyloss=criterion1(l_e_out,r_e_out.detach())
                    s_seg_loss = criterion(l_out, label_batch)
                    iou, dice, _, _, _, _, _ = iou_score(l_out, label_batch)
                    x_seg_loss = 0.9 * s_seg_loss + 0.1 * e_studyloss
                    optimizer_seg.zero_grad()
                    x_seg_loss.backward()
                    optimizer_seg.step()

                    lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                    for param_group in optimizer_seg.param_groups:
                        param_group['lr'] = lr_
                    for param_group in optimizer_l.param_groups:
                        param_group['lr'] = lr_

                    iter_num = iter_num + 1
                    avg_meters['loss_label'].update(loss_label.item(), img_batch.size(0))
                    avg_meters['loss'].update(s_seg_loss.item(), img_batch.size(0))
                    avg_meters['s_loss'].update(e_studyloss.item(), img_batch.size(0))
                    avg_meters['iou'].update(iou, img_batch.size(0))
                    bar()

                model.eval()
                with torch.no_grad():
                    for i_batch, sampled_batch in enumerate(valloader):
                        img_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                        img_batch, label_batch = img_batch.cuda(), label_batch.cuda()
                        output = model.val(img_batch)
                        loss = criterion(output, label_batch)
                        iou, Dice, SE, PC, F1, _, ACC = iou_score(output, label_batch)
                        avg_meters['val_loss'].update(loss.item(), img_batch.size(0))
                        avg_meters['val_iou'].update(iou, img_batch.size(0))
                        avg_meters['val_SE'].update(SE, img_batch.size(0))
                        avg_meters['val_PC'].update(PC, img_batch.size(0))
                        avg_meters['val_F1'].update(F1, img_batch.size(0))
                        avg_meters['val_Dice'].update(Dice, img_batch.size(0))
                        avg_meters['val_ACC'].update(ACC, img_batch.size(0))
                        bar()
                total_avg_meters['loss_label'].update(avg_meters['loss_label'].avg, 1)
                total_avg_meters['loss'].update(avg_meters['loss'].avg, 1)
                total_avg_meters['s_loss'].update(avg_meters['s_loss'].avg, 1)
                total_avg_meters['iou'].update(avg_meters['iou'].avg, 1)
                total_avg_meters['val_loss'].update(avg_meters['val_loss'].avg, 1)
                total_avg_meters['val_iou'].update(avg_meters['val_iou'].avg, 1)
                total_avg_meters['val_SE'].update(avg_meters['val_SE'].avg, 1)
                total_avg_meters['val_PC'].update(avg_meters['val_F1'].avg, 1)
                total_avg_meters['val_F1'].update(avg_meters['val_F1'].avg, 1)
                total_avg_meters['val_Dice'].update(avg_meters['val_Dice'].avg, 1)
                total_avg_meters['val_ACC'].update(avg_meters['val_ACC'].avg, 1)

                print(
                    'epoch [%d/%d],label_loss : %.4f,  train_loss : %.4f,study_loss : %.4f, train_iou: %.4f , val_loss %.4f - '
                    'val_iou %.4f -'
                    'val_SE %.4f -'
                    'val_PC %.4f - val_F1 %.4f - val_ACC %.4f - val_Dice %.4f'
                    % (epoch_num, max_epoch,avg_meters['loss_label'].avg, avg_meters['loss'].avg,avg_meters['s_loss'].avg, avg_meters['iou'].avg,
                       avg_meters['val_loss'].avg, avg_meters['val_iou'].avg, avg_meters['val_SE'].avg,
                       avg_meters['val_PC'].avg, avg_meters['val_F1'].avg, avg_meters['val_ACC'].avg, avg_meters['val_Dice'].avg))
                train_result.write(
                    'epoch [%d/%d],%.4f,  %.4f,%.4f,  %.4f ,  %.4f ,  %.4f ,  %.4f ,   %.4f '
                    ' %.4f ,  %.4f ,  %.4f '
                    % (epoch_num, max_epoch,avg_meters['loss_label'].avg, avg_meters['loss'].avg,avg_meters['s_loss'].avg, avg_meters['iou'].avg,
                       avg_meters['val_loss'].avg, avg_meters['val_iou'].avg, avg_meters['val_SE'].avg,
                       avg_meters['val_PC'].avg, avg_meters['val_F1'].avg, avg_meters['val_ACC'].avg, avg_meters['val_Dice'].avg) + '\n')
                train_result.flush()

                if avg_meters['val_iou'].avg > best_iou:
                    if not os.path.isdir("./checkpoint"):
                        os.makedirs("./checkpoint")
                    torch.save(model.state_dict(), 'checkpoint/{}_model{}.pth'.format(args.model,folds))
                    best_iou = avg_meters['val_iou'].avg
                    print("=> saved best model")

        print('AVE , train_loss : %.4f, train_iou: %.4f - val_loss %.4f - val_iou %.4f - val_SE %.4f - '
              'val_PC %.4f - val_F1 %.4f - val_ACC %.4f '
              % (total_avg_meters['loss'].avg, total_avg_meters['iou'].avg,
                 total_avg_meters['val_loss'].avg, total_avg_meters['val_iou'].avg, total_avg_meters['val_SE'].avg,
                 total_avg_meters['val_PC'].avg, total_avg_meters['val_F1'].avg, total_avg_meters['val_ACC'].avg))
        train_result.write('AVE  ,  %.4f,  %.4f , %.4f , %.4f ,  %.4f , '
                           ' %.4f ,  %.4f ,  %.4f ,  %.4f '
                           % (total_avg_meters['loss'].avg, total_avg_meters['iou'].avg,
                              total_avg_meters['val_loss'].avg, total_avg_meters['val_iou'].avg,
                              total_avg_meters['val_SE'].avg,
                              total_avg_meters['val_Dice'].avg,
                              total_avg_meters['val_PC'].avg, total_avg_meters['val_F1'].avg,
                              total_avg_meters['val_ACC'].avg))
        train_result.flush()

    return "Training Finished!"


if __name__ == "__main__":
    main(args)
