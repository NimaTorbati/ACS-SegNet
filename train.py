# python train.py --base_dir ./data --train_file_dir GCPS_train.txt --val_file_dir GCPS_val.txt --base_lr 0.01 --epoch 150 --batch_size 8 --model TransUnet
import os
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from alive_progress import alive_bar
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from torch.utils.data import DataLoader
from albumentations.core.composition import Compose
from albumentations import RandomRotate90, Resize, Flip
from src.dataloader.dataset import MedicalDataSets
from src.utils import losses
from src.utils.metrics import iou_score
from src.utils.util import AverageMeter
from model import DualEncoderUNet
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import segmentation_models_pytorch as smp
import torch.nn.functional as F
def adapt_checkpoint(checkpoint, model):
    model_dict = model.state_dict()
    new_checkpoint = {}
    for key, value in checkpoint.items():
        # print(f"Skipping {key} due to shape mismatch")
        if key in model_dict and model_dict[key].shape == value.shape:
            new_checkpoint[key] = value
        else:
            print(f"Skipping {key} due to shape mismatch")
    return new_checkpoint


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
parser.add_argument('--model', type=str, default="ACSSegNet",
                    choices=["TransUnet", "ACSSegNet", "segformer", "ResnetUnet"], help='model')
parser.add_argument('--base_dir', type=str, default="./data", help='dir')
parser.add_argument('--train_file_dir', type=str, default="GCPS_train.txt", help='dir')
parser.add_argument('--val_file_dir', type=str, default="GCPS_val.txt", help='dir')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--epoch', type=int, default=150, help='train epoch')
parser.add_argument('--img_size', type=int, default=256, help='img size of per batch')
parser.add_argument('--num_classes', type=int, default=1, help='seg num_classes')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--variant', type=str, default='1', help='random seed')
parser.add_argument('--folds', type=int, default=None)
args = parser.parse_args()
seed_torch(args.seed)


def get_model(args):
    if args.model == "ACSSegNet":
        variant = int(args.variant)
        fine_tune = False
        # decoder_channels = (512, 256, 128, 64,32)
        IgnoreBottleNeck = False
        segformer_variant = "nvidia/segformer-b2-finetuned-ade-512-512"
        model = DualEncoderUNet(
            segformer_variant=segformer_variant,
            simple_fusion=variant,
            regression=False,
            classes=args.num_classes,
            in_channels=3,
            unet_encoder_weights="imagenet",
            unet_encoder_name="resnet34",
            IgnoreBottleNeck=IgnoreBottleNeck,
            decoder_channels=(256, 128, 64, 32, 16),
            model_depth = 5,
        ).cuda()

    elif args.model == "segformer":
        segformer_variant = "nvidia/segformer-b2-finetuned-ade-512-512"
        config = SegformerConfig.from_pretrained(segformer_variant)

        # Modify the configuration to match your dataset
        config.num_channels = 3
        config.num_labels = args.num_classes  # Set the number of segmentation classes
        config.image_size = 256  # Ensure input image size is 1024x1024

        # Initialize the model (without pretrained weights)
        model = SegformerForSemanticSegmentation(config).cuda()
    elif args.model == "ResnetUnet":
        model = smp.Unet(classes=args.num_classes, in_channels=3).cuda()

    elif args.model == "TransUnet":
        config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]  # R50-ViT-B_16
        config_vit.n_classes = args.num_classes
        config_vit.n_skip = 3
        config_vit.patches.grid = (int(256 / 16), int(256 / 16))
        model = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
        model.load_from(weights=np.load(config_vit.pretrained_path))
    else:
        print('model error')
        return None
    return model


def getDataloader(args):
    img_size = args.img_size
    train_transform = Compose([
        RandomRotate90(),
        Flip(),
        Resize(img_size, img_size),
    ])

    val_transform = Compose([
        Resize(img_size, img_size),
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
    os.makedirs(os.path.join("./train_result", 'total'), exist_ok=True)
    for folds in [0,1,2]:
        args.folds = folds
        args.train_file_dir = 'GCPS{}_train.txt'.format(folds)
        args.val_file_dir = 'GCPS{}_val.txt'.format(folds)
        train_result = open(os.path.join("./train_result", 'total', '{}_train_result_v1{}.txt'.format(args.model,folds)), 'w')
        base_lr = args.base_lr
        trainloader, valloader = getDataloader(args)
        model = get_model(args)
        print("train file dir:{} val file dir:{}".format(args.train_file_dir, args.val_file_dir))
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
        criterion = losses.__dict__['BCEDiceLoss']().cuda()
        print("{} iterations per epoch".format(len(trainloader)))
        best_iou = 0
        iter_num = 0
        max_epoch = args.epoch
        total_avg_meters = {'loss': AverageMeter(),
                            'iou': AverageMeter(),
                            'val_loss': AverageMeter(),
                            'val_iou': AverageMeter(),
                            'val_SE': AverageMeter(),
                            'val_PC': AverageMeter(),
                            'val_F1': AverageMeter(),
                            'val_ACC': AverageMeter(),
                            'val_Dice': AverageMeter(),
                            'val_SP': AverageMeter()

                            }
        for epoch_num in range(max_epoch):
            model.train()

            avg_meters = {'loss': AverageMeter(),
                          'iou': AverageMeter(),
                          'val_loss': AverageMeter(),
                          'val_iou': AverageMeter(),
                          'val_SE': AverageMeter(),
                          'val_PC': AverageMeter(),
                          'val_F1': AverageMeter(),
                          'val_ACC': AverageMeter(),
                          'val_Dice': AverageMeter(),
                          'val_SP': AverageMeter(),
                          }
            with alive_bar(len(trainloader) + len(valloader), force_tty=True,
                           title="epoch %d/%d" % (epoch_num + 1, max_epoch)) as bar:
                for i_batch, sampled_batch in enumerate(trainloader):
                    img_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                    img_batch, label_batch = img_batch.cuda(), label_batch.cuda()


                    if args.model == "segformer":
                        outputs = model(img_batch)
                        outputs = F.interpolate(outputs.logits, size=label_batch.size()[2:],
                                                   mode='bilinear', align_corners=False)

                    else:
                        outputs = model(img_batch)

                    loss = criterion(outputs, label_batch)
                    iou, dice, _, _, _, _, _ = iou_score(outputs, label_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    iter_num = iter_num + 1
                    avg_meters['loss'].update(loss.item(), img_batch.size(0))
                    avg_meters['iou'].update(iou, img_batch.size(0))
                    bar()

                model.eval()
                with torch.no_grad():
                    for i_batch, sampled_batch in enumerate(valloader):
                        img_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                        img_batch, label_batch = img_batch.cuda(), label_batch.cuda()
                        if args.model == "segformer":
                            output = model(img_batch)
                            output = F.interpolate(output.logits, size=label_batch.size()[2:],
                                                    mode='bilinear', align_corners=False)

                        else:
                            output = model(img_batch)
                        loss = criterion(output, label_batch)
                        iou, Dice, SE, PC, F1, SP, ACC = iou_score(output, label_batch)
                        avg_meters['val_loss'].update(loss.item(), img_batch.size(0))
                        avg_meters['val_iou'].update(iou, img_batch.size(0))
                        avg_meters['val_SE'].update(SE, img_batch.size(0))
                        avg_meters['val_PC'].update(PC, img_batch.size(0))
                        avg_meters['val_F1'].update(F1, img_batch.size(0))
                        avg_meters['val_ACC'].update(ACC, img_batch.size(0))
                        avg_meters['val_Dice'].update(Dice, img_batch.size(0))
                        avg_meters['val_SP'].update(SP, img_batch.size(0))
                        bar()

                total_avg_meters['loss'].update(avg_meters['loss'].avg, 1)
                total_avg_meters['iou'].update(avg_meters['iou'].avg, 1)
                total_avg_meters['val_loss'].update(avg_meters['val_loss'].avg, 1)
                total_avg_meters['val_iou'].update(avg_meters['val_iou'].avg, 1)
                total_avg_meters['val_SE'].update(avg_meters['val_SE'].avg, 1)
                total_avg_meters['val_PC'].update(avg_meters['val_PC'].avg, 1)
                total_avg_meters['val_F1'].update(avg_meters['val_F1'].avg, 1)
                total_avg_meters['val_ACC'].update(avg_meters['val_ACC'].avg, 1)
                total_avg_meters['val_Dice'].update(avg_meters['val_Dice'].avg, 1)
                total_avg_meters['val_SP'].update(avg_meters['val_SP'].avg, 1)

                print('epoch [%d/%d]  train_loss : %.4f, train_iou: %.4f - val_loss %.4f - val_iou %.4f - val_SE %.4f - '
                      'val_PC %.4f - val_F1 %.4f - val_ACC %.4f - val_Dice %.4f - val_SP %.4f'
                      % (epoch_num, max_epoch, avg_meters['loss'].avg, avg_meters['iou'].avg,
                         avg_meters['val_loss'].avg, avg_meters['val_iou'].avg, avg_meters['val_SE'].avg,
                         avg_meters['val_PC'].avg, avg_meters['val_F1'].avg, avg_meters['val_ACC'].avg,
                         avg_meters['val_Dice'].avg, avg_meters['val_SP'].avg))
                train_result.write('epoch [%d/%d],    %.4f,  %.4f ,  %.4f ,  %.4f ,  %.4f , %.4f ,  %.4f'
                                   ' %.4f ,  %.4f ,  %.4f '
                                   % (epoch_num, max_epoch, avg_meters['loss'].avg, avg_meters['iou'].avg,
                                      avg_meters['val_loss'].avg, avg_meters['val_iou'].avg, avg_meters['val_SE'].avg,
                                      avg_meters['val_PC'].avg, avg_meters['val_F1'].avg, avg_meters['val_ACC'].avg,
                                      avg_meters['val_Dice'].avg, avg_meters['val_SP'].avg) + '\n')
                train_result.flush()

                if avg_meters['val_iou'].avg > best_iou:
                    if not os.path.isdir("./checkpoint"):
                        os.makedirs("./checkpoint")
                    torch.save(model.state_dict(), 'checkpoint/{}_model{}_v1.pth'.format(args.model,folds))
                    best_iou = avg_meters['val_iou'].avg
                    print("=> saved best model")
        print('AVE , train_loss : %.4f, train_iou: %.4f - val_loss %.4f - val_iou %.4f - val_SE %.4f - '
              'val_PC %.4f - val_F1 %.4f - val_ACC %.4f '
              % (total_avg_meters['loss'].avg, total_avg_meters['iou'].avg,
                 total_avg_meters['val_loss'].avg, total_avg_meters['val_iou'].avg, total_avg_meters['val_SE'].avg,
                 total_avg_meters['val_PC'].avg, total_avg_meters['val_F1'].avg, total_avg_meters['val_ACC'].avg))
        train_result.write('AVE  ,  %.4f,  %.4f , %.4f , %.4f ,  %.4f , '
                           ' %.4f ,  %.4f ,  %.4f ,  %.4f ,  %.4f '
                           % (total_avg_meters['loss'].avg, total_avg_meters['iou'].avg,
                              total_avg_meters['val_loss'].avg, total_avg_meters['val_iou'].avg,
                              total_avg_meters['val_SE'].avg,
                              total_avg_meters['val_PC'].avg, total_avg_meters['val_F1'].avg,
                              total_avg_meters['val_ACC'].avg,
                              total_avg_meters['val_Dice'].avg,
                              total_avg_meters['val_SP'].avg,
                              ))
        train_result.flush()

    return "Training Finished!"


if __name__ == "__main__":
    for models in ["ACSSegNet", "ResnetUnet","TransUnet", "segformer"]:
        print(models)
        args.model = models
        main(args)
