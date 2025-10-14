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
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations import RandomRotate90, Resize, Flip
from transformers import SegformerImageProcessor
from src.dataloader.dataset import MedicalDataSets
# from src.network.New.DGAUNet_one_encoder import DGAUNet_one_encoder
from src.network.conv_based.AttU_Net import AttU_Net
from src.network.conv_based.CGNet import Context_Guided_Network
from src.network.conv_based.CMUNeXt import cmunext
from src.network.conv_based.CMUNet import CMUNet
from src.network.conv_based.ELUnet import ELUnet
from src.network.conv_based.ULite import ULite
from src.network.conv_based.UNeXt import UNext
from src.network.conv_based.UNet3plus import UNet3plus
from src.network.conv_based.UNetplus import ResNet34UnetPlus
from src.network.conv_based.U_Net import U_Net
from src.network.conv_based.cfpnet import CFPNet
from src.network.conv_based.pp_liteseg import PPLiteSeg
from src.network.transfomer_based.transformer_based_network import get_transformer_based_model
from src.utils import losses
from src.utils.metrics import iou_score
from src.utils.util import AverageMeter
from model import DualEncoderUNet
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, SegformerConfig
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from model_TransSegUnet import DualEncoderUNet as DualEncoderTranUNet
from utils.utils import adapt_checkpoint_dualEncoder
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
parser.add_argument('--model', type=str, default="ours",
                    choices=["CMUNeXt", "CMUNet", "AttU_Net", "TransUnet", "R2U_Net", "U_Net","DGAUNet_one_encoder","ELUnet","CGNet","CFPNet","PPLiteSeg", "ULite",
                             "UNext", "UNetplus", "UNet3plus", "SwinUnet", "MedT", "TransUnet", "ours","ours_s","ours_n", "segformer", "ResnetUnet", "oursTrans"], help='model')
parser.add_argument('--base_dir', type=str, default="./data", help='dir')
parser.add_argument('--train_file_dir', type=str, default="GCPS_train.txt", help='dir')
parser.add_argument('--val_file_dir', type=str, default="GCPS_val.txt", help='dir')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--epoch', type=int, default=150, help='train epoch')
parser.add_argument('--img_size', type=int, default=256, help='img size of per batch')
parser.add_argument('--num_classes', type=int, default=1, help='seg num_classes')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--variant', type=str, default='11', help='random seed')
parser.add_argument('--folds', type=int, default=None)
args = parser.parse_args()
seed_torch(args.seed)


def get_model(args):
    if args.model == "CMUNet":
        model = CMUNet(output_ch=args.num_classes).cuda()
    elif args.model == "CMUNeXt":
        model = cmunext(num_classes=args.num_classes).cuda()
        # model.load_state_dict(torch.load('checkpoint/CMUNeXt_model.pth'))
    elif args.model == "U_Net":
        model = U_Net(output_ch=args.num_classes).cuda()
    elif args.model == "AttU_Net":
        model = AttU_Net(output_ch=args.num_classes).cuda()
    elif args.model == "UNext":
        model = UNext(output_ch=args.num_classes).cuda()
    elif args.model == ("UNetplus"):
        model = ResNet34UnetPlus(num_class=args.num_classes).cuda()
    elif args.model == "UNet3plus":
        model = UNet3plus(n_classes=args.num_classes).cuda()
    elif args.model == "DGAUNet_one_encoder":
        model = DGAUNet_one_encoder(output_ch=args.num_classes).cuda()
    elif args.model == "ELUnet":
        model = ELUnet().cuda()
    elif args.model == "ULite":
        model = ULite().cuda()
    elif args.model == "CGNet":
        model = Context_Guided_Network().cuda()
    elif args.model == "PPLiteSeg":
        model = PPLiteSeg().cuda()
    elif args.model == "CFPNet":
        model = CFPNet().cuda()
    elif args.model == "ours" or args.model == "ours_weit" or args.model == "ours_n" or args.model == "ours_s":
        # variant = int(args.variant)
        if args.model == "ours_s":
            variant = 0
            cof_unet = 0
            cof_seg = 1
        elif args.model == "ours_n":
            variant = 0
            cof_unet = 1
            cof_seg = 0
        else:
            variant = int(args.variant)
            cof_unet = 1
            cof_seg = 1

        fine_tune = False
        # decoder_channels = (512, 256, 128, 64,32)
        IgnoreBottleNeck = False
        segformer_variant = "nvidia/segformer-b2-finetuned-ade-512-512"
        print('B3333')
        model = DualEncoderUNet(
            m=256,
            segformer_variant=segformer_variant,
            simple_fusion=variant,
            regression=False,
            classes=args.num_classes,
            in_channels=3,
            unet_encoder_weights="imagenet",
            unet_encoder_name="resnet34",
            # decoder_type="unet++",
            IgnoreBottleNeck=IgnoreBottleNeck,
            # model_depth=model_depth,
            decoder_channels=(256, 128, 64, 32, 16),
            # model_depth = 4,
            cof_unet=cof_unet,
            cof_seg=cof_seg,
        ).cuda()

        if args.model == 'ours_s':
            for seg_temp in model.segformer.parameters():
                seg_temp.requires_grad = True
            for unet_temp in model.unet_encoder.parameters():
                unet_temp.requires_grad = False

        if args.model == 'ours_n':
            for seg_temp in model.segformer.parameters():
                seg_temp.requires_grad = False
            for unet_temp in model.unet_encoder.parameters():
                unet_temp.requires_grad = True



        cp = adapt_checkpoint_dualEncoder(model, args.folds, dg = True)
        model.load_state_dict(cp, strict=False)  # strict=True to ensure all trained layers match

        for seg_temp in model.segformer.parameters():
            seg_temp.requires_grad = False

        for unet_temp in model.unet_encoder.parameters():
            unet_temp.requires_grad = False

        if fine_tune:
            fineTune_PATH = '/home/ntorbati/PycharmProjects/SegformerPlusUnet/TissueExpb2_v0_preTrainsegformerUnet1segformerUnet0/checkpoint_epoch1.pth'
            cp = torch.load(fineTune_PATH, weights_only=True)
            if "module." in list(cp.keys())[0]:
                cp = {k.replace("module.", ""): v for k, v in cp.items()}

            cp = adapt_checkpoint(cp, model)
            model.load_state_dict(cp, strict=False)  # strict=True to ensure all trained layers match
    elif args.model == "segformer":
        segformer_variant = "nvidia/segformer-b2-finetuned-ade-512-512"
        config = SegformerConfig.from_pretrained(segformer_variant)

        # Modify the configuration to match your dataset
        config.num_channels = 3
        config.num_labels = args.num_classes  # Set the number of segmentation classes
        config.image_size = 256  # Ensure input image size is 1024x1024

        # Initialize the model (without pretrained weights)
        model = SegformerForSemanticSegmentation(config).cuda()
    elif args.model == "oursTrans":
        variant = 7
        fine_tune = False
        # decoder_channels = (512, 256, 128, 64,32)
        IgnoreBottleNeck = False
        segformer_variant = "nvidia/segformer-b2-finetuned-ade-512-512"
        print('T4444')
        model = DualEncoderTranUNet(wa_outs=[32, 64, 128, 256],
                                 wb_outs=[32, 64, 128, 256],
                                 m=256,
                                 segformer_variant=segformer_variant,
                                 simple_fusion=variant,
                                 regression=False,
                                 classes=args.num_classes,
                                 in_channels=3,
                                 # unet_encoder_weights="imagenet",
                                 # unet_encoder_name="resnet34",
                                 # decoder_type="unet++",
                                 IgnoreBottleNeck=IgnoreBottleNeck,
                                 # model_depth=model_depth,
                                 input_size=args.img_size,
                                 decoder_channels=(512, 256, 128, 64),
                                 ).cuda()
        if fine_tune:
            fineTune_PATH = '/home/ntorbati/PycharmProjects/DualNet_New/Pumaours_weit70/checkpoint_epoch1.pth'
            cp = torch.load(fineTune_PATH, weights_only=True)
            if "module." in list(cp.keys())[0]:
                cp = {k.replace("module.", ""): v for k, v in cp.items()}

            cp = adapt_checkpoint(cp, model)
            model.load_state_dict(cp, strict=False)  # strict=True to ensure all trained layers match



    elif args.model == "ResnetUnet":
        model = smp.Unet(classes=args.num_classes, in_channels=3).cuda()

    elif args.model == "TransUnet":
        config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]  # R50-ViT-B_16
        config_vit.n_classes = args.num_classes
        config_vit.n_skip = 3
        config_vit.patches.grid = (int(256 / 16), int(256 / 16))
        model = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

        # import torch
        # from ptflops import get_model_complexity_info
        # with torch.cuda.device(0):
        #     macs, params = get_model_complexity_info(net, (1, 256, 256), as_strings=True,
        #                                              print_per_layer_stat=True, verbose=True)
        #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        model.load_from(weights=np.load(config_vit.pretrained_path))
    else:
        model = get_transformer_based_model(parser=parser, model_name=args.model, img_size=args.img_size,
                                            num_classes=args.num_classes, in_ch=3).cuda()
    return model
    # return model


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
    os.makedirs(os.path.join("./train_result", 'total'), exist_ok=True)
    for folds in [0,1]:
        args.folds = folds
        args.train_file_dir = 'GCPS{}_train.txt'.format(folds)
        args.val_file_dir = 'GCPS{}_val.txt'.format(folds)
        # train_result = open(os.path.join("./train_result", 'total', '{}_train_resultFinal_v9.txt'.format(args.model)), 'w')
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
        segformer_variant = "nvidia/segformer-b2-finetuned-ade-512-512"
        processor = SegformerImageProcessor.from_pretrained(
            segformer_variant, do_resize=False, do_rescale=False)
        max_iterations = len(trainloader) * max_epoch
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
                        # batch_numpy = [img.permute(1, 2, 0).cpu().numpy() for img in
                        #                img_batch[:, 0:3]]  # (H, W, C) format
                        # # Process input images
                        # images1 = processor(images=batch_numpy,
                        #                     return_tensors="pt")  # Now it's ready for SegFormer
                        # images1 = {key: value.to(img_batch.device) for key, value in images1.items()}
                        # if img_batch.shape[1] > 3:
                        #     images1['pixel_values'] = torch.concatenate(
                        #         (images1['pixel_values'], img_batch[:, 3].unsqueeze(1)), dim=1)
                        # masks_pred = model(**images1)
                        # outputs = F.interpolate(masks_pred.logits, size=label_batch.size()[2:],
                        #                            mode='bilinear', align_corners=False)

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

                    # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                    # for param_group in optimizer.param_groups:
                    #     param_group['lr'] = lr_

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
                            # batch_numpy = [img.permute(1, 2, 0).cpu().numpy() for img in
                            #                img_batch[:, 0:3]]  # (H, W, C) format
                            # # Process input images
                            # images1 = processor(images=batch_numpy,
                            #                     return_tensors="pt")  # Now it's ready for SegFormer
                            # images1 = {key: value.to(img_batch.device) for key, value in images1.items()}
                            # if img_batch.shape[1] > 3:
                            #     images1['pixel_values'] = torch.concatenate(
                            #         (images1['pixel_values'], img_batch[:, 3].unsqueeze(1)), dim=1)
                            # masks_pred = model(**images1)
                            # outputs = F.interpolate(masks_pred.logits, size=label_batch.size()[2:],
                            #                            mode='bilinear', align_corners=False)

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
    for models in ["ours"]:# ["ours", "ResnetUnet"]:#,"TransUnet", "segformer"]:#,"ours", "ResnetUnet"]:
        print(models)
        args.model = models
        main(args)
