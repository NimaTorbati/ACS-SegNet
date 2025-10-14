import numpy as np
from tensorboard.compat.tensorflow_stub.dtypes import variant

np.bool=np.bool_
import segmentation_models_pytorch as smp
import torch
from utils.train_puma_dice import train_model
from sklearn.model_selection import KFold
from pathlib import Path
from utils.utils import fill_background_holes_batch,copy_data,adapt_checkpoint, adapt_checkpoint_dualEncoder
from torch import nn
import cv2
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import os
import argparse
import random
import numpy as np
import torch
from src.network.New.DGAUNet import  DGAUNet
from model_TransSegUnet import DualEncoderUNet as DualEncoderTranUNet
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
from model import DualEncoderUNet
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, SegformerConfig
import segmentation_models_pytorch as smp
from utils.utils_puma import Mine_resize
from model_triple_encoder import TripleEncoderUNet
# from cit.CiT_Net_T import CIT as cit
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
                             "UNext", "UNetplus", "UNet3plus", "SwinUnet", "MedT", "TransUnet", "ours","ours_s","ours_n", "segformer", "ResnetUnet", "DGAUNet", "ours_weit", "oursTrans", "oursTriple", "cit"], help='model')
# parser.add_argument('--base_dir', type=str, default="./data", help='dir')
# parser.add_argument('--train_file_dir', type=str, default="GCPS_train.txt", help='dir')
# parser.add_argument('--val_file_dir', type=str, default="GCPS_val.txt", help='dir')
# parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--batch_size', type=int, default=5, help='batch_size per gpu')
# parser.add_argument('--epoch', type=int, default=150, help='train epoch')
parser.add_argument('--img_size', type=int, default=512, help='img size of per batch')
parser.add_argument('--num_classes', type=int, default=5, help='seg num_classes')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--variant', type=str, default='9', help='random seed')
parser.add_argument('--iter', type=int, default=111, help='random seed')
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
    elif args.model == "DGAUNet":
        model = DGAUNet(output_ch=args.num_classes, img_size=512).cuda()
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
                                cof_unet= cof_unet,
                                cof_seg = cof_seg,
                                 )

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


        if fine_tune:
            fineTune_PATH = '/home/ntorbati/PycharmProjects/DualNet_New/Pumaours_weit70/checkpoint_epoch1.pth'
            cp = torch.load(fineTune_PATH, weights_only=True)
            if "module." in list(cp.keys())[0]:
                cp = {k.replace("module.", ""): v for k, v in cp.items()}

            cp = adapt_checkpoint(cp, model)
            model.load_state_dict(cp, strict=False)  # strict=True to ensure all trained layers match

    elif args.model == "oursTriple":
        variant = int(args.variant)
        fine_tune = False
        # decoder_channels = (512, 256, 128, 64,32)
        IgnoreBottleNeck = False
        segformer_variant = "nvidia/segformer-b2-finetuned-ade-512-512"
        print('Triple')
        model = TripleEncoderUNet(wa_outs=[32, 64, 128, 256],
                                 wb_outs=[32, 64, 128, 256],
                                 m=256,
                                 segformer_variant=segformer_variant,
                                 simple_fusion=variant,
                                 regression=False,
                                 classes=args.num_classes,
                                 in_channels=3,
                                 unet_encoder_weights="imagenet",
                                 unet_encoder_name="resnet50",
                                 # decoder_type="unet++",
                                 IgnoreBottleNeck=IgnoreBottleNeck,
                                 # model_depth=model_depth,
                                input_size=args.img_size,
                                 decoder_channels=(400, 250, 128, 64, 32),
                                 ).cuda()
        if fine_tune:
            fineTune_PATH = '/home/ntorbati/PycharmProjects/DualNet_New/Pumaours_weit70/checkpoint_epoch1.pth'
            cp = torch.load(fineTune_PATH, weights_only=True)
            if "module." in list(cp.keys())[0]:
                cp = {k.replace("module.", ""): v for k, v in cp.items()}

            cp = adapt_checkpoint(cp, model)
            model.load_state_dict(cp, strict=False)  # strict=True to ensure all trained layers match
    elif args.model == "oursTrans":
        variant = int(args.variant)
        fine_tune = False
        # decoder_channels = (512, 256, 128, 64,32)
        IgnoreBottleNeck = False
        segformer_variant = "nvidia/segformer-b2-finetuned-ade-512-512"
        print('T4444')
        model = DualEncoderTranUNet(
                                 m=128,
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
                                 decoder_channels=(256, 128, 64, 32),
                                 ).cuda()
        if fine_tune:
            fineTune_PATH = '/home/ntorbati/PycharmProjects/DualNet_New/Pumaours_weit70/checkpoint_epoch1.pth'
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

    elif args.model == "ResnetUnet":
        model = smp.Unet(classes=args.num_classes, in_channels=3).cuda()

    elif args.model == "TransUnet":
        config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]  # R50-ViT-B_16
        config_vit.n_classes = args.num_classes
        config_vit.n_skip = 3
        config_vit.patches.grid = (int(512 / 16), int(512 / 16))
        model = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

        # import torch
        # from ptflops import get_model_complexity_info
        # with torch.cuda.device(0):
        #     macs, params = get_model_complexity_info(net, (1, 256, 256), as_strings=True,
        #                                              print_per_layer_stat=True, verbose=True)
        #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        model.load_from(weights=np.load(config_vit.pretrained_path))
    elif args.model == "cit":
        model = cit(img_size=args.img_size, out_chans=args.num_classes,in_chans=3).cuda()

    else:
        model = get_transformer_based_model(parser=parser, model_name=args.model, img_size=args.img_size,
                                            num_classes=args.num_classes, in_ch=3).cuda()


    return model

    # return model




def main(args):
    model_name = args.model# + str(args.variant)#segformerUnet'#'unet'#'segformer'
    print(model_name)
    # variant = 1
    # model_depth = 4
    # IgnoreBottleNeck = False
    segformer_variant = "nvidia/segformer-b1-finetuned-ade-512-512"
    parallel = False
    tissue_labels = ['tissue_white_background','tissue_stroma','tissue_blood_vessel','tissue_tumor','tissue_epidermis','tissue_necrosis']
    tissue_images_path ='/home/ntorbati/STORAGE/PumaDataset/01_training_dataset_tif_ROIs/'
    tissue_labels_path = '/home/ntorbati/STORAGE/PumaDataset/01_training_dataset_geojson_tissue/'
    if args.model == "cit":
        args.img_size = 224
    final_target_size = (args.img_size,args.img_size)
    n_class = args.num_classes
    use_necros = False



    device2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = 3

    val_percent = 0.2

    if use_necros:
        image_data_necros = np.load('/home/ntorbati/STORAGE/PumaDataset/1024_ims/necros_ims.npy')
        image_data_necros = image_data_necros[0:100]
        mask_data_necros = np.load('/home/ntorbati/STORAGE/PumaDataset/1024_ims/necros_masks.npy')
        mask_data_necros = mask_data_necros[0:100]
        mask_data_necros = fill_background_holes_batch(mask_data_necros)

        res = 880
        image_data_necros = image_data_necros[:, 0:res, 0:res, :]
        mask_data_necros = mask_data_necros[:, 0:res, 0:res]

        image_data_necros = np.array([cv2.resize(img, (1024,1024), interpolation=cv2.INTER_LINEAR) for img in image_data_necros])

        mask_data_necros = np.array([cv2.resize(mask, (1024,1024), interpolation=cv2.INTER_NEAREST) for mask in mask_data_necros])
        #


        # mask_data_necros[mask_data_necros == 5] = 3

    image_data = np.load('/home/ntorbati/STORAGE/PumaDataset/1024_ims/ims.npy')
    mask_data = np.load('/home/ntorbati/STORAGE/PumaDataset/1024_ims/masks.npy')




    image_data_metas = image_data[0:102]
    mask_data_metas = mask_data[0:102]





    image_data_primary = image_data[103:]
    mask_data_primary = mask_data[103:]


    indices_metas = np.arange(image_data_metas.shape[0])
    indices_primary = np.arange(image_data_primary.shape[0])




    inds_m = []
    for k in range(mask_data_metas.shape[0]):
        im = mask_data_metas[k]
        if np.max(im) == 5:
            inds_m.append(k)
            # mask_data_metas[k] = 0*mask_data_metas[k]
            # image_data_metas[k] = np.ones(image_data_metas[k].shape)*255
    # mask_data_metas = mask_data_metas[inds]
    # image_data_metas = image_data_metas[inds]
    # indices_metas = indices_metas[inds]

    inds_p = []
    for k in range(mask_data_primary.shape[0]):
        im = mask_data_primary[k]
        if np.max(im) == 5:
            inds_p.append(k)
            # mask_data_primary[k] = 0*mask_data_primary[k]
            # image_data_primary[k] = np.ones(image_data_primary[k].shape)*255
    # image_data_primary = image_data_primary[inds]
    # mask_data_primary = mask_data_primary[inds]
    # indices_primary = indices_primary[inds]

    del image_data
    del mask_data



    n_folds = 3
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    splits_metas = list(kf.split(indices_metas))
    splits_primary = list(kf.split(indices_primary))





    for folds in range(0,n_folds):
        print('training fold ', str(folds))
        train_index_primary = indices_primary[splits_primary[folds][0]]
        for indd in inds_p:
            train_index_primary = np.delete(train_index_primary,np.where(train_index_primary == indd))

        val_index_primary = indices_primary[splits_primary[folds][1]]
        for indd in inds_p:
            val_index_primary = np.delete(val_index_primary,np.where(val_index_primary == indd))

        print(val_index_primary)
        train_index_metas = indices_metas[splits_metas[folds][0]]
        for indd in inds_m:
            train_index_metas = np.delete(train_index_metas,np.where(train_index_metas == indd))

        val_index_metas = indices_metas[splits_metas[folds][1]]
        for indd in inds_m:
            val_index_metas = np.delete(val_index_metas,np.where(val_index_metas == indd))


        train_data_primary = image_data_primary[train_index_primary]

        val_data_primary = image_data_primary[val_index_primary]

        train_data_metas = image_data_metas[train_index_metas]

        val_data_metas = image_data_metas[val_index_metas]


        val_images = np.concatenate((val_data_metas,val_data_primary),axis=0)
        val_masks = np.concatenate((mask_data_metas[val_index_metas], mask_data_primary[val_index_primary]), axis=0)

        train_images = np.concatenate((train_data_metas,train_data_primary),axis=0)##
        train_masks = np.concatenate((mask_data_metas[train_index_metas], mask_data_primary[train_index_primary]), axis=0)##



        if use_necros:
            train_images = np.concatenate((train_images,image_data_necros),axis=0)
            train_masks = np.concatenate((train_masks,mask_data_necros),axis=0)

        # train_masks[train_masks == 5] = 3
        # train_indexes = np.append(train_indexes, diff_cases)

        dir_checkpoint = Path('/home/ntorbati/PycharmProjects/DGAUNet/Puma' + model_name + str(folds) + str(args.iter) + str(args.variant) +  '/')
        val_save_path = '/home/ntorbati/PycharmProjects/DGAUNet/validation_ground0/' + model_name + str(folds)+ str(args.iter) + str(args.variant)
        val_save_path1 = '/home/ntorbati/PycharmProjects/DGAUNet/validation_images0/' + model_name + str(folds)+ str(args.iter)+ str(args.variant)
        output_folder = '/home/ntorbati/PycharmProjects/DGAUNet/validation_prediction0/' + model_name + str(folds)+ str(args.iter)+ str(args.variant)


        copy_data(validation_indices = val_index_primary, data_path = tissue_labels_path,data_path1= tissue_images_path, save_path = val_save_path,save_path1 = val_save_path1, data_type = 'primary',tissue=False,masks=mask_data_primary)
        copy_data(validation_indices = val_index_metas, data_path = tissue_labels_path,data_path1=tissue_images_path, save_path = val_save_path,save_path1 = val_save_path1, data_type = 'metastatic',tissue=False,masks=mask_data_metas)

        # train_images = np.concatenate((train_images, val_images), axis=0)
        # train_masks = np.concatenate((train_masks, val_masks), axis=0)


        # train_images =  np.concatenate((image_data_primary[val_index_primary], image_data_primary[train_index_primary]), axis=0)
        # train_masks =  np.concatenate((mask_data_primary[val_index_primary], mask_data_primary[train_index_primary]), axis=0)
        # train_images, train_masks = blood_vessel_aug(train_masks = train_masks, train_images = train_images, num_classes = n_class)
        train_indexes = np.linspace(0, len(train_images)-1, len(train_images))


        class_weights_unet = [1, 0.8, 1.5, 0.8, 1.9]
        class_weights_segformer = [1, 1.2, 1.3, 1.2, 0.7]
        class_weights_unet = torch.tensor(class_weights_unet, device=device2,dtype=torch.float16)
        class_weights_segformer = torch.tensor(class_weights_segformer, device=device2,dtype=torch.float16)

        class_weights = [1, 1, 1, 1, 1]

        class_weights = torch.tensor(class_weights, device=device2,dtype=torch.float16)
        iters = [50]
        if model_name == 'DGAUNet':
            lr = 0.5*1e-2
        else:
            lr = 1e-4

        # if model_name  == 'unet':
        #     model1 = smp.Unet(classes=n_class,in_channels=in_channels)
        #
        # elif model_name == 'segformer':
        #     config = SegformerConfig.from_pretrained(segformer_variant)
        #
        #     # Modify the configuration to match your dataset
        #     config.num_channels = in_channels
        #     config.num_labels = n_class  # Set the number of segmentation classes
        #     config.image_size = final_target_size[0]  # Ensure input image size is 1024x1024
        #
        #     # Initialize the model (without pretrained weights)
        #     model1 = SegformerForSemanticSegmentation(config)
        #
        #
        #
        # elif model_name == 'segformerUnet':
        #     model1 = USS(wa_outs=[32, 64, 128, 256],
        #                             wb_outs=[32, 64, 128, 256],
        #                             m=256,
        #                             segformer_variant=segformer_variant,
        #                             simple_fusion = variant,
        #                             regression = False,
        #                             classes=n_class,
        #                             in_channels=in_channels,
        #                              unet_encoder_weights="imagenet",
        #                              unet_encoder_name="resnet34",
        #                              # decoder_type="unet++",
        #                              IgnoreBottleNeck = IgnoreBottleNeck,
        #                              # model_depth=model_depth,
        #                              decoder_channels=(254, 126, 62, 30,14),
        #                  device=device2,
        #                              )
        #
        #

        #

        model1 = get_model(args)

        cp = adapt_checkpoint_dualEncoder(model1, folds)
        model1.load_state_dict(cp, strict=False)  # strict=True to ensure all trained layers match

        # fineTune_PATH = '/home/ntorbati/PycharmProjects/DGAUNet/Puma' + model_name + str(folds) + str(5) + str(args.variant) + '/checkpoint_epoch1.pth'
        # cp = torch.load(fineTune_PATH, weights_only=True)
        # if "module." in list(cp.keys())[0]:
        #     cp = {k.replace("module.", ""): v for k, v in cp.items()}
        #
        # cp = adapt_checkpoint(cp, model1)
        # model1.load_state_dict(cp, strict=False)  # strict=True to ensure all trained layers match

        if parallel:
            model1 = nn.DataParallel(model1)


        model1.to(device2)
        model1.n_classes = n_class

        target_size = final_target_size
        size = target_size[0]

        train_model(
        model = model1,
        device = device2,
        epochs = iters[0],
        batch_size = 4,
        learning_rate = lr,
        val_percent = 0.2,
        save_checkpoint = True,
        img_scale = 0.5,
        amp = False,
        weight_decay=0.7,  # learning rate decay rate
        momentum = 0.999,
        gradient_clipping = 1.0,
        target_siz=target_size,
        n_class=n_class,
        image_data1=train_images,
        mask_data1=train_masks,
        val_images = val_images,
        val_masks = val_masks,
        class_weights = class_weights,
        class_weights_unet = class_weights_unet,
        class_weights_segformer = class_weights_segformer,
        augmentation=True,# default None
        val_batch=1,
        early_stopping=10,
        ful_size=final_target_size,
            val_augmentation=True,
            train_indexes = train_indexes,
            input_folder=val_save_path1,
            output_folder=output_folder,
            ground_truth_folder=val_save_path,
            folds = n_folds,
            dir_checkpoint=dir_checkpoint,
            logg=False,
            model_name=model_name,
            val_sleep_time = 10,
            stick_tissue = False,
            nuclei=False,
            segformer_variant = segformer_variant
            # grad_wait=int(20 / 10)
        )
        del model1
if __name__ == "__main__":
    # if args.iter == 210:
    #     models = ["TransUnet","segformer", "DGAUNet"]
    # else:
    #     models = ["ResnetUnet","ours","ours_s","ours_n"]
    # models = [ "ours"]    # segunet iter 3
    # models = [ "v1",""] #iter2
    models = ["ours"]
    for model in models:

        # if model == "v0":
        #     args.variant = '0'
        #     model = "ours"
        # if model == "ours_weit":
        #     args.variant = '0'
        #     model = "ours_weit"
        # if model == "v1":
        #     args.variant = '1'
        #     model = "ours"
        # if model == "v9":
        #     args.variant = '9'
        #     model = "ours"
        # if model == "new":
        #     from model_new import DualEncoderUNet
        #     args.variant = '9'
        #     model = "ours_new"
        args.model = model

        main(args)

