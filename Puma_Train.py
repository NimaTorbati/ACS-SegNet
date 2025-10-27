import numpy as np
np.bool=np.bool_
from utils.train_puma_dice import train_model
from sklearn.model_selection import KFold
from pathlib import Path
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import os
import argparse
import random
import numpy as np
import torch
from src.network.New.DGAUNet import  DGAUNet
from model import DualEncoderUNet
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import segmentation_models_pytorch as smp
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
                    choices=["TransUnet", "ACSSegNet", "segformer", "ResnetUnet", "DGAUNet"], help='model')
parser.add_argument('--batch_size', type=int, default=5, help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=512, help='img size of per batch')
parser.add_argument('--num_classes', type=int, default=5, help='seg num_classes')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--variant', type=str, default='1', help='fusion variant')
parser.add_argument('--iter', type=int, default=0, help='random seed')
args = parser.parse_args()
seed_torch(args.seed)


def get_model(args):
    if args.model == "DGAUNet":
        model = DGAUNet(output_ch=args.num_classes, img_size=512).cuda()
    elif args.model == "ACSSegNet":
        variant = int(args.variant)
        IgnoreBottleNeck = False
        segformer_variant = "nvidia/segformer-b2-finetuned-ade-512-512"
        model = DualEncoderUNet(
                                 segformer_variant=segformer_variant,
                                 simple_fusion=variant,
                                 regression=False,
                                 classes=args.num_classes,
                                 in_channels=3,
                                 model_depth=5,
                                 unet_encoder_weights="imagenet",
                                 unet_encoder_name="resnet34",
                                 IgnoreBottleNeck=IgnoreBottleNeck,
                                 decoder_channels=(256, 128, 64, 32, 16),
                                 )
    elif args.model == "segformer":
        segformer_variant = "nvidia/segformer-b2-finetuned-ade-512-512"
        config = SegformerConfig.from_pretrained(segformer_variant)

        # Modify the configuration to match your dataset
        config.num_channels = 3
        config.num_labels = args.num_classes  # Set the number of segmentation classes
        config.image_size = args.img_size  # Ensure input image size is 1024x1024

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

        model.load_from(weights=np.load(config_vit.pretrained_path))
    else:
        print('model error')
        return None

    return model





def main(args):
    model_name = args.model

    final_target_size = (args.img_size,args.img_size)
    n_class = args.num_classes
    device2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## load data
    image_data = np.load('/home/ntorbati/STORAGE/PumaDataset/1024_ims/ims.npy')
    mask_data = np.load('/home/ntorbati/STORAGE/PumaDataset/1024_ims/masks.npy')
    image_data_metas = image_data[0:102]
    mask_data_metas = mask_data[0:102]
    image_data_primary = image_data[103:]
    mask_data_primary = mask_data[103:]
    indices_metas = np.arange(image_data_metas.shape[0])
    indices_primary = np.arange(image_data_primary.shape[0])

    ## exxclude necrosis samples from data
    inds_m = []
    for k in range(mask_data_metas.shape[0]):
        im = mask_data_metas[k]
        if np.max(im) == 5:
            inds_m.append(k)

    inds_p = []
    for k in range(mask_data_primary.shape[0]):
        im = mask_data_primary[k]
        if np.max(im) == 5:
            inds_p.append(k)

    del image_data
    del mask_data



    n_folds = 3
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    splits_metas = list(kf.split(indices_metas))
    splits_primary = list(kf.split(indices_primary))





    for folds in range(0,n_folds):
        print(f'{args.model} training fold : {folds} ......................................................')
        ## exclude necrosis samples from data
        train_index_primary = indices_primary[splits_primary[folds][0]]
        for indd in inds_p:
            train_index_primary = np.delete(train_index_primary,np.where(train_index_primary == indd))
        val_index_primary = indices_primary[splits_primary[folds][1]]
        for indd in inds_p:
            val_index_primary = np.delete(val_index_primary,np.where(val_index_primary == indd))
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

        ## Micro Dice Initialization
        dir_checkpoint = Path('/home/ntorbati/PycharmProjects/ACS-SegNet/ModelWeights/Puma' + model_name + str(folds) + str(args.iter) + str(args.variant) +  '/')



        class_weights = [1, 1, 1, 1, 1]
        class_weights = torch.tensor(class_weights, device=device2,dtype=torch.float16)

        iters = [150]

        ## higher learning rate for DGAUNet to help it converge faster
        if model_name == 'DGAUNet':
            lr = 0.5*1e-2
        else:
            lr = 1e-4

        model1 = get_model(args)
        model1.to(device2)
        model1.n_classes = n_class

        target_size = final_target_size

        train_model(
        model = model1,
        device = device2,
        epochs = iters[0],
        batch_size = 4,
        learning_rate = lr,
        amp = False,
        weight_decay=0.7,  # learning rate decay rate
        target_siz=target_size,
        n_class=n_class,
        image_data1=train_images,
        mask_data1=train_masks,
        val_images = val_images,
        val_masks = val_masks,
        class_weights = class_weights,
        augmentation=True,# default None
        val_batch=1,
        early_stopping=10,
        ful_size=final_target_size,
        dir_checkpoint=dir_checkpoint,
        model_name=model_name,
        val_sleep_time = -1,
        nuclei=False,
        )
        del model1
if __name__ == "__main__":
    models = ["ACSSegNet", "ResnetUnet","TransUnet", "segformer", "DGAUNet"]
    for model in models:
        args.model = model
        main(args)

