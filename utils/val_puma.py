import logging
from transformers import SegformerImageProcessor
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from utils.utils_puma import Mine_resize, puma_f1_loss_custom, compute_puma_dice_micro_dice_eval
from utils.utils_puma import KorniaAugmentation
from torch import optim
from torch.utils.data import DataLoader, random_split
from utils.LoadPumaData import load_data_tissue,PumaTissueDataset
from utils.utils_puma import compute_puma_dice_micro_dice
import os
from utils.utils_puma import FocalLoss, puma_f1,ocelot_f1,tile_or_pad_image_and_mask,collate_tile_patches,PQ_Mean,prepare_pq
from torch.cuda.amp import autocast, GradScaler
import torch.nn.utils
from tqdm import tqdm
from utils.utils_puma import puma_dice_loss, circular_augmentation,random_sub_image_sampling,puma_f1_loss
import numpy as np
from utils.class_augs import tissue_aug_gpu
from src.utils import losses
from src.utils.kd_loss import MSELoss, DistillKL


dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('E:/PumaDataset/checkpoints/')


def upsample_difficult_cases(dice_epoch = None, inddss = None):
    diff_inds = inddss[np.where(dice_epoch > np.median(dice_epoch))[0]]

    return diff_inds




def val_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale = 0,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        target_siz = (128,128),
        n_class = 6,
        image_data1 = None,
        mask_data1 = None,
        val_images = None,
        val_masks = None,
        class_weights = torch.ones(6),
        augmentation = True,
        val_batch = 1,
    early_stopping = 8,
        ful_size = (1024,1024),
        grad_wait = 1,
        logg = False,
        logg_selected = False,
        val_augmentation = None,

        train_indexes=None,
        input_folder='',
        output_folder='',
        ground_truth_folder='',
        tis_path = '',
        phase_mode = ['train', 'val'],
        test_images=None,
        test_masks=None,
        test_folder='',
        test_output_folder='',
        test_ground_truth_folder='',
        folds = None,
        dir_checkpoint = Path('E:/PumaDataset/checkpoints/'),
        er_di = False,
        progressive = False,
        necros_im = None,
        model_name = '',
        val_sleep_time = 0,
        fine_tune_cofs = None,
        best_val = 0,
        stick_tissue = True,
        nuclei = False,
        dataset_name = None,
        segformer_variant = "nvidia/segformer-b2-finetuned-ade-512-512"
):
    # 1. Create dataset


    if val_images is not None:
        val_set = PumaTissueDataset(val_images,
                                          val_masks,
                                          n_class1=n_class,
                                          size1=target_siz,
                                        device1=device,
                                          transform = None,
                                    target_size=ful_size,
                                    mode='valid',
                                    er_di = er_di)
        n_val = len(val_set)



    # 3. Create data loaders
    val_loader_args = dict(batch_size=val_batch, num_workers=0)#os.cpu_count(), pin_memory=True)


    if val_images is not None:
        dataloaders = {
            'val': DataLoader(val_set, shuffle=False, drop_last=False,collate_fn=collate_tile_patches, **val_loader_args),
        }



    counter = 0
    random.seed(42)


    all_epochs = epochs
    processor = SegformerImageProcessor.from_pretrained(
        segformer_variant, do_resize=False, do_rescale=False)
    dice_cof = 0.5
    m_dice_cof = 0.5
    best_val_score = 0
    # if progressive:
    #     for param in model.encoder.parameters():
    #         param.requires_grad = False
    #     for param in model.decoder.parameters():
    #         param.requires_grad = False
    for pro_phase in ["all"]:#+ (["decoder", "encoder"] if progressive else []):
        for epoch in range(0,1):
            for phase in phase_mode:
                if 1 :  # Validation phase
                    if 1:
                        model.eval()
                        val_loss = 0
                        val_dice = torch.zeros(n_class, device=device)
                        total_val_images = 0  # Track total validation images
                        frame_type = []
                        f1_dice_val = 0
                        with torch.no_grad():
                            if 1:
                                for images, true_masks in dataloaders['val']:
                                    images = images.to(device=device, dtype=torch.float32)
                                    true_masks = true_masks.to(device=device, dtype=torch.long)
                                    images, true_masks = Mine_resize(image=images, mask=true_masks, final_size=target_siz)
                                    with autocast(enabled=amp):
                                        if model_name == 'segformer':

                                            batch_numpy = [img.permute(1, 2, 0).cpu().numpy() for img in
                                                           images[:, 0:3]]  # (H, W, C) format
                                            # Process input images
                                            images1 = processor(images=batch_numpy,
                                                                return_tensors="pt")  # Now it's ready for SegFormer
                                            images1 = {key: value.to(device) for key, value in images1.items()}
                                            if images.shape[1] > 3:
                                                images1['pixel_values'] = torch.concatenate(
                                                    (images1['pixel_values'], images[:, 3].unsqueeze(1)), dim=1)
                                            masks_pred = model(**images1)
                                            masks_pred = F.interpolate(masks_pred.logits, size=true_masks.size()[1:],
                                                                       mode='bilinear', align_corners=False)
                                        else:
                                            if model_name == 'DGAUNet':
                                                masks_pred = model.val(images)
                                            else:
                                                masks_pred = model(images)
                                        masks_pred = F.softmax(masks_pred, dim=1)
                                        loss = puma_dice_loss(masks_pred, true_masks)

                                    val_loss += loss.item() * true_masks.size(0)  # Accumulate loss weighted by batch size

                                    num_classes = masks_pred.shape[1]
                                    dice_scores = torch.zeros(n_class, device=device)
                                    for c in range(n_class):
                                        preds_class = (masks_pred.argmax(dim=1) == c).float()
                                        true_class = (true_masks == c).float()
                                        intersection = (preds_class * true_class).sum()
                                        union = preds_class.sum() + true_class.sum()
                                        dice_scores[c] = (2 * intersection + 1e-7) / (union + 1e-7)
                                    val_dice += dice_scores * true_masks.size(0)
                                    total_val_images += true_masks.size(0)

                            if 1:
                                mean_puma_dice, metrics = compute_puma_dice_micro_dice_eval(model=model,
                                                                                                            target_siz=target_siz,
                                                                                                            epoch=epoch,
                                                                                                            input_folder=input_folder,
                                                                                                            output_folder=output_folder,
                                                                                                            ground_truth_folder=ground_truth_folder,
                                                                                                            device=device,
                                                                                                            er_di = er_di,
                                                                                                            save_jpg=True,
                                                                                                            nuclei = nuclei,
                                                                                                            in_channles=images.shape[1],)

                                # mean_micro_dice = mean_micro_dice[0:num_classes-1]
                                # micro_dices = np.mean(np.array(mean_micro_dice[0:num_classes-1]))
                                # print('mean micro dice = ', micro_dices, 'mean dice = ', mean_puma_dice)
                                # total_dice = dice_cof * mean_puma_dice + m_dice_cof * micro_dices
                                # print('mean micro dice = ', micro_dices, 'mean dice = ', mean_puma_dice, 'total dice = ', total_dice)
                                np.save('/home/ntorbati/PycharmProjects/DualNet_New/Puma_Checkpoint/' + model_name + 'metrics.npy', [metrics, mean_puma_dice])

            return [metrics,mean_puma_dice]
