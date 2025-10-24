from transformers import SegformerImageProcessor
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from utils.utils_puma import Mine_resize, puma_f1_loss_custom
from utils.utils_puma import KorniaAugmentation
from torch import optim
from utils.LoadPumaData import PumaTissueDataset
from utils.utils_puma import collate_tile_patches
from torch.cuda.amp import autocast, GradScaler
import torch.nn.utils
from tqdm import tqdm
from utils.utils_puma import puma_dice_loss
from src.utils.kd_loss import MSELoss
from torch.utils.data import DataLoader
import copy

def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        save_checkpoint: bool = True,
        amp: bool = False,
        weight_decay: float = 1e-8,
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
        phase_mode = ['train', 'val'],
        dir_checkpoint = Path('E:/PumaDataset/checkpoints/'),
        er_di = False,
        model_name = '',
        val_sleep_time = 0,
        nuclei = False,
        segformer_variant = "nvidia/segformer-b2-finetuned-ade-512-512",
):
    # 1. Create dataset

    train_set = PumaTissueDataset(image_data1,
                                      mask_data1,
                                      n_class1=n_class,
                                      size1=target_siz,
                                    device1=device,
                                      transform = augmentation,
                                  target_size=ful_size,
                                  # train_indexes = train_indexes,
                                  mode='train',
                                  er_di = er_di,)
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

    n_train = len(train_set)

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=0)#os.cpu_count(), pin_memory=True)
    val_loader_args = dict(batch_size=val_batch, num_workers=0)#os.cpu_count(), pin_memory=True)
    aug_pipeline = KorniaAugmentation(
        mode="train", num_classes=n_class, seed=42, size=target_siz
    )


    if val_images is not None:
        dataloaders = {
            'train': DataLoader(train_set,shuffle=True,collate_fn=collate_tile_patches, **loader_args),
            'val': DataLoader(val_set, shuffle=False, drop_last=False,collate_fn=collate_tile_patches, **val_loader_args),
        }
    else :
        dataloaders = {
            'train': DataLoader(train_set, shuffle=True,collate_fn=collate_tile_patches, **loader_args),
        }
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Default optimizer setup
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=30,factor=weight_decay,min_lr=0.5*1e-7,cooldown=5)  # goal: maximize Dice score

    optimizer.zero_grad()
    criterion = nn.CrossEntropyLoss(weight=class_weights.float()) if n_class > 1 else nn.BCEWithLogitsLoss()

    # 5. Begin training
    counter = 0
    random.seed(42)

    if model_name == 'DGAUNet':
        optimizer_seg = optim.SGD(model.get_parameters(net="seg_net"), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
        optimizer_l = optim.SGD(model.get_parameters(net="l"), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
        criterion1 = MSELoss().cuda()



    processor = SegformerImageProcessor.from_pretrained(
        segformer_variant, do_resize=False, do_rescale=False)
    best_val_score = 0
    for epoch in range(1, epochs + 1):
        if counter > early_stopping:
            break
        scaler = GradScaler(enabled=amp)
        gradient_clipping = 1.0  # Gradient clipping value
        for phase in phase_mode:
            if phase == 'train':
                model.train()
                epoch_loss = 0
                f1_dice = 0
                epoch_dice = torch.zeros(n_class, device=device)
                with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
                    for images, true_masks in dataloaders[phase]:
                        if model_name == "SwinUnet":
                            target_siz = (224,224)
                        images, true_masks = Mine_resize(image=images, mask=true_masks, final_size=target_siz)
                        # Move to device
                        images = images.to(device=device, dtype=torch.float32)
                        true_masks = true_masks.to(device=device, dtype=torch.long)
                        aug_num = random.choice([0,1,2,3,5,6,7,8,9,10])
                        if aug_num > 2:
                            if augmentation:
                                images, true_masks = aug_pipeline(image=images, mask=true_masks)

                        torch.clamp(images, 0, 1)
                        true_masks = true_masks.long()
                        if model_name == 'DGAUNet':
                            r_out, r_e_out = model.get_R_out(images + true_masks.unsqueeze(1)/n_class)
                            loss_label = criterion(r_out, true_masks)
                            optimizer_l.zero_grad()
                            loss_label.backward()
                            optimizer_l.step()

                            l_out, l_e_out = model.get_L_out(images)
                            e_studyloss = criterion1(l_e_out, r_e_out.detach())
                            s_seg_loss = criterion(l_out, true_masks)
                            x_seg_loss = 0.9 * s_seg_loss + 0.1 * e_studyloss
                            optimizer_seg.zero_grad()
                            x_seg_loss.backward()
                            optimizer_seg.step()
                            loss = x_seg_loss
                            masks_pred = r_out
                        else:
                            optimizer.zero_grad()
                            # Mixed Precision Training
                            with autocast(enabled=amp):
                                if model_name == 'segformer':
                                    batch_numpy = [img.permute(1, 2, 0).cpu().numpy() for img in
                                                   images[:,0:3]]  # (H, W, C) format
                                    # Process input images
                                    images1 = processor(images=batch_numpy,
                                                       return_tensors="pt")  # Now it's ready for SegFormer
                                    images1 = {key: value.to(device) for key, value in images1.items()}
                                    if images.shape[1]>3:
                                        images1['pixel_values'] = torch.concatenate((images1['pixel_values'],images[:,3].unsqueeze(1)),dim=1)
                                    masks_pred = model(**images1)
                                    masks_pred = F.interpolate(masks_pred.logits, size=true_masks.size()[1:],
                                                                    mode='bilinear', align_corners=False)

                                else:
                                    masks_pred = model(images)
                            loss2 = 0.5 * puma_dice_loss(masks_pred, true_masks)
                            loss1 = 0.5 * criterion(masks_pred, true_masks)
                            loss = loss1 + loss2
                            scaler.scale(loss).backward()

                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

                            # Optimizer step
                            scaler.step(optimizer)
                            scaler.update()

                        # Update progress
                        epoch_loss += loss.item() * true_masks.size(0)  # Accumulate loss weighted by batch size
                        pbar.update(true_masks.shape[0])
                        masks_pred = F.softmax(masks_pred, dim=1)
                        # Calculate dice scores
                        with torch.no_grad():
                            if nuclei:
                                f1_dice += (puma_f1_loss_custom(masks_pred, true_masks, f1_ret=True))
                            else:
                                dice_scores = torch.zeros(n_class, device=device)
                                for c in range(n_class):
                                    preds_class = (masks_pred.argmax(dim=1) == c).float()
                                    true_class = (true_masks == c).float()
                                    intersection = (preds_class * true_class).sum()
                                    union = preds_class.sum() + true_class.sum()
                                    dice_scores[c] = (2 * intersection + 1e-7) / (union + 1e-7)
                                epoch_dice += dice_scores * true_masks.size(0)  # Accumulate weighted Dice
            if phase == 'train':
                epoch_loss /= n_train  # Divide by total training samples
                epoch_dice /= n_train
                print(epoch_dice.mean(), epoch_loss)
            elif phase == 'val' and epoch> val_sleep_time:  # Validation phase
                if epoch%1 == 0:
                    model.eval()
                    val_loss = 0
                    val_dice = torch.zeros(n_class, device=device)
                    val_iou = torch.zeros(n_class, device=device)
                    total_val_images = 0  # Track total validation images
                    TP = torch.zeros(n_class-1, device=device)
                    FP = torch.zeros(n_class-1, device=device)
                    FN = torch.zeros(n_class-1, device=device)
                    f1_dice_val = 0

                    with torch.no_grad():
                        if not nuclei:
                            for images, true_masks in dataloaders[phase]:
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

                                # Calculate dice scores
                                dice_scores = torch.zeros(n_class, device=device)
                                iou_scores = torch.zeros(n_class, device=device)
                                eps = 1e-6
                                for c in range(1,n_class):
                                    # Binary masks for the current class
                                    preds_class = (masks_pred.argmax(dim=1) == c).float()
                                    true_class = (true_masks == c).float()

                                    # True Positive (TP), False Positive (FP), False Negative (FN)
                                    tp = (preds_class * true_class).sum()
                                    fp = (preds_class * (1 - true_class)).sum()
                                    fn = ((1 - preds_class) * true_class).sum()
                                    TP[c-1]+=tp
                                    FP[c-1]+=fp
                                    FN[c-1]+=fn

                                    # Intersection and Union for Dice and IoU
                                    intersection = tp
                                    union = preds_class.sum() + true_class.sum()

                                    # Dice Score
                                    dice_scores[c] = (2 * intersection + 1e-7) / (union + 1e-7)

                                    # IoU Score
                                    union_iou = tp + fp + fn
                                    iou_scores[c] = (tp + 1e-7) / (union_iou + 1e-7)

                                val_dice += dice_scores * true_masks.size(0)
                                val_iou += iou_scores * true_masks.size(0)
                                total_val_images += true_masks.size(0)
                            total_dice = (2*TP + eps)/(2*TP+FP+FN + eps)
                            total_iou = (TP + eps)/(TP+FP+FN +eps)
                            print('new metrics: ', total_dice, total_iou)
                            val_loss /= total_val_images  # Divide by total validation samples
                            val_dice /= total_val_images
                            val_iou /= total_val_images
            if phase == 'train':
                epoch_loss /= n_train  # Divide by total training samples
                epoch_dice /= n_train
                # print(epoch_dice)
            else:
                try:
                    total_dice[total_dice.isnan()] = 0
                    print(f"{phase.capitalize()} Loss: {epoch_loss if phase == 'train' else val_loss:.4f}")
                    print(f'uDice:{total_dice.mean().item():.4f}', f'Mdice:{val_dice[1:].mean().item():.4f}',
                          f'dice: {val_dice}')
                    th = total_dice.mean()
                    if th > best_val_score:
                        best_val_score = th
                        print( 'Dice:' ,total_dice.mean(),'IoU:' ,total_iou.mean())
                        if save_checkpoint:
                            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                            state_dict = model.state_dict()
                            # state_dict['mask_values'] = dataset.mask_values
                            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(1)))
                            best_model_wts = copy.deepcopy(state_dict)

                except:
                    print('not validated')

    try:
        model.load_state_dict(best_model_wts)
    except:
        print('no model loaded')
    return model

