### Here is the training script. We used several loss terms (l1, l2, combination of l1 and l2, AJI loss),
# an ad-hoc spatial attention to improve the segmentation accuracy.
from scipy import ndimage
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from utils.utils import (dilate_instances_no_overlap,
                   KorniaAugmentation)
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
import matplotlib.pyplot as plt
from skimage import filters, morphology, feature
from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects
from scipy.interpolate import splprep, splev
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.ndimage import convolve
import torch
import numpy as np
from tqdm import tqdm
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import kornia.filters as KF
import torch
import torch.nn.functional as F
import cv2
from skimage.morphology import skeletonize
from scipy.ndimage import label as ndi_label




class SmoothToSharpLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='mean')
        self.l2 = nn.MSELoss(reduction='mean')

    def forward(self, pred, target, epoch, total_epochs):
        # Linearly decrease L2 weight, increase L1 weight
        l2_weight = max(0.0, 1.0 - (epoch / total_epochs))
        l1_weight = 1.0 - l2_weight

        loss = l2_weight * self.l2(pred, target) + l1_weight * self.l1(pred, target)
        return loss

### this version applies l1 loss in cell regions and l2 in background
class SmoothToSharpLoss1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        mask = torch.tensor(target < 0.9, dtype=pred.dtype, device=pred.device)
        # Compute per-element L1 and L2 losses
        l1_loss = torch.abs(pred - target)
        l2_loss = (pred - target) ** 2

        # Blend losses using the mask
        combined_loss = mask * l1_loss + (1 - mask) * l2_loss

        # Return the mean loss
        return combined_loss.mean()


def find_endpoints_torch(skeleton: torch.Tensor) -> torch.BoolTensor:
    """
    skeleton: [H, W] or [1, H, W] binary
    Returns: same shape, True where endpoints
    """
    if skeleton.dim() == 2:
        skeleton = skeleton.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    else:
        skeleton = skeleton.unsqueeze(1)  # [B,1,H,W]

    kernel = torch.tensor([[1,1,1],[1,10,1],[1,1,1]], dtype=torch.float32, device=skeleton.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1,1,3,3]

    neighbor = F.conv2d(skeleton.float(), kernel, padding=1)
    endpoints = (neighbor == 11) & (skeleton.bool())
    return endpoints.squeeze(1)


from scipy.spatial.distance import pdist


def analyze_skeleton_labeled_torch(skeleton_labeled: torch.Tensor):
    """
    Input: [H, W] int32 labels
    Returns: analysis dict and binary candidate mask
    """
    device = skeleton_labeled.device
    sk_np = skeleton_labeled.cpu().numpy()

    labels = np.unique(sk_np)
    labels = labels[labels != 0]
    candidates = torch.zeros_like(skeleton_labeled, dtype=torch.uint8)

    analysis = {}
    for lid in labels:
        mask = (sk_np == lid)
        mask_t = torch.from_numpy(mask).to(device)
        endpoints = find_endpoints_torch(mask_t)
        coords = torch.nonzero(endpoints, as_tuple=False).cpu().numpy()

        n_end = coords.shape[0]
        if n_end >= 2:
            dist = pdist(coords)
            min_d = dist.min()
        else:
            min_d = 0.0
        length = mask.sum()

        if length > 5 and (n_end >= 3 or min_d / length < 0.8):
            candidates |= mask_t

        analysis[int(lid)] = {
            'n_endpoints': int(n_end),
            'min_distance': float(min_d),
            'length': int(length)
        }

    return analysis, candidates

def post_process_remove_fps_torch(pred_int1: torch.Tensor, pred_cent: torch.Tensor, inst_th=0.8):
    """
    Batched version: inputs shape [B, H, W], returns tensor [B, H, W]
    """
    B, H, W = pred_int1.shape
    device = pred_int1.device
    outputs = []
    kernel = torch.tensor(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                          dtype=torch.float32, device=device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    for b in range(B):

        pi = (pred_int1[b] > 0.5).cpu().numpy().astype(np.uint8)
        pc = (pred_cent[b] > inst_th).cpu().numpy().astype(np.uint8)

        inv_skel = skeletonize(1 - pc).astype(np.uint8)
        lbl, _ = ndi_label(inv_skel)

        lbl_t = torch.from_numpy(lbl).long().to(device)

        _, candidates = analyze_skeleton_labeled_torch(lbl_t)

        # Dilate using PyTorch conv2d
        cand = F.conv2d(candidates.unsqueeze(0).unsqueeze(0).float(), kernel, padding=3)
        cand = (cand > 0).squeeze().to(torch.uint8)
        outputs.append(cand)

    return torch.stack(outputs, dim=0)


def gaussian_blur_mask(mask, kernel_size=51, sigma=10.0):
    """
    mask: [B, 1, H, W] - binary mask tensor
    Returns: [B, 1, H, W] - blurred (soft) mask
    """
    blurred = KF.gaussian_blur2d(mask.float(), (kernel_size, kernel_size), (sigma, sigma))
    return blurred / blurred.max()  # normalize to [0,1]

def weighted_mse_loss(pred, target, importance_mask, alpha=5.0):
    """
    pred, target: [B, 1, H, W] - predicted and ground truth maps
    importance_mask: [B, 1, H, W] - binary mask of important regions
    alpha: scalar weight multiplier for important regions
    """
    weight = 1 + (alpha - 1) * importance_mask  # 1 in background, alpha in important areas
    loss = weight * (pred - target) ** 2
    return loss.mean()


def post_process_remove_fps(pred_int1 = None, pred_cent = None, inst_th = 0.5, find_instances = True, find_masks = True, train = False):
    th = inst_th
    pred_int = np.array(pred_int1 > 0.5, dtype=np.uint8)
    cent_certain = np.array(pred_cent > th, dtype=np.uint8)
    radius = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    # Apply erosion

    cent_certain_skelt1 = morphology.skeletonize(1-cent_certain)
    cent_certain_skelt2 = cv2.connectedComponents(np.array(cent_certain_skelt1, dtype=np.uint8))
    if train:
        cent_certain_skelt3 = analyze_skeleton_labeled(cent_certain_skelt2[1], pred_int=pred_int, find_masks=find_masks)
        cent_certain_s = cv2.dilate(np.array(cent_certain_skelt3[1],dtype=np.uint8), kernel, iterations=2)
        return cent_certain_s
    # if np.sum(cent_certain_skelt3[1]) > 0:
    #     plt.imshow(cent_certain_skelt3[1])
    #     plt.show()
    else:
        cent_certain_skelt = morphology.skeletonize(cent_certain)
        cent_certain_skelt = cv2.dilate(np.array(cent_certain_skelt, dtype=np.uint8), kernel, iterations=1)

        cent_certain_skelt = cv2.connectedComponents(np.array(cent_certain_skelt, dtype=np.uint8))
        # cent_certain_skelt = analyze_skeleton_labeled(cent_certain_skelt[1])
        cent_certain_skelt = np.array(cent_certain_skelt[1], dtype=np.uint8)
        seperated_init = (pred_int  + cent_certain_skelt1) * (1 - cent_certain_skelt)

        insts = cv2.connectedComponents(np.array(seperated_init,dtype=np.uint8))
        instances = insts[1]
        for ids in range(1,insts[0]):
            if ids == 0:
                continue
            mask = (insts[1] == ids)
            if np.max(seperated_init*mask) == 1:
                # plt.imshow(mask)
                # plt.show()
                # print(ids)
                instances = instances * (1-mask)
                # plt.imshow(instances)
                # plt.show()
        return instances

def find_endpoints(skeleton):
    """
    Find endpoints in a binary skeleton (pixels with only one neighbor).
    Returns a binary mask of endpoints.
    """
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]])

    neighbor_count = convolve(skeleton.astype(np.uint8), kernel, mode='constant')
    # A pixel is an endpoint if it has value 11 (10 for itself + 1 neighbor)
    endpoints = (skeleton > 0) & (neighbor_count == 11)
    return endpoints

def find_candidate_masks(candidates = None, pred_int = None):
    new = candidates + pred_int
    new = new.astype(np.uint8)
    comps = cv2.connectedComponents(new)
    out = np.zeros_like(new, dtype=np.uint8)
    for inds in range(1,comps[0]):
        mask = (comps[1] == inds)
        if np.max(new[mask]) == 2:
            out += mask
    return out




def analyze_skeleton_labeled(skeleton_labeled = None, pred_int = None, find_masks = True):
    """
    Analyze each labeled skeleton:
    - Count endpoints
    - Find shortest distance between endpoints

    Args:
        skeleton_labeled: 2D array where each skeleton is labeled with a unique integer

    Returns:
        dict: Mapping from label -> {'n_endpoints': int, 'min_distance': float}
    """
    analysis = {}
    labels = np.unique(skeleton_labeled)
    labels = labels[labels != 0]  # remove background
    candidates = np.zeros_like(skeleton_labeled)
    for label_id in labels:
        skeleton = (skeleton_labeled == label_id)
        endpoints_mask = find_endpoints(skeleton)
        endpoints = np.column_stack(np.nonzero(endpoints_mask))

        n_endpoints = len(endpoints)
        if n_endpoints >= 2:
            distances = pdist(endpoints)  # pairwise distances
            min_distance = distances.min()
        else:
            min_distance = 0.0
        length = np.sum(skeleton)
        if length > 5:
            if n_endpoints >= 3 or (min_distance/length) < 0.8:
                candidates += skeleton

        analysis[label_id] = {
            'n_endpoints': n_endpoints,
            'min_distance': min_distance,
            'length': length
        }
    if find_masks:
        candidates_masks = find_candidate_masks(candidates=candidates, pred_int=pred_int)
        return analysis, candidates, candidates_masks
    else:
        return analysis, candidates, None


def extract_centerline(mask, sigma=2.0, min_size=20, smoothness=2, num_points=200):
    """
    Extract a smooth spline centerline from a binary mask.

    Args:
        mask (np.ndarray): Binary mask of a single object.
        sigma (float): Gaussian smoothing for ridge detection.
        min_size (int): Minimum object size to keep in final output.
        smoothness (float): Spline smoothness parameter.
        num_points (int): Number of interpolated points along the centerline.

    Returns:
        np.ndarray: Interpolated centerline points as (num_points, 2) array [(y1,x1), (y2,x2), ...]
    """
    # Ensure binary mask
    mask = (mask > 0).astype(np.uint8)
    radius = 2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))

    # Apply erosion
    eroded = cv2.erode(mask, kernel, iterations=1)
    if np.sum(eroded) > 0:
        mask = eroded
    # Compute distance transform
    distance_map = ndi.distance_transform_edt(mask)

    # Apply Laplacian of Gaussian (LoG) to find ridges
    log_ridges = -filters.laplace(filters.gaussian(distance_map, sigma=sigma))
    ridge_thresh = log_ridges > filters.threshold_otsu(log_ridges)

    # Refine ridge with original mask and skeletonize
    ridge_clean = (1-ridge_thresh)  & mask
    ridge_clean = morphology.skeletonize(ridge_clean)
    ridge_clean = remove_small_objects(ridge_clean, min_size=1)

    # Extract coordinates
    coords = np.column_stack(np.nonzero(ridge_clean))  # (y, x)

    if len(coords) < 2:
        print("Warning: Not enough ridge points for spline.")
        return coords

    # Sort by y (or use other logic for more complex shapes)
    coords = coords[np.argsort(coords[:, 0])]

    # Fit spline
    x = coords[:, 1]
    y = coords[:, 0]
    im = np.zeros_like(mask, dtype=np.uint8)

    try:
        tck, _ = splprep([x, y], s=smoothness)
        u_fine = np.linspace(0, 1, num_points)
        x_smooth, y_smooth = splev(u_fine, tck)
        smoothed_coords = np.stack([y_smooth, x_smooth], axis=1)
        x_new = np.array(x_smooth, dtype=np.int64)
        y_new = np.array(y_smooth, dtype=np.int64)
        im[y_new, x_new] = 1
    except Exception as e:
        print("Spline fitting failed:", e)
        im[y, x] = 1

        return coords



    return smoothed_coords




def generate_gaussian_center_map(instance_mask, sigma=5):
    heatmap = np.zeros(instance_mask.shape, dtype=np.float32)
    unique_ids = np.unique(instance_mask)
    for inst_id in unique_ids:
        if inst_id == 0:
            continue
        mask = (instance_mask == inst_id).astype(np.uint8)
        if mask.sum() == 0:
            continue
        center = np.round(ndimage.center_of_mass(mask)).astype(int)
        if center[0] >= heatmap.shape[0] or center[1] >= heatmap.shape[1]:
            continue
        gaussian = generate_gaussian_heatmap(instance_mask.shape, center, sigma=sigma)
        heatmap = np.maximum(heatmap, gaussian)
    return heatmap

def generate_gaussian_center_map_new(instance_mask, sigma=3):
    heatmap = np.zeros(instance_mask.shape, dtype=np.float32)
    unique_ids = np.unique(instance_mask)
    for inst_id in unique_ids:
        if inst_id == 0:
            continue
        mask = (instance_mask == inst_id).astype(np.uint8)
        if mask.sum() == 0:
            continue
        centers = extract_centerline(mask, sigma=2.0, min_size=20, smoothness=1, num_points=20)
        # if center[0] >= heatmap.shape[0] or center[1] >= heatmap.shape[1]:
        #     continue
        gaussian = generate_gaussian_heatmap(instance_mask.shape, centers, sigma=sigma)
        heatmap = np.maximum(heatmap, gaussian)
    return heatmap


def generate_binary_mask(instance_mask):
    instance_mask = dilate_instances_no_overlap(instance_mask,dilate=False)
    return (instance_mask > 0).astype(np.float32)








def generate_gaussian_heatmap_torch(shape, center, sigma=5):
    """Generate Gaussian heatmap using PyTorch."""
    height, width = shape
    y = torch.arange(0, height, device=device).view(-1, 1).float()
    x = torch.arange(0, width, device=device).view(1, -1).float()
    cy, cx = center

    heatmap = torch.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
    return heatmap


def compute_centers_and_masks_torch(instance_mask):
    """
    Compute binary and center (heatmap) masks using PyTorch.
    instance_mask: numpy array [H, W]
    Returns: (center_heatmap, binary_mask) as numpy arrays
    """
    instance_mask_tensor = torch.from_numpy(np.array(instance_mask,dtype=np.int64)).to(device)

    binary_mask = (instance_mask_tensor > 0).float()

    heatmap = torch.zeros_like(instance_mask_tensor, dtype=torch.float32)

    unique_instances = torch.unique(instance_mask_tensor)
    for inst_id in unique_instances:
        if inst_id.item() == 0:
            continue

        mask = (instance_mask_tensor == inst_id)
        ys, xs = mask.nonzero(as_tuple=True)

        if len(xs) == 0 or len(ys) == 0:
            continue

        center_y = ys.float().mean()
        center_x = xs.float().mean()
        heatmap += generate_gaussian_heatmap_torch(instance_mask.shape, (center_y, center_x), sigma=5)

    return heatmap.cpu().numpy(), binary_mask.cpu().numpy()


def preprocess_and_save_masks(instance_masks, dist_path, bin_path):
    print("Starting GPU-accelerated preprocessing of center and binary masks...")

    center_maps = []
    binary_masks = []

    for instance_mask in tqdm(instance_masks, desc="Processing masks"):
        center, binary = compute_centers_and_masks_torch(instance_mask)
        center_maps.append(center)
        binary_masks.append(binary)

    center_maps = np.stack(center_maps)
    binary_masks = np.stack(binary_masks)

    np.save(dist_path, center_maps)
    np.save(bin_path, binary_masks)

    print(f"Finished preprocessing. Total masks processed and saved: {len(center_maps)}")




class NucleiDataset(Dataset):
    def __init__(self, images, center_maps, binary_masks, transform=None):
        self.images = images
        self.center_maps = center_maps
        self.binary_masks = binary_masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32) / 255.0
        center_map = self.center_maps[idx]
        binary_mask = self.binary_masks[idx]

        if self.transform:
            augmented = self.transform(image=image, masks=[center_map, binary_mask])
            image = augmented['image']
            center_map, binary_mask = augmented['masks']

        return image, torch.tensor(center_map, dtype=torch.float32).unsqueeze(0), \
            torch.tensor(binary_mask, dtype=torch.float32).unsqueeze(0)


def train_model(model, train_loader, val_loader, device,
                epochs=20, lr=1e-5,aug = False,
                fold = None,
                debug = False):
    early_stop = 0
    counter = 0
    random.seed(42)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True, cooldown=3)
    scaler = GradScaler()
    center_loss_fn = nn.MSELoss()#SmoothToSharpLoss1()#nn.L1Loss()#nn.MSELoss()
    # semantic_loss_fn = smp.losses.DiceLoss(mode='binary')
    aug_pipeline = KorniaAugmentation(
        mode="train", seed=42, regression=True,
    )
    val_loss_max = 1e3
    weighted_mse = False
    all_losses = []
    att_enable = True
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, center_maps, binary_masks in train_loader:
            images = torch.tensor(images, dtype=torch.float32).to(device).permute(0, 3, 1, 2)
            center_maps = torch.tensor(center_maps, dtype=torch.float32).to(device)
            # binary_masks = torch.tensor(binary_masks, dtype=torch.uint8).to(device)


            aug_num = random.choice([0, 1, 2, 3, 5, 6, 7, 8, 9, 10])
            # print('random_staff = ', aug_num, class_num)
            counter+= 1
            if aug_num > 2:
                if aug:
                    if counter  == 10:
                        print(counter)
                    true_masks = center_maps# torch.cat([center_maps, binary_masks], dim=1)
                    images, true_masks = aug_pipeline(image=images, mask=true_masks)
                    center_maps = true_masks[:,0,:,:].unsqueeze(1)
                    # binary_masks = true_masks[:,1,:,:].unsqueeze(1)
                    torch.clamp(images, 0, 1)
                    # binary_masks = binary_masks.long()
            optimizer.zero_grad()
            with autocast():
                # model.center_decoder.gradient_masks = None
                pred_center = model(images)

                # # attention
                # if att_enable:
                #     # print('attention enabled')
                #     if epoch > 10:
                #         aug_num = random.choice([0, 1, 2, 3, 5, 6, 7, 8, 9, 10])
                #         if aug_num > 5:
                #             # print('attention')
                #             diff_case_list = []
                #             for btch in range(pred_center.shape[0]):
                #                 diff_cases = post_process_remove_fps(pred_int1=pred_binary[btch, 0].detach().cpu().numpy(),
                #                                                  pred_cent=pred_center[btch, 0].detach().cpu().numpy(),
                #                                                  find_masks=False, find_instances=False, inst_th=0.8,
                #                                                      train=True)
                #                 diff_case_list.append(torch.tensor(diff_cases).unsqueeze(0).to(device))
                #             stacked = torch.stack(diff_case_list, dim=0)
                #
                #             # stacked = post_process_remove_fps_torch(pred_binary[:,0], pred_center[:,0], inst_th=0.8)
                #             stacked = stacked.unsqueeze(1) if stacked.ndim == 3 else stacked
                #
                #             smooth_mask = gaussian_blur_mask(stacked, kernel_size=71, sigma=15.0)
                #             smooth_mask = smooth_mask.to(images.device).float()
                #             model.center_decoder.gradient_masks = (1 + 0.5*smooth_mask)
                #             pred_center, pred_binary = model(images)
                #             weighted_mse = False # it was true during my last attention experiment



                if weighted_mse:
                    loss_center = weighted_mse_loss(pred_center, center_maps,smooth_mask, alpha=50)
                    weighted_mse = False
                else:
                    loss_center = center_loss_fn(pred_center, center_maps)
#                                                , epoch=epoch, total_epochs=max(50,epochs))
                    weighted_mse = False
                # # aji loss

                loss = loss_center# + aji_loss
                t_loss_center = loss_center.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_loss_cent = 0.0

        itr = 0
        with torch.no_grad():
            for images, center_maps, binary_masks in val_loader:
                images = images.to(device).permute(0, 3, 1, 2)
                center_maps = center_maps.to(device)
                # binary_masks = binary_masks.to(device)
                pred_center = model(images)
                loss_center = center_loss_fn(pred_center, center_maps)
                                            # , epoch=70, total_epochs=100)
                # loss_semantic = semantic_loss_fn(pred_binary, binary_masks)

                # #adj_loss
                # pred_inst = predict_fast(pred_center, pred_binary)
                # pred_inst = pred_inst.to(device)
                # aji_loss = 1 - aji_score_torch(binary_masks, pred_inst>0)


                val_loss_cent = loss_center
                val_loss += (loss_center).item() * images.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)  # Adjust LR based on val_loss  # Adjust the LR
        if val_loss < val_loss_max:
            best_model_wts = model.state_dict()

            # new_pred = post_process_remove_fps(pred_int1=pred_binary[1, 0].cpu().numpy(), pred_cent = pred_center[1, 0].cpu().numpy())
            if debug:
                plt.imshow(pred_center[0, 0].cpu())
                plt.show()
                plt.imshow(center_maps[0, 0].cpu())
                plt.show()
                plt.imshow(images[0,0:3].permute(1,2,0).cpu())
                plt.show()
                val_loss_max = val_loss
                early_stop = 0
            print(val_loss_cent)
        else:
            early_stop += 1
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        all_losses.append([epoch, train_loss, val_loss, val_loss_cent.item(), t_loss_center])
        if early_stop > 30:
            print("Early stopping triggered.")
            if att_enable:
                break
            else:
                early_stop = 0
                att_enable = True

    np.save('/home/ntorbati/STORAGE/PumaDataset/1024_ims/' + str(fold) + '_batch_losses_monuseg_3ch_weight.npy', all_losses)
    try:
        model.load_state_dict(best_model_wts)
    except:
        print('no model loaded')
    return model




