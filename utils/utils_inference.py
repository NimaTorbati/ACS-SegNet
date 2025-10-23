### ignore segformer :)
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

class Dataset_test(Dataset):
    def __init__(self,
                 imgs1,
                 n_class1 = 6,
                 size1 = (1024,1024),
                 device1 = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 mode="valid",
                 target_size = (1024,1024),
                 paths= ''
                 ):
        self.imgs = imgs1
        self.paths = paths
        self.n_class = n_class1
        self.image = torch.zeros((3,size1[0],size1[1]), device=device1)
        # self.mask = torch.zeros((n_class1,size1[0],size1[1]), device=device1)
        self.size = size1
        self.target_size = target_size
        self.device = device1
        self.mode = mode
        return
    def __len__(self):
        lent = len(self.imgs)
        return lent

    def __getitem__(self, idx):
        image = self.imgs[idx].astype("float32")
        image = image/255
        image = np.transpose(image, (2, 0, 1))
        image = np.copy(image)
        pth = self.paths[idx]






        return image,pth

def validate_with_augmentations_and_ensembling(model, image_tensor, weights_list=None, device = '', regression = False):
    """
    Perform validation with test-time augmentations (TTA), ensembling, and debugging visualizations.

    Args:
        model: The PyTorch model.
        image_tensor: The input tensor stored on GPU.
        weights_list: List of paths to model weight files for ensembling.

    Returns:
        Final ensembled prediction as a NumPy array.
    """


    def augment(tensor):
        """
        Generate 7 unique augmentations of the input tensor.
        The augmentations include:
        1. Original
        2. Rotate 90°
        3. Rotate 180°
        4. Rotate 270°
        5. Horizontal flip
        6. Horizontal flip + Rotate 90°
        7. Horizontal flip + Rotate 270°

        Args:
            tensor: Input tensor of shape (B, C, H, W).

        Returns:
            List of tensors with augmentations applied.
        """
        return [
            tensor,  # Original
            torch.rot90(tensor, k=1, dims=[2, 3]),  # Rotate 90°
            torch.rot90(tensor, k=2, dims=[2, 3]),  # Rotate 180°
            torch.rot90(tensor, k=3, dims=[2, 3]),  # Rotate 270°
            torch.flip(tensor, dims=[3]),  # Horizontal flip
            torch.rot90(torch.flip(tensor, dims=[3]), k=1, dims=[2, 3]),  # Horizontal flip + Rotate 90°
            torch.rot90(torch.flip(tensor, dims=[3]), k=2, dims=[2, 3]),  # Horizontal flip + Rotate 180°
            torch.rot90(torch.flip(tensor, dims=[3]), k=3, dims=[2, 3]),  # Horizontal flip + Rotate 270°
        ]

    def reverse_augment(tensor, idx):
        """
        Reverse the augmentation applied at a given index.

        Args:
            tensor: Augmented tensor of shape (C, H, W).
            idx: Index of the augmentation (0-6).

        Returns:
            Tensor with the reverse transformation applied.
        """
        if idx == 0:  # Original
            return tensor
        elif idx == 1:  # Rotate 90° (reverse by rotating 270°)
            return torch.rot90(tensor, k=3, dims=[2, 3])
        elif idx == 2:  # Rotate 180° (reverse by rotating 180°)
            return torch.rot90(tensor, k=2, dims=[2, 3])
        elif idx == 3:  # Rotate 270° (reverse by rotating 90°)
            return torch.rot90(tensor, k=1, dims=[2, 3])
        elif idx == 4:  # Horizontal flip
            return torch.flip(tensor, dims=[3])
        elif idx == 5:  # Horizontal flip + Rotate 90° (reverse rotation first, then flip)
            return torch.flip(torch.rot90(tensor, k=3, dims=[2, 3]), dims=[3])
        elif idx == 6:  # Horizontal flip + Rotate 270° (reverse rotation first, then flip)
            return torch.flip(torch.rot90(tensor, k=2, dims=[2, 3]), dims=[3])
        elif idx == 7:  # Horizontal flip + Rotate 270° (reverse rotation first, then flip)
            return torch.flip(torch.rot90(tensor, k=1, dims=[2, 3]), dims=[3])
    # Initialize final probability predictions
    model.to(device)
    model.eval()
    final_probabilities = None  # Will hold the ensembled probabilities

    # Iterate through each model weight
    for k, weight_path in enumerate(weights_list):
        # Load model weights
        state_dict = torch.load(weight_path, map_location="cuda:0")
        if "module." in list(state_dict.keys())[0]:
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

        # Apply augmentations to the input tensor
        augmented_inputs = augment(image_tensor)

        # Apply the model to each augmented input and store predictions
        augmented_probabilities = []
        for i, augmented_input in enumerate(augmented_inputs):
            augmented_input = augmented_input.to(device)
            try:
                model.segformer
                # Process input images
                images = processor(images=[augmented_input[0, 0:3].permute(1, 2, 0).cpu().numpy()],
                                   return_tensors="pt")  # Now it's ready for SegFormer
                images = {key: value.to(model.device) for key, value in images.items()}
                if augmented_input.shape[1] > 3:
                    images['pixel_values'] = torch.concatenate((images['pixel_values'], augmented_input[:, 3].unsqueeze(1)),dim=1)
                pred = model(**images)
                pred = F.interpolate(pred.logits, size=augmented_input.size()[2:],
                                           mode='bilinear', align_corners=False)




            except:
                padded_input, pad = pad_to_multiple(augmented_input, multiple=32)
                # Forward pass with the padded image

                if regression:
                    pred_center, pred_binary = model(padded_input)
                    pred_center = remove_pad(pred_center, pad)
                    pred_binary = remove_pad(pred_binary, pad)
                    pred = torch.cat((pred_center, pred_binary), dim=1)
                else:
                    pred_padded = model(padded_input)  # Forward pass
                    pred = remove_pad(pred_padded, pad)

            if not regression:
                prob = F.softmax(pred, dim=1)  # Get probabilities
            else:
                prob = pred
            augmented_probabilities.append(prob)

            # Debugging: Visualize forward-augmented probability maps
            # plt.figure(figsize=(6, 4))
            # plt.title(f"Forward Augmentation {i}")
            # plt.imshow(prob[0].cpu().detach().numpy()[0], cmap="viridis")
            # plt.colorbar()
            # plt.show()

        # Reverse augmentations to align predictions

        aligned_probabilities = []
        for i, prob in enumerate(augmented_probabilities):
            reversed_prob = reverse_augment(prob, i)
            aligned_probabilities.append(reversed_prob)

            # Debugging: Visualize reverse-augmented probability maps
            # plt.figure(figsize=(6, 4))
            # plt.title(f"Reversed Augmentation {i}")
            # plt.imshow(reversed_prob[0].cpu().detach().numpy()[0], cmap="viridis")
            # plt.colorbar()
            # plt.show()
        # Ensure alignment of dimensions
        aligned_probabilities = torch.stack([p.squeeze(0) for p in aligned_probabilities], dim=0)

        # Average probabilities across augmentations for TTA
        tta_probability = torch.mean(aligned_probabilities, dim=0)

        # Add TTA probability to the ensemble
        if final_probabilities is None:
            final_probabilities = tta_probability
        else:
            final_probabilities += tta_probability

    # Average probabilities across all models in the ensemble
    final_probabilities /= len(weights_list)
    # final_probabilities[5][final_probabilities[5]>0.1] = 1
    # Final prediction (class-wise argmax)
    final_predictions = torch.argmax(final_probabilities, dim=0)

    # Move to CPU and convert to NumPy
    return final_probabilities


def add_pad(x,pad_siz = (0,0)):
    padded = F.pad(x, (pad_siz[0], pad_siz[0], pad_siz[1], pad_siz[1]), mode='reflect')
    return padded, (pad_siz[0], pad_siz[0], pad_siz[1], pad_siz[1])


def remove_pad(x, pad):
    """
    Removes padding from the tensor.

    Args:
        x (torch.Tensor): Tensor after processing (B, C, H_padded, W_padded).
        pad (tuple): Padding amounts (pad_left, pad_right, pad_top, pad_bottom).

    Returns:
        cropped (torch.Tensor): Tensor cropped back to original size.
    """
    pad_left, pad_right, pad_top, pad_bottom = pad
    # If no padding was applied, simply return x
    if pad_bottom > 0:
        x = x[:, :, : -pad_bottom, :]
    if pad_right > 0:
        x = x[:, :, :, : -pad_right]
    if pad_top > 0:
        x = x[:, :, pad_top:, :]
    if pad_left > 0:
        x = x[:, :, :, pad_left:]
    return x
def pad_to_multiple(x, multiple=32):
    """
    Pads the input tensor so that its height and width are divisible by 'multiple'.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W).
        multiple (int): The value by which height and width should be divisible.

    Returns:
        padded (torch.Tensor): Padded tensor.
        pad (tuple): Amount of padding applied (pad_left, pad_right, pad_top, pad_bottom).
    """
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple  # extra rows needed
    pad_w = (multiple - w % multiple) % multiple  # extra columns needed

    # For simplicity, we pad only to the bottom and right
    pad_top, pad_left = 0, 0
    pad_bottom, pad_right = pad_h, pad_w

    padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
    return padded, (pad_left, pad_right, pad_top, pad_bottom)

def aji_score(y_true, y_pred):
    """
    Compute the Aggregated Jaccard Index (AJI) between the groud truth and prediction masks.

    Args:
        y_true (numpy.ndarray): Ground truth mask.
        y_pred (numpy.ndarray): Predicted mask.

    Returns:
        float: Aggregated Jaccard Index.
    """
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    epsilon = 1e-6

    intersection = sum(y_true_f * y_pred_f)
    union = sum(y_true_f) + sum(y_pred_f) - intersection
    return (intersection + epsilon) / (union + epsilon)

def compute_aji_components(inst_true, inst_pred):
    true_ids = np.unique(inst_true)
    true_ids = true_ids[true_ids != 0]
    pred_ids = np.unique(inst_pred)
    pred_ids = pred_ids[pred_ids != 0]

    matched_pred = set()
    C = 0
    U = 0

    for gt_id in true_ids:
        gt_mask = (inst_true == gt_id)
        best_iou = 0.0
        best_pred = None

        for pr_id in pred_ids:
            if pr_id in matched_pred:
                continue
            pr_mask = (inst_pred == pr_id)
            inter = np.logical_and(gt_mask, pr_mask).sum()
            if inter == 0:
                continue
            union = np.logical_or(gt_mask, pr_mask).sum()
            iou = inter / union
            if iou > best_iou:
                best_iou = iou
                best_pred = pr_id

        if best_pred is not None:
            pr_mask = (inst_pred == best_pred)
            inter = np.logical_and(gt_mask, pr_mask).sum()
            union = np.logical_or(gt_mask, pr_mask).sum()
            C += inter
            U += union
            matched_pred.add(best_pred)
        else:
            U += gt_mask.sum()
    C1 = C
    U1 = U
    cntr = 0
    for pr_id in pred_ids:
        if pr_id not in matched_pred:
            U += (inst_pred == pr_id).sum()
            cntr += 1


    return C, U, C1, U1
