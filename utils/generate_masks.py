import torch
import numpy as np
from skimage import morphology
import cv2
from scipy.interpolate import splprep, splev

from scipy.ndimage import convolve

def generate_gaussian_center_map_torch(instance_mask, sigma=3, device='cuda'):
    """
    Generate a center heatmap from an instance mask using GPU-accelerated Gaussian generation.

    Args:
        instance_mask (np.ndarray): 2D numpy array with instance labels.
        sigma (float): Gaussian sigma.
        device (str): 'cuda' or 'cpu'.

    Returns:
        torch.Tensor: (H, W) normalized heatmap tensor.
    """
    H, W = instance_mask.shape
    heatmap = torch.zeros((H, W), dtype=torch.float32, device=device)
    unique_ids = np.unique(instance_mask)

    for inst_id in unique_ids:
        if inst_id == 0:
            continue
        mask = (instance_mask == inst_id).astype(np.uint8)
        if mask.sum() == 0:
            continue

        # Centerline extraction remains CPU-based
        centers_np = extract_centerline(mask, sigma=2.0, min_size=20, smoothness=1)
        if centers_np is None or len(centers_np) == 0:
            continue

        centers = torch.tensor(centers_np, dtype=torch.float32, device=device)
        gaussian = generate_gaussian_heatmap_torch((H, W), centers, sigma=sigma, device=device)
        heatmap = torch.maximum(heatmap, gaussian)

    heatmap /= heatmap.max() + 1e-8  # normalize to [0, 1]
    return heatmap
def generate_binary_mask(instance_mask):
    instance_mask = dilate_instances_no_overlap(instance_mask,dilate=False)
    return (instance_mask > 0).astype(np.float32)

def generate_gaussian_heatmap_torch(shape, centers, sigma=3, device='cuda'):
    """
    Generate a normalized Gaussian heatmap from multiple centers using PyTorch on GPU.

    Args:
        shape (tuple): (height, width) of the output heatmap.
        centers (Tensor or np.ndarray): Tensor of shape (N, 2) with (y, x) center coordinates.
        sigma (float): Standard deviation of the Gaussian.
        device (str): 'cuda' or 'cpu'.

    Returns:
        heatmap (torch.Tensor): Normalized heatmap tensor on the specified device, shape (H, W).
    """
    H, W = shape
    if not isinstance(centers, torch.Tensor):
        centers = torch.tensor(centers, dtype=torch.float32, device=device)

    yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    yy = yy.unsqueeze(0)  # (1, H, W)
    xx = xx.unsqueeze(0)  # (1, H, W)

    cy = centers[:, 0].view(-1, 1, 1)  # (N, 1, 1)
    cx = centers[:, 1].view(-1, 1, 1)  # (N, 1, 1)

    gaussian_maps = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))  # (N, H, W)
    heatmap = gaussian_maps.sum(dim=0)  # sum over N → (H, W)
    heatmap /= heatmap.max() + 1e-8  # normalize

    return heatmap

def extract_centerline(mask, sigma=2.0, min_size=20, smoothness=1, num_points=200):
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
    # distance_map = ndi.distance_transform_edt(mask)

    # Apply Laplacian of Gaussian (LoG) to find ridges
    # log_ridges = -filters.laplace(filters.gaussian(distance_map, sigma=sigma))
    # ridge_thresh = log_ridges > filters.threshold_otsu(log_ridges)

    # Refine ridge with original mask and skeletonize
    # ridge_clean = (1-ridge_thresh)  & mask
    ridge_clean = morphology.skeletonize(mask)    # ridge_clean = morphology.skeletonize(ridge_clean)

    ridge_clean = remove_short_branches(ridge_clean)

    # ridge_clean = remove_small_objects(ridge_clean, min_size=1)

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
    # im = np.zeros_like(mask, dtype=np.uint8)
    if len(x) < 11:
        num_points = 100
    else:
        num_points = 100
        # indices = np.linspace(0, len(x) - 1, 10, dtype=int)
        # x = x[indices]
        # y = y[indices]
    try:
        tck, _ = splprep([x, y], s=smoothness)
        u_fine = np.linspace(0, 1, num_points)
        x_smooth, y_smooth = splev(u_fine, tck)
        smoothed_coords = np.stack([y_smooth, x_smooth], axis=1)
        x_new = np.array(x_smooth, dtype=np.int64)
        y_new = np.array(y_smooth, dtype=np.int64)
        # im[y_new, x_new] = 1
    except:
        # print("Spline fitting failed:", e)
        # im[y, x] = 1

        return coords



    return smoothed_coords

def remove_short_branches(skeleton, min_endpoints=2):
    """Remove shortest branches from skeleton if endpoints > min_endpoints."""
    endpoints_mask = find_endpoints(skeleton)
    junctions_mask = find_junctions(skeleton)

    endpoints = np.column_stack(np.nonzero(endpoints_mask))

    if len(endpoints) <= min_endpoints:
        return skeleton  # nothing to prune

    branches = []
    for ep in endpoints:
        path = trace_branch(skeleton, tuple(ep), junctions_mask)
        branches.append((len(path), path))

    # Sort branches by length
    branches.sort(key=lambda x: x[0])

    # Remove shortest branches (until only 2 remain)
    skeleton = skeleton.copy()
    to_remove = len(endpoints) - min_endpoints
    for i in range(to_remove):
        _, path = branches[i]
        for y, x in path:
            skeleton[y, x] = 0

    return skeleton
def find_endpoints(skel):
    """Return binary mask of endpoints (1 neighbor)."""
    kernel = np.array([[1,1,1],
                       [1,10,1],
                       [1,1,1]])
    conv = convolve(skel.astype(np.uint8), kernel, mode='constant')
    return (conv == 11)  # 10 (self) + 1 neighbor

def find_junctions(skel):
    """Return binary mask of junctions (3 or more neighbors)."""
    kernel = np.array([[1,1,1],
                       [1,10,1],
                       [1,1,1]])
    conv = convolve(skel.astype(np.uint8), kernel, mode='constant')
    return (conv >= 13)  # 10 (self) + 3+ neighbors

def trace_branch(skel, start, junction_mask):
    """Trace pixels from start to nearest junction, return list of coords."""
    visited = set()
    queue = [start]
    path = []

    while queue:
        current = queue.pop(0)
        path.append(current)
        visited.add(current)

        y, x = current
        neighbors = [(y+dy, x+dx) for dy in [-1,0,1] for dx in [-1,0,1]
                     if not (dy == 0 and dx == 0)]

        for ny, nx in neighbors:
            if (0 <= ny < skel.shape[0] and 0 <= nx < skel.shape[1]
                and skel[ny, nx] and (ny, nx) not in visited):
                if junction_mask[ny, nx]:
                    return path  # stop at junction
                queue.append((ny, nx))
    return path  # fallback
