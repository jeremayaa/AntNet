import numpy as np

def combine_semantic_masks(masks: np.ndarray) -> np.ndarray:
    """
    Combine the first five semantic masks with a logical OR.

    Parameters
    ----------
    masks : np.ndarray
        Array of shape (N, H, W, 6), where the last dim are one-hot masks.

    Returns
    -------
    np.ndarray
        Boolean array of shape (N, H, W, 1), where each pixel is True
        if it was positive in any of the first five masks.
    """
    # masks[..., :5] has shape (N, H, W, 5)
    # any(..., axis=-1, keepdims=True) collapses 5 → 1
    return np.any(masks[..., :5], axis=-1, keepdims=True).astype(np.uint8)

import math
import math
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

def show_n_images(image_grid, titles=None, cmaps=None, figsize=None):
    """
    Display a 2D grid of images, leaving None entries as blank cells.

    Parameters
    ----------
    image_grid : list of list of array-like or None
        image_grid[i][j] is either an image array to show, or None to leave blank.
    titles : list of list of str, optional
    cmaps : list of list of str or None, optional
    figsize : tuple (width, height), optional
    """
    # grid dimensions
    n_rows = len(image_grid)
    if n_rows == 0:
        raise ValueError("image_grid must have at least one row")
    n_cols = max(len(row) for row in image_grid)
    if n_cols == 0:
        raise ValueError("Each row must have at least one column")

    # default size
    if figsize is None:
        figsize = (4 * n_cols, 4 * n_rows)

    # pad titles/cmaps to the same shape
    if titles is None:
        titles = [[None]*len(row) for row in image_grid]
    if cmaps is None:
        cmaps = [[None]*len(row) for row in image_grid]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes)  # ensure 2D

    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j]
            # if this cell exists and is not None, plot it
            if j < len(image_grid[i]) and image_grid[i][j] is not None:
                img = image_grid[i][j]
                cmap = cmaps[i][j] if j < len(cmaps[i]) else None
                title = titles[i][j] if j < len(titles[i]) else None

                ax.imshow(img, cmap=cmap)
                ax.axis('off')
                if title:
                    ax.set_title(title)
            else:
                # either out of bounds or explicitly None: leave blank
                ax.axis('off')

    plt.tight_layout()
    plt.show()


from scipy.ndimage import binary_erosion

def extract_internal_edges(binary_mask: np.ndarray, erosion_iters: int = 1) -> np.ndarray:
    """
    Extracts internal edges of blobs in a binary mask using morphological erosion.

    Parameters:
        binary_mask (np.ndarray): Binary input mask (values should be 0 or 1).
        erosion_iters (int): Number of erosion iterations.

    Returns:
        np.ndarray: Binary mask containing only the internal edges.
    """
    if binary_mask.dtype != bool:
        binary_mask = binary_mask.astype(bool)
    
    # Perform erosion
    eroded_mask = binary_erosion(binary_mask, iterations=erosion_iters)
    
    # Subtract eroded mask from original to get internal edge
    internal_edges = binary_mask & ~eroded_mask

    return internal_edges.astype(np.uint8)





def make_heatmap_from_mask(binary_mask: np.ndarray) -> np.ndarray:
    """
    Creates a heatmap from a binary mask where the center of each blob has the highest value.
    Values increase toward the center of the cell, with boundaries = 1, next layer = 2, ..., center = N.

    Parameters:
        binary_mask (np.ndarray): Binary input mask (values 0 or 1).

    Returns:
        np.ndarray: Heatmap with increasing intensity toward the center of blobs.
    """
    if binary_mask.dtype != bool:
        binary_mask = binary_mask.astype(bool)

    heatmap = np.zeros_like(binary_mask, dtype=np.uint16)
    current_mask = binary_mask.copy()
    layer_value = 1

    while np.any(current_mask):
        eroded = binary_erosion(current_mask)
        edge_layer = current_mask & ~eroded
        heatmap[edge_layer] = layer_value
        current_mask = eroded
        layer_value += 1

    return heatmap


import matplotlib.pyplot as plt
import os
from PIL import Image

def save_images_and_masks(
    images: np.ndarray,
    masks: np.ndarray,
    images_dir: str,
    masks_dir: str
) -> None:
    """
    Save each image and its corresponding binary mask as JPGs in parallel folders,
    numbering files sequentially (1.jpg, 2.jpg, ...).

    Parameters
    ----------
    images : np.ndarray
        Array of shape (N, H, W, 3), dtype uint8 (or convertible to uint8).
    masks : np.ndarray
        Array of shape (N, H, W) or (N, H, W, 1) with binary values (0/1) or bool.
    images_dir : str
        Path to the folder where RGB images will be saved.
    masks_dir : str
        Path to the folder where masks will be saved.
    """
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    for i, (img, m) in enumerate(zip(images, masks), start=1):
        # Convert image to PIL and save
        img_pil = Image.fromarray(img.astype(np.uint8))

        # Prepare and convert mask to PIL
        m_arr = np.squeeze(m)       # (H, W)
        mask_pil = Image.fromarray((m_arr.astype(np.uint8) * 255))

        # Filename: "1.jpg", "2.jpg", ...
        fname = f"{i}.jpg"

        img_pil.save(os.path.join(images_dir, fname))
        mask_pil.save(os.path.join(masks_dir, fname))




import cv2
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt


from scipy import ndimage

def compute_vector_field(mask: np.ndarray,
                                  sigma: float = 2.0,
                                  eps: float = 1e-8) -> np.ndarray:
    """
    Compute the usual nearest‐cell unit vector field, then smooth it
    with a Gaussian kernel and renormalize.

    Parameters
    ----------
    mask : (H, W) array of {0,1}
        Binary mask defining the ‘1’-pixels (cells).
    sigma : float
        Standard deviation of the Gaussian smoothing (in pixels).
    eps : float
        Tiny constant to avoid divide‐by‐zero when renormalizing.

    Returns
    -------
    sm_vf : (H, W, 2) float
        At each pixel, a unit vector pointing toward the nearest cell—
        but smoothed across space to remove the grid‐artifact “stalactites.”
    """
    # 1) raw nearest‐cell field
    distances, indices = ndimage.distance_transform_edt(
        1 - mask,
        return_distances=True,
        return_indices=True
    )
    nearest_y, nearest_x = indices
    H, W = mask.shape
    Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    vy = nearest_y - Y
    vx = nearest_x - X
    lengths = np.sqrt(vx**2 + vy**2)
    lengths[lengths == 0] = 1.0
    vf_x = vx / lengths
    vf_y = vy / lengths

    # 2) Gaussian‐smooth each component
    sf_x = ndimage.gaussian_filter(vf_x, sigma=sigma, mode='nearest')
    sf_y = ndimage.gaussian_filter(vf_y, sigma=sigma, mode='nearest')

    # 3) renormalize to unit length
    mag = np.sqrt(sf_x**2 + sf_y**2)
    mag = np.where(mag < eps, 1.0, mag)
    sf_x /= mag
    sf_y /= mag

    # stack and return
    sm_vf = np.stack([sf_x, sf_y], axis=-1)
    return sm_vf


def compute_inverse_vector_field(mask: np.ndarray,
                                          sigma: float = 2.0,
                                          eps: float = 1e-8) -> np.ndarray:
    """
    Compute the inverse vector field (pointing from cell→background) and then
    smooth it with a Gaussian kernel, renormalizing to unit length.

    Parameters
    ----------
    mask : (H, W) array of {0,1}
        Binary mask where 1=cells, 0=background.
    sigma : float
        Gaussian smoothing sigma (in pixels).
    eps : float
        Small constant to avoid division‐by‐zero during renormalization.

    Returns
    -------
    sm_inv_vf : (H, W, 2) float
        At each cell‐pixel, a unit vector pointing toward the nearest background—
        but smoothed to remove jagged artifacts. Background pixels are (0,0).
    """
    # 1) raw inverse field
    distances, indices = ndimage.distance_transform_edt(
        mask,
        return_distances=True,
        return_indices=True
    )
    nearest_y, nearest_x = indices
    H, W = mask.shape
    Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    vy = nearest_y - Y
    vx = nearest_x - X
    lengths = np.sqrt(vx**2 + vy**2)
    lengths[lengths == 0] = 1.0
    inv_vf_x = vx / lengths
    inv_vf_y = vy / lengths

    # zero‐out background
    inv_vf_x[mask == 0] = 0
    inv_vf_y[mask == 0] = 0

    # 2) smooth each component
    sf_x = ndimage.gaussian_filter(inv_vf_x, sigma=sigma, mode='nearest')
    sf_y = ndimage.gaussian_filter(inv_vf_y, sigma=sigma, mode='nearest')

    # 3) renormalize to unit length
    mag = np.sqrt(sf_x**2 + sf_y**2)
    mag = np.where(mag < eps, 1.0, mag)
    sf_x /= mag
    sf_y /= mag

    # 4) zero‐out background again (smoothing can bleed)
    sf_x[mask == 0] = 0
    sf_y[mask == 0] = 0

    # stack and return
    sm_inv_vf = np.stack([sf_x, sf_y], axis=-1)
    return sm_inv_vf


import numpy as np
from scipy.ndimage import binary_erosion
from utils import compute_inverse_vector_field, extract_internal_edges

def compute_tangent_field(mask: np.ndarray,
                                   erosion_iters: int = 1,
                                   sigma: float = 2.0,
                                   eps: float = 1e-8) -> np.ndarray:
    """
    Compute a smooth tangent field along the internal edges of `mask` by
    Gaussian‐smoothing the outward normals before rotating them +90°.

    Parameters
    ----------
    mask : (H,W) bool or {0,1}
        Binary mask of cells.
    erosion_iters : int
        How many erosions for extracting the internal edge.
    sigma : float
        Standard deviation for the Gaussian filter (in pixels).
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    tangent_field : (H,W,2) float
        At each edge pixel a unit‐length tangent vector; elsewhere (0,0).
    """
    # 1) get outward normals everywhere
    normal_field = compute_inverse_vector_field(mask)   # shape (H,W,2)
    nx = normal_field[..., 0]
    ny = normal_field[..., 1]

    # 2) extract edge mask
    edges = extract_internal_edges(mask, erosion_iters=erosion_iters).astype(float)

    # 3) mask the normals, then smooth both normals & mask
    nx_masked = nx * edges
    ny_masked = ny * edges

    # gaussian smoothing
    smooth_nx_num = ndimage.gaussian_filter(nx_masked, sigma=sigma, mode='nearest')
    smooth_ny_num = ndimage.gaussian_filter(ny_masked, sigma=sigma, mode='nearest')
    smooth_den    = ndimage.gaussian_filter(edges, sigma=sigma, mode='nearest')

    # avoid divide‐by‐zero
    smooth_den = np.maximum(smooth_den, eps)

    # 4) normalized, smoothed normals
    snx = smooth_nx_num / smooth_den
    sny = smooth_ny_num / smooth_den
    lengths = np.sqrt(snx**2 + sny**2)
    lengths[lengths == 0] = 1.0
    snx /= lengths
    sny /= lengths

    # 5) rotate +90° to get tangents, zero‐out non‐edges
    H, W = mask.shape
    tangent_field = np.zeros((H, W, 2), dtype=float)
    # t = R90(n) = (-n_y, n_x)
    tangent_field[..., 0] = -sny * (edges > 0)
    tangent_field[..., 1] =  snx * (edges > 0)

    return tangent_field

def visualize_vector_field(vf: np.ndarray,
                           background: np.ndarray = None,
                           stride: int = 16,
                           scale: float = 10.0):
    """
    Plot a subsampled quiver of vf (H×W×2).  
    If `background` is provided (H×W or H×W×3), show it behind arrows.
    """
    import matplotlib.pyplot as plt

    H, W, _ = vf.shape
    ys = np.arange(0, H, stride)
    xs = np.arange(0, W, stride)
    Xg, Yg = np.meshgrid(xs, ys)
    U = vf[Yg, Xg, 0]
    V = vf[Yg, Xg, 1]

    plt.figure(figsize=(6,6))
    if background is not None:
        if background.ndim == 2:
            plt.imshow(background, cmap='gray', alpha=0.5)
        else:
            plt.imshow(background, alpha=0.5)
    plt.quiver(
        Xg, Yg, U, V,
        angles='xy', scale_units='xy', scale=1/scale, width=0.002
    )
    plt.axis('off')
    plt.tight_layout()
    plt.show()






# importing and managing jpeg images
def load_image(path: str) -> np.ndarray:
    """Load an RGB image or raise a clear error if not found."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image at '{path}'")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_mask(path: str) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not load mask at '{path}'")
    _, bin_mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
    return bin_mask


def extract_patch(image: np.ndarray,
                  center: tuple[int,int],
                  patch_size: tuple[int,int]) -> np.ndarray:
    """
    Crop a patch of size patch_size (h,w) centered at (x,y).
    Pads with zeros if near the border.
    """
    h, w = image.shape[:2]
    ph, pw = patch_size
    x, y = center

    x1 = max(0, x - pw//2)
    y1 = max(0, y - ph//2)
    x2 = min(w, x1 + pw)
    y2 = min(h, y1 + ph)

    patch = image[y1:y2, x1:x2]
    # pad if needed
    pad_h = ph - patch.shape[0]
    pad_w = pw - patch.shape[1]
    if pad_h > 0 or pad_w > 0:
        patch = np.pad(
            patch,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode='constant', constant_values=0
        )
    return patch
