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


import matplotlib.pyplot as plt

def plot_image_and_mask(images: np.ndarray, masks: np.ndarray, idx: int) -> None:
    """
    Plot the idx-th RGB image and its corresponding binary mask side by side.
    If masks has 6 channels, it will first OR together the first 5 channels.

    Parameters
    ----------
    images : np.ndarray
        Array of shape (N, H, W, 3) with RGB images.
    masks : np.ndarray
        Array of shape (N, H, W, 6), (N, H, W, 1), or (N, H, W) with masks.
    idx : int
        Index of the image/mask pair to plot.

    Returns
    -------
    None
    """
    img = images[idx]
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Expected images[...,3], got shape {img.shape}")

    m = masks[idx]
    # collapse 6-channel → (H,W)
    if m.ndim == 3 and m.shape[-1] == 6:
        m = np.any(m[..., :5], axis=-1)
    # drop singleton channel
    elif m.ndim == 3 and m.shape[-1] == 1:
        m = m[..., 0]
    # now m should be 2D
    if m.ndim != 2:
        raise ValueError(f"Could not interpret mask shape {m.shape}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(img.astype('uint8'))
    ax1.set_title(f"Image {idx}")
    ax1.axis('off')

    ax2.imshow(m, cmap='gray', vmin=0, vmax=1)
    ax2.set_title(f"Mask {idx}")
    ax2.axis('off')

    plt.tight_layout()
    plt.show()


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