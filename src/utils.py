from pathlib import Path

# Paths to images and masks
IMG_DIR = Path("data/imgs")
MASK_DIR = Path("data/masks")

def find_image(img_id):
    """Search for an image in the imgs folder."""
    path = IMG_DIR / img_id
    if path.exists():
        return path
    return None

def find_mask(img_id):
    """Find the mask for a given image."""
    mask_name = img_id.replace(".png", "_mask.png")
    path = MASK_DIR / mask_name
    if path.exists():
        return path
    return None