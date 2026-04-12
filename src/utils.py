from pathlib import Path

# Paths to images and masks
IMG_DIRS = [
    Path("data/imgs/imgs_part_1"),
    Path("data/imgs/imgs_part_2"),
    Path("data/imgs/imgs_part_3"),
]
MASK_DIR = Path("data/masks/masks")

def find_image(img_id):
    """Search for an image across all image subfolders."""
    for img_dir in IMG_DIRS:
        path = img_dir / img_id
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