import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.utils import find_image, find_mask
from src.feature_area import get_lesion_area
from src.feature_shape import get_lesion_dimensions, get_perimeter, get_compactness
from src.feature_color import get_color_features

def extract_features_for_image(img_id):
    """
    Extracts all features for a single image.
    
    :param img_id: image filename (e.g. PAT_1516_1765_530.png)
    :return: dictionary of features, or None if image/mask not found
    """
    # Find image and mask
    img_path = find_image(img_id)
    mask_path = find_mask(img_id)
    
    if img_path is None or mask_path is None:
        return None
    
    # Load image and mask
    img = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    if img is None or mask is None:
        return None
    
    # Extract all features
    features = {}
    features["img_id"] = img_id
    
    # Shape features
    features["area"] = get_lesion_area(mask, img)
    height, width = get_lesion_dimensions(mask, img)
    features["height"] = height
    features["width"] = width
    features["perimeter"] = get_perimeter(mask, img)
    features["compactness"] = get_compactness(mask, img)
    
    # Color features
    color = get_color_features(img, mask)
    features.update(color)
    
    return features


def main():
    # Load metadata
    print("Loading metadata...")
    df = pd.read_csv("data/metadata.csv")
    print(f"Total images: {len(df)}")
    
    # Loop over all images and extract features
    print("Extracting features...")
    results = []
    failed = 0
    
    for img_id in tqdm(df["img_id"]):
        features = extract_features_for_image(img_id)
        if features is not None:
            results.append(features)
        else:
            failed += 1
    
    print(f"Successfully processed: {len(results)} images")
    print(f"Failed: {failed} images")
    
    # Convert to dataframe
    features_df = pd.DataFrame(results)
    
    # Add the diagnostic label from metadata
    features_df = features_df.merge(
        df[["img_id", "diagnostic", "patient_id"]],
        on="img_id",
        how="left"
    )
    
    # Save to CSV
    output_path = Path("data/features.csv")
    features_df.to_csv(output_path, index=False)
    print(f"Features saved to {output_path}")
    print(f"Feature columns: {list(features_df.columns)}")


if __name__ == "__main__":
    main()