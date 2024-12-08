import os
import shutil
import random

def split_paired_dataset(satellite_dir, mask_dir, output_base_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios must sum to 1.0"

    # Create output directories
    os.makedirs(os.path.join(output_base_dir, 'train', 'satellite'), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, 'train', 'mask'), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, 'val', 'satellite'), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, 'val', 'mask'), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, 'test', 'satellite'), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, 'test', 'mask'), exist_ok=True)

    # Get all image files
    satellite_images = sorted([f for f in os.listdir(satellite_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    # Shuffle images
    random.seed(42)  # For reproducibility
    random.shuffle(satellite_images)

    # Calculate split indices
    total_images = len(satellite_images)
    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)

    # Split indices
    train_images = satellite_images[:train_end]
    val_images = satellite_images[train_end:val_end]
    test_images = satellite_images[val_end:]

    # Function to copy paired images
    def copy_paired_images(image_list, split_name):
        for img_name in image_list:
            # Construct corresponding mask name
            mask_name = img_name.replace('satellite', 'mask')
            
            # Copy satellite image
            src_sat_path = os.path.join(satellite_dir, img_name)
            dst_sat_path = os.path.join(output_base_dir, split_name, 'satellite', img_name)
            shutil.copy(src_sat_path, dst_sat_path)
            
            # Copy mask image
            src_mask_path = os.path.join(mask_dir, mask_name)
            dst_mask_path = os.path.join(output_base_dir, split_name, 'mask', mask_name)
            shutil.copy(src_mask_path, dst_mask_path)

    # Copy images to respective splits
    copy_paired_images(train_images, 'train')
    copy_paired_images(val_images, 'val')
    copy_paired_images(test_images, 'test')

    # Print dataset split information
    print(f"Total images: {total_images}")
    print(f"Train images: {len(train_images)} ({train_ratio*100:.1f}%)")
    print(f"Val images: {len(val_images)} ({val_ratio*100:.1f}%)")
    print(f"Test images: {len(test_images)} ({test_ratio*100:.1f}%)")

# Example usage
satellite_dir = r'C:\Users\User\Downloads\imgs_for_model-20241206T180504Z-001\imgs_for_model\augmented_SAT'
mask_dir = r'C:\Users\User\Downloads\imgs_for_model-20241206T180504Z-001\imgs_for_model\augmented_MASK'
output_base_dir = r'C:\Users\User\Downloads\imgs_for_model-20241206T180504Z-001\imgs_for_model\split_dataset'

split_paired_dataset(satellite_dir, mask_dir, output_base_dir)