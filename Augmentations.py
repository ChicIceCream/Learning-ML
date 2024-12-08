import os
import cv2
import numpy as np
import albumentations as A

def augment_images(satellite_dir, mask_dir, output_satellite_dir, output_mask_dir):
    # Create output directories if they don't exist
    os.makedirs(output_satellite_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    # Collect image paths
    satellite_images = sorted([f for f in os.listdir(satellite_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    mask_images = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    # Ensure we have matching numbers of satellite and mask images
    assert len(satellite_images) == len(mask_images), "Mismatch in number of satellite and mask images"

    # Define augmentation transforms
    transforms = [
        None,  # Original images
        A.Compose([
            A.HorizontalFlip(p=1.0)
        ], is_check_shapes=False),
        A.Compose([
            A.VerticalFlip(p=1.0)
        ], is_check_shapes=False)
    ]

    # Counter for output image numbering
    counter = 0

    # Process each original image
    for sat_img_name, mask_img_name in zip(satellite_images, mask_images):
        # Read images
        satellite_path = os.path.join(satellite_dir, sat_img_name)
        mask_path = os.path.join(mask_dir, mask_img_name)
        
        # Read satellite image
        satellite_img = cv2.imread(satellite_path)
        if satellite_img is None:
            print(f"Warning: Could not read {satellite_path}")
            continue
        
        # Read mask image
        mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask_img is None:
            print(f"Warning: Could not read {mask_path}")
            continue

        # Ensure mask is the same size as satellite image
        if satellite_img.shape[:2] != mask_img.shape[:2]:
            mask_img = cv2.resize(mask_img, (satellite_img.shape[1], satellite_img.shape[0]))

        # Apply original image first
        sat_output_name = f'{counter:04d}_satellite.png'
        mask_output_name = f'{counter:04d}_mask.png'
        cv2.imwrite(os.path.join(output_satellite_dir, sat_output_name), satellite_img)
        cv2.imwrite(os.path.join(output_mask_dir, mask_output_name), mask_img)
        counter += 1

        # Apply each transform
        for transform in transforms[1:]:
            # Apply augmentation
            augmented = transform(image=satellite_img, mask=mask_img)
            aug_satellite = augmented['image']
            aug_mask = augmented['mask']

            # Save augmented images
            sat_output_name = f'{counter:04d}_satellite.png'
            mask_output_name = f'{counter:04d}_mask.png'

            cv2.imwrite(os.path.join(output_satellite_dir, sat_output_name), aug_satellite)
            cv2.imwrite(os.path.join(output_mask_dir, mask_output_name), aug_mask)

            counter += 1

    print(f"Augmentation complete. Total images generated: {counter}")
    print(f"This includes: {len(satellite_images)} original images")
    print(f"               {len(satellite_images)} horizontally flipped images")
    print(f"               {len(satellite_images)} vertically flipped images")

# Example usage
satellite_dir = r'C:\Users\User\Downloads\imgs_for_model-20241206T180504Z-001\imgs_for_model\SAT'
mask_dir = r'C:\Users\User\Downloads\imgs_for_model-20241206T180504Z-001\imgs_for_model\MASK'
output_satellite_dir = r'C:\Users\User\Downloads\imgs_for_model-20241206T180504Z-001\imgs_for_model\augmented_SAT'
output_mask_dir = r'C:\Users\User\Downloads\imgs_for_model-20241206T180504Z-001\imgs_for_model\augmented_MASK'

augment_images(satellite_dir, mask_dir, output_satellite_dir, output_mask_dir)