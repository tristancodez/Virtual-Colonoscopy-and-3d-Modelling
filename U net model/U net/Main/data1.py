import os
import cv2
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from albumentations import HorizontalFlip, VerticalFlip, Rotate

def create_dir(path):
    """ Create a directory if it doesn't exist. """
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path, split=0.2):
    """ Load the images and masks, split into training and validation sets. """
    images = sorted(glob(os.path.join(path, "*", "image", "*.png")))
    masks = sorted(glob(os.path.join(path, "*", "mask", "*.png")))

    split_size = int(len(images) * split)
    train_x, valid_x = train_test_split(images, test_size=split_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=split_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y)

def augment_data(images, masks, save_path, augment=True):
    """ Perform data augmentation and save original and augmented images and masks. """
    H = 512
    W = 512

    create_dir(os.path.join(save_path, "image"))
    create_dir(os.path.join(save_path, "mask"))

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        # Extract directory name and base name
        dir_name = os.path.basename(os.path.dirname(x))
        base_name = os.path.splitext(os.path.basename(x))[0]

        # Read original image and mask
        original_image = cv2.imread(x, cv2.IMREAD_COLOR)
        original_mask = cv2.imread(y, cv2.IMREAD_COLOR)

        # Save original image
        save_image(original_image, base_name, idx, "_original", save_path, "image")
        # Save original mask
        save_image(original_mask, base_name, idx, "_original_mask", save_path, "mask")

        if augment:
            # Apply augmentations
            aug1 = HorizontalFlip(p=1.0)
            augmented1 = aug1(image=original_image, mask=original_mask)
            augmented_image1 = augmented1["image"]
            augmented_mask1 = augmented1["mask"]
            save_image(augmented_image1, base_name, idx, "_augmented1", save_path, "image")
            save_image(augmented_mask1, base_name, idx, "_augmented1_mask", save_path, "mask")

            aug2 = VerticalFlip(p=1.0)
            augmented2 = aug2(image=original_image, mask=original_mask)
            augmented_image2 = augmented2["image"]
            augmented_mask2 = augmented2["mask"]
            save_image(augmented_image2, base_name, idx, "_augmented2", save_path, "image")
            save_image(augmented_mask2, base_name, idx, "_augmented2_mask", save_path, "mask")

            # aug3 = Rotate(limit=45, p=1.0)
            # augmented3 = aug3(image=original_image, mask=original_mask)
            # augmented_image3 = augmented3["image"]
            # augmented_mask3 = augmented3["mask"]
            # save_image(augmented_image3, base_name, idx, "_augmented3", save_path, "image")
            # save_image(augmented_mask3, base_name, idx, "_augmented3_mask", save_path, "mask")

def save_image(image, base_name, idx, suffix, save_path, directory):
    """ Save image with unique filename in specified directory. """
    H = 512
    W = 512

    # Resize image if necessary
    image = cv2.resize(image, (W, H))

    # Generate unique filename
    image_name = f"{base_name}_{idx}{suffix}.jpg"

    # Construct path
    image_path = os.path.join(save_path, directory, image_name)

    # Save image
    cv2.imwrite(image_path, image)

if __name__ == "__main__":
    # Load dataset
    dataset_path = os.path.join("data", "train")
    (train_x, train_y), (valid_x, valid_y) = load_data(dataset_path, split=0.2)

    print("Train: ", len(train_x))
    print("Valid: ", len(valid_x))

    # Create directories
    create_dir("new_data/train/image")
    create_dir("new_data/train/mask")
    create_dir("new_data/valid/image")
    create_dir("new_data/valid/mask")

    # Augment and save training data
    augment_data(train_x, train_y, "new_data/train", augment=True)

    # Note: Uncomment the following line to augment and save validation data
    augment_data(valid_x, valid_y, "new_data/valid", augment=False)
