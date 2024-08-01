import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from metrics import dice_loss, dice_coef, iou
import pydicom
import nibabel as nib

H = 512
W = 512

""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_results(image, mask, y_pred, save_image_path):
    ## i - m - y
    line = np.ones((H, 10, 3)) * 128

    """ Mask """
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)

    """ Predicted Mask """
    y_pred = np.expand_dims(y_pred, axis=-1)    ## (512, 512, 1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)  ## (512, 512, 3)
    y_pred = y_pred * 255

    cat_images = np.concatenate([image, line, mask, line, y_pred], axis=1)
    cv2.imwrite(save_image_path, cat_images)

def load_dicom_series(directory):
    dicom_files = sorted(glob(os.path.join(directory, "*.dcm")))
    dicom_series = [pydicom.dcmread(file) for file in dicom_files]
    return dicom_series

def create_mask(image, model):
    x = image / 255.0
    x = np.expand_dims(x, axis=0)
    y_pred = model.predict(x)[0]
    mask = np.squeeze(y_pred > 0.5, axis=-1).astype(np.uint8) * 255
    return mask

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("results")

    """ Loading model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model("files/model.h5")

    """ Load the DICOM series """
    dicom_directory = "C:/Users/T_Care/Desktop/U net/data/test/2.16.124.113543.6003.1319083618.565.16654.2892285873/"
    dicom_series = load_dicom_series(dicom_directory)
    print(f"Loaded {len(dicom_series)} DICOM files.")

    """ Processing each DICOM image """
    SCORE = []
    for i, dicom in tqdm(enumerate(dicom_series), total=len(dicom_series)):
        """ Extract the name """
        name = f"dicom_{i}"

        """ Convert DICOM to numpy array """
        image = dicom.pixel_array

        """ Create mask prediction """
        mask = create_mask(image, model)

        """ Save the prediction as PNG """
        save_image_path = os.path.join("results", f"{name}.png")
        save_results(image, mask, mask, save_image_path)

        """ Save the mask as NIfTI """
        save_nii_path = os.path.join("results", f"{name}.nii.gz")
        nii_img = nib.Nifti1Image(mask, affine=np.eye(4))  # Assuming identity affine matrix
        nib.save(nii_img, save_nii_path)

        """ Append metrics for evaluation """
        y = (dicom.pixel_array > 0).astype(np.int32)
        y_pred = (mask > 0).astype(np.int32)
        acc_value = accuracy_score(y.flatten(), y_pred.flatten())
        f1_value = f1_score(y.flatten(), y_pred.flatten(), labels=[0, 1], average="binary", zero_division=1)
        jac_value = jaccard_score(y.flatten(), y_pred.flatten(), labels=[0, 1], average="binary", zero_division=1)
        recall_value = recall_score(y.flatten(), y_pred.flatten(), labels=[0, 1], average="binary", zero_division=1)
        precision_value = precision_score(y.flatten(), y_pred.flatten(), labels=[0, 1], average="binary", zero_division=1)
        SCORE.append([name, acc_value, f1_value, jac_value, recall_value, precision_value])

    """ Metrics values """
#     score = np.mean(np.array([s[1:] for s in SCORE]), axis=0)
#    # print(f"Accuracy: {score[0]:0.5f}")
#     print(f"F1: {score[1]:0.5f}")
#     print(f"Jaccard: {score[2]:0.5f}")
#     print(f"Recall: {score[3]:0.5f}")
#     print(f"Precision: {score[4]:0.5f}")

#     df = pd.DataFrame(SCORE, columns=["Image", "Accuracy", "F1", "Jaccard", "Recall", "Precision"])
#     df.to_csv("files/score.csv")
