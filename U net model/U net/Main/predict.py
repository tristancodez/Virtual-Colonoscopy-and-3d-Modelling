import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
import pydicom as dicom
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from glob import glob
from metrics import dice_loss, dice_coef, iou

""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("test")

    """ Loading model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model(r"D:/U net actual/U net/files/model2.h5")

    """ Load the dataset """
    dicom_dir = r"C:\Tcare\Dicom Dataset\430\dicom"
    test_x = glob(os.path.join(dicom_dir, "*.dcm"))
    print(f"Test: {len(test_x)}")
    print("Sample files:", test_x[:10])  # Print the first 10 file paths if available

    """ Loop over the data """
    for x in (test_x):
        """ Extract the names """
        dir_name = x.split(os.sep)[-2]
        name = dir_name + "_" + x.split(os.sep)[-1].split(".")[0]

        """ Read the image """
        image = dicom.dcmread(x).pixel_array
        image = np.expand_dims(image, axis=-1)
        image = image/np.max(image) * 255.0
        x = image/255.0
        x = np.concatenate([x, x, x], axis=-1)
        x = np.expand_dims(x, axis=0)

        """ Prediction """
        mask = model.predict(x)[0]
        mask = mask > 0.5
        mask = mask.astype(np.int32)
        mask = mask * 255

        cat_images = np.concatenate([image, mask], axis=1)
        cv2.imwrite(f"test/{name}.png", cat_images)
