import os
import cv2
import numpy as np
from tensorflow import keras
from dataset import DrivableAreaDataset, getImagesAndMasksPath
import PIL
import cv2
import matplotlib.pyplot as plt

# from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img


img_size = (368, 640)
num_classes = 3
batch_size = 6
epochs = 20

IMG_SIZE = (368, 640)
VIDEO_PATH = r"J:\PROJET\IA\ROAD FOOTAGE"
MODEL_PATH = r"J:\PROJET\IA\BDD100K\models\UNET_DRIVABLE_segmentation_20210903-004809.h5"

MAX_DATA = 50


# Récupération des paths des fichiers
print("\n> Récupération des fichiers")
val_input_img_paths, val_target_img_paths = getImagesAndMasksPath("images/100k/val/", "labels/drivable/colormaps/val/")

# Génération des datasets
print("\n> Génération des datasets")
val_gen = DrivableAreaDataset(batch_size, img_size, val_input_img_paths[:MAX_DATA], val_target_img_paths[:MAX_DATA])

# Load model
model = keras.models.load_model(MODEL_PATH)

# Predict from dataset
val_preds = model.predict(val_gen)

print("DONE")


def display_mask(i):
    """Quick utility to display a model's prediction."""

    print(val_preds[i])
    plt.imshow(val_preds[i])
    plt.show()

    mask = np.argmax(val_preds[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    plt.imshow(keras.preprocessing.image.array_to_img(mask))
    plt.show()



# Display results for validation image #10
i = 1

# Display input image
plt.imshow(load_img(val_input_img_paths[i]))
plt.show()


# Display ground-truth target mask
plt.imshow(load_img(val_target_img_paths[i]))
plt.show()

# Display mask predicted by our model
display_mask(i)  # Note that the model only sees inputs at 150x150.