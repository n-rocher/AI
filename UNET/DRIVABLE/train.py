import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

from dataset import DrivableAreaDataset, getImagesAndMasksPath
import model as MODEL

from datetime import datetime 

from matplotlib import pyplot as plt

img_size = (256, 256)
num_classes = 3
batch_size = 20
epochs = 100

MAX_DATA_TRAINING = 2000
MAX_DATA_EVALUATION = 1000

# Récupération des paths des fichiers
print("\n> Récupération des fichiers")
train_input_img_paths, train_target_img_paths = getImagesAndMasksPath("images/100k/train/", "labels/drivable/colormaps/train/")
val_input_img_paths, val_target_img_paths = getImagesAndMasksPath("images/100k/val/", "labels/drivable/colormaps/val/")

# Génération des datasets
print("\n> Génération des datasets")
train_gen = DrivableAreaDataset(batch_size, img_size, train_input_img_paths[:MAX_DATA_TRAINING], train_target_img_paths[:MAX_DATA_TRAINING])
val_gen = DrivableAreaDataset(batch_size, img_size, val_input_img_paths[:MAX_DATA_EVALUATION], val_target_img_paths[:MAX_DATA_EVALUATION])

# Création du modele
print("\n> Création du modèle")
keras.backend.clear_session()

model = MODEL.UNET(img_size, num_classes)

now_str = datetime.now().strftime("%Y%m%d-%H%M%S")

callbacks = [
    keras.callbacks.ModelCheckpoint("models/UNET_" + str(img_size[0])  +  "-" + str(img_size[1]) + "_STATIC_DRIVABLE_segmentation_" + now_str + "_epoch-{epoch:02d}_loss-{val_loss:.2f}.h5"),
    keras.callbacks.TensorBoard(log_dir="models/logs/UNET_" + str(img_size[0])  +  "-" + str(img_size[1]) + "_STATIC_DRIVABLE_segmentation_" + now_str , histogram_freq=1)
]

# Entrainement
print("\n> Entrainement")
model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)