from tensorflow import keras
from dataset import DrivableAreaDataset, getClipImage
import model as MODEL

import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img

from datetime import datetime 

from matplotlib import pyplot as plt

CLIP_SIZE = 5
IMG_SIZE = (256, 256)
NUM_CLASSES = 3

BATCH_SIZE = 20
EPOCHS = 100

MAX_DATA_TRAINING = 45
MAX_DATA_EVALUATION = 45

# Récupération des paths des fichiers
print("\n> Récupération des fichiers")
train_input_img_paths, train_target_img_paths = getClipImage(CLIP_SIZE, "images/track/train/", "labels/drivable/colormaps/train/")

i = 1

train_input_img_paths = train_input_img_paths[i : i + BATCH_SIZE]
train_target_img_paths = train_target_img_paths[i : i + BATCH_SIZE]

x = np.zeros((BATCH_SIZE, CLIP_SIZE,) + IMG_SIZE + (3,), dtype="float32")

for j, list_clip_image_path in enumerate(train_input_img_paths): #
    for i, clip_image_path in enumerate(list_clip_image_path):
        x[j, i] = load_img(clip_image_path, target_size=IMG_SIZE)

print(train_input_img_paths[0])

img = load_img(train_target_img_paths[0], target_size=IMG_SIZE, color_mode="grayscale")
img = np.expand_dims(img, 2)
img[img==130]=1
img[img==174]=2



plt.axis('off')
plt.grid(b=None)

fig, axs = plt.subplots(2, 5)
axs[0, 4].imshow(img)
axs[1, 0].imshow(x[0][0]/255.)
axs[1, 1].imshow(x[0][1]/255.)
axs[1, 2].imshow(x[0][2]/255.)
axs[1, 3].imshow(x[0][3]/255.)
axs[1, 4].imshow(x[0][4]/255.)

fig.show()
plt.show()

exit()
















from tensorflow.keras.preprocessing.image import load_img
import numpy as np
path = "labels/drivable/colormaps/val/b1c66a42-6f7d68ca.png"
# path = "labels/keras-tuto.png"


import matplotlib.pyplot as plt










img_size = (200, 347)
# img_size = (720//2, 1280//2)
print(img_size)

img = load_img(path, target_size=img_size, color_mode="grayscale")

print(np.array(img).shape)

# print(len(img))

# plt.imshow(img)

print(np.unique(img))

IMG = np.expand_dims(img, 2)

img2 = IMG

IMG[IMG==130]=1
IMG[IMG==174]=2
plt.imshow(IMG)

print(np.unique(IMG))
print(IMG.shape)

print(np.unique(img2))

plt.show()