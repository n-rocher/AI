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