import os
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img


class DrivableAreaDataset(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size

        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]

        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            x[j] = load_img(path, target_size=self.img_size)

        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            img = np.expand_dims(img, 2)
            
            # Conversion des couleurs des colormaps (0=0=noir, 130=1=rouge, 174=2=bleu)
            img[img==130]=1
            img[img==174]=2

            y[j] = img
            
        return x, y


def getImagesAndMasksPath(images_path, masks_path):
    input_train_img_paths = sorted([os.path.join(images_path, fname) for fname in os.listdir(images_path) if fname.endswith(".jpg")])
    target_train_img_paths = sorted([os.path.join(masks_path, fname) for fname in os.listdir(masks_path) if fname.endswith(".png")])
    return input_train_img_paths, target_train_img_paths