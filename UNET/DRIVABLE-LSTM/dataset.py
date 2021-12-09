import os
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img


class DrivableAreaDataset(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, clip_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.clip_size = clip_size
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

        x = np.zeros((self.batch_size, self.clip_size,) + self.img_size + (3,), dtype="float32")
        for j, list_clip_image_path in enumerate(batch_input_img_paths):
            for i, clip_image_path in enumerate(list_clip_image_path):
                x[j, i] = load_img(clip_image_path, target_size=self.img_size)

        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            img = np.expand_dims(img, 2)
            
            # Conversion des couleurs des colormaps (0=0=noir, 130=1=rouge, 174=2=bleu)
            img[img==130]=1
            img[img==174]=2

            y[j] = img
            
        return x, y


def getClipImage(clip_path, masks_path):

    list_masks = os.listdir(masks_path)
    ext_mask = os.path.splitext(list_masks[0])[1]

    list_masks = [ os.path.splitext(mask_name)[0] for mask_name in list_masks]
    list_clips = [ clip_name for clip_name in os.listdir(clip_path) if clip_name in list_masks]

    # Récupération des masques qui ont le clip disponible 
    list_masks = [os.path.join(masks_path, name + ext_mask) for name in list_clips]
    
    # On sélectionne les 4 images avant l'image du masque pour chacun des clips disponibles 
    list_clips = [[ os.path.join(clip_path, clip_name, file_name) for file_name in os.listdir(os.path.join(clip_path, clip_name)) ] for clip_name in list_clips]

    return list_clips, list_masks