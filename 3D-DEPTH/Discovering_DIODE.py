import glob
import cv2
import numpy as np
from os import path
import matplotlib.pyplot as plt
import tensorflow as tf

ROOT_PATH = 'C:\\Users\\thena\\Desktop\\AI\\3D-DEPTH\\DIODE\\'
SIZE = (640, 480)

def discoverFile(dataset_type: str, outdoor_data: bool = True):

    root_path = path.join(ROOT_PATH, dataset_type, "outdoor" if outdoor_data else "indoor")

    file_names = sorted(glob.glob(path.join(root_path, '*_depth.npy')))

    return file_names


def getImageName(depth_map_path: str, ext : str = ".png") -> str:
    return depth_map_path[:-10] + ext


if __name__ == "__main__":
    depth_map_list = discoverFile("train")
    images_map_list = [getImageName(path) for path in depth_map_list]
    masks_map_list = [getImageName(path, "_depth_mask.npy") for path in depth_map_list]

    for depth_map, image_path, mask in zip(depth_map_list, images_map_list, masks_map_list):

        image_ = cv2.imread(image_path)
        image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
        image_ = cv2.resize(image_, SIZE)
        image_ = tf.image.convert_image_dtype(image_, tf.float32)

        depth_map = np.load(depth_map).squeeze()

        mask = np.load(mask)
        mask = mask > 0

        max_depth = min(300, np.percentile(depth_map, 99))
        depth_map = np.clip(depth_map, 0.1, max_depth)
        depth_map = np.log(depth_map, where=mask)

        depth_map = np.ma.masked_where(~mask, depth_map)

        depth_map = np.clip(depth_map, 0.1, np.log(max_depth))
        depth_map = cv2.resize(depth_map, SIZE)
        depth_map = np.expand_dims(depth_map, axis=2)
        depth_map = tf.image.convert_image_dtype(depth_map, tf.float32)


        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(image_)
        ax2.imshow(depth_map)
        plt.axis('off')
        plt.show()