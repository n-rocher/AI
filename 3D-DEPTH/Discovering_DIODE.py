import glob
import cv2
import numpy as np
from os import path
import matplotlib.pyplot as plt

ROOT_PATH = 'F:\\DIODE\\'

def discoverFile(dataset_type: str, outdoor_data: bool = True):

    root_path = path.join(ROOT_PATH, dataset_type, "outdoor" if outdoor_data else "indoor")

    file_names = sorted(glob.glob(path.join(root_path, '*/*/*_depth.npy')))

    return file_names


def getImageName(depth_map_path: str) -> str:
    return depth_map_path[:-10] + ".png"


if __name__ == "__main__":
    depth_map_list = discoverFile("train")
    images_map_list = [getImageName(path) for path in depth_map_list]

    print(len(depth_map_list))
    print(len(images_map_list))
    
    print(depth_map_list[0], images_map_list[0])

    for depth, image in zip(depth_map_list, images_map_list):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(cv2.imread(image))
        ax2.imshow(np.load(depth))
        plt.axis('off')
        plt.show()