import cv2
import glob
import numpy as np
from os.path import join
import matplotlib.pyplot as plt

ROOT_PATH = 'F:\\DIODE\\val\\outdoor\\scene_00022\\scan_00193'

if __name__ == "__main__":

    # get the list of files in lidar directory
    file_names = sorted(glob.glob(join(ROOT_PATH, '*depth.npy')))

    for file in file_names:
        plt.imshow(np.load(file))
        plt.axis('off')
        plt.show()
