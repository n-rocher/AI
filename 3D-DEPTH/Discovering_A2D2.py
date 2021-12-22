import cv2
import glob
import numpy as np
from tqdm import tqdm
from numpy import load
import multiprocessing
from os.path import join
import matplotlib.pyplot as plt
from inpainting import fill_depth_colorization

ROOT_PATH = 'F:\\camera_lidar\\'

def map_lidar_points_to_map(image_orig, lidar, pixel_size=10):

    image = np.zeros(image_orig.shape[:-1])
    
    # get rows and cols
    rows = (lidar['pcloud_attr.row'] + 0.5).astype(np.int)
    cols = (lidar['pcloud_attr.col'] + 0.5).astype(np.int)
  
    # get distances
    distances = lidar['pcloud_attr.distance']  
    # determine point colours from distance
    pixel_rowoffs = np.indices([pixel_size, pixel_size])[0] - pixel_size // 2
    pixel_coloffs = np.indices([pixel_size, pixel_size])[1] - pixel_size // 2
    canvas_rows = image.shape[0]
    canvas_cols = image.shape[1]
   
    for i in range(len(rows)):
        pixel_rows = np.clip(rows[i] + pixel_rowoffs, 0, canvas_rows - 1)
        pixel_cols = np.clip(cols[i] + pixel_coloffs, 0, canvas_cols - 1)
        image[pixel_rows, pixel_cols] = distances[i]
 
    return image.astype(np.uint8)


def extract_image_file_name_from_lidar_file_name(file_name_lidar):
    file_name_image = file_name_lidar.split('\\')
    file_name_image = file_name_image[-1].split('.')[0]
    file_name_image = file_name_image.split('_')
    file_name_image = file_name_image[0] + '_' + 'camera_' + file_name_image[2] + '_' + file_name_image[3]
    return file_name_image


def processImageDepth(lidar_file):

    # Image path
    seq_name = lidar_file.split('\\')[2]
    file_name_image_end = extract_image_file_name_from_lidar_file_name(lidar_file)
    file_name_image = join(ROOT_PATH + seq_name, 'camera/cam_front_center/', file_name_image_end + '.png')

    # Loading data
    image = cv2.cvtColor(cv2.imread(file_name_image), cv2.COLOR_BGR2RGB)
    data = load(lidar_file)

    distance_map_points = map_lidar_points_to_map(image, data)

    SIZE = (480, 360) # (1920, 1080)

    image = cv2.resize(image, SIZE, interpolation=cv2.INTER_CUBIC)
    distance_map_points = cv2.resize(distance_map_points, SIZE, interpolation=cv2.INTER_CUBIC)

    data = fill_depth_colorization(image, distance_map_points)

    np.save(join(ROOT_PATH, file_name_image_end), data)

    return True

    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    # ax1.imshow(image)
    # ax2.imshow(distance_map_points)
    # ax3.imshow(data)
    # plt.axis('off')
    # plt.show()

if __name__ == "__main__":

    # get the list of files in lidar directory
    file_names = sorted(glob.glob(join(ROOT_PATH, '*/lidar/cam_front_center/*.npz')))

    with multiprocessing.Pool(8) as pool:
        r = list(tqdm(pool.imap(processImageDepth, file_names), total=len(file_names)))
