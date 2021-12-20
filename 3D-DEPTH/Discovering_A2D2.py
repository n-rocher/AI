from numpy import load
import matplotlib.pyplot as plt
import numpy as np
import cv2

NPZ_FILE = r"F:\camera_lidar\20180810_150607\lidar\cam_front_center\20180810150607_lidar_frontcenter_000000060.npz"
IMAGE_FILE = r"F:\camera_lidar\20180810_150607\camera\cam_front_center\20180810150607_camera_frontcenter_000000060.png"

image = cv2.imread(IMAGE_FILE)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

data = load(NPZ_FILE)

print(list(data.keys()))

lst = data.files
for item in lst:
    print(item)
    print(data[item])
    print()


import open3d as o3

# Create array of RGB colour values from the given array of reflectance values
def colours_from_reflectances(reflectances):
    return np.stack([reflectances, reflectances, reflectances], axis=1)
    

def create_open3d_pc(lidar, cam_image=None):
    # create open3d point cloud
    pcd = o3.geometry.PointCloud()
    
    # assign point coordinates
    pcd.points = o3.utility.Vector3dVector(lidar['pcloud_points'])
    
    # assign colours
    if cam_image is None:
        median_reflectance = np.median(lidar['pcloud_attr.reflectance'])
        colours = colours_from_reflectances(lidar['pcloud_attr.reflectance']) / (median_reflectance * 5)
        
        # clip colours for visualisation on a white background
        colours = np.clip(colours, 0, 0.75)
    else:
        rows = (lidar['pcloud_attr.row'] + 0.5).astype(np.int)
        cols = (lidar['pcloud_attr.col'] + 0.5).astype(np.int)
        colours = cam_image[rows, cols, :] / 255.0
        
    pcd.colors = o3.utility.Vector3dVector(colours)
    
    return pcd

pcd_front_center = create_open3d_pc(data, image)

o3.visualization.draw_geometries([pcd_front_center])

def hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v
    
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q

def map_lidar_points_onto_image(image_orig, lidar, pixel_size=3, pixel_opacity=1):
    image = np.copy(image_orig)
    
    # get rows and cols
    rows = (lidar['pcloud_attr.row'] + 0.5).astype(np.int)
    cols = (lidar['pcloud_attr.col'] + 0.5).astype(np.int)
  
    # lowest distance values to be accounted for in colour code
    MIN_DISTANCE = np.min(lidar['pcloud_attr.distance'])
    # largest distance values to be accounted for in colour code
    MAX_DISTANCE = np.max(lidar['pcloud_attr.distance'])

    # get distances
    distances = lidar['pcloud_attr.distance']  
    # determine point colours from distance
    colours = (distances - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)
    colours = np.asarray([np.asarray(hsv_to_rgb(0.75 * c, \
                        np.sqrt(pixel_opacity), 1.0)) for c in colours])
    pixel_rowoffs = np.indices([pixel_size, pixel_size])[0] - pixel_size // 2
    pixel_coloffs = np.indices([pixel_size, pixel_size])[1] - pixel_size // 2
    canvas_rows = image.shape[0]
    canvas_cols = image.shape[1]
    for i in range(len(rows)):
        pixel_rows = np.clip(rows[i] + pixel_rowoffs, 0, canvas_rows - 1)
        pixel_cols = np.clip(cols[i] + pixel_coloffs, 0, canvas_cols - 1)
        image[pixel_rows, pixel_cols, :] = \
                (1. - pixel_opacity) * \
                np.multiply(image[pixel_rows, pixel_cols, :], \
                colours[i]) + pixel_opacity * 255 * colours[i]
    return image.astype(np.uint8)


image = map_lidar_points_onto_image(image, data)


plt.fig = plt.figure(figsize=(20, 20))
plt.imshow(image)
plt.axis('off')
plt.show()



from inpainting import fill_depth_colorization


# Grand tableau avec les distances pour chaque index data['pcloud_attr.distance']
# A remap en H*W avec uniquement les points ou il y a des données comme dans la fonction au dessus

data = fill_depth_colorization(image, data['pcloud_attr.distance'])

plt.fig = plt.figure(figsize=(20, 20))
plt.imshow(data)
plt.axis('off')
plt.show()