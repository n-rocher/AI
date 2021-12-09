import os
import cv2
import numpy as np
from shutil import copyfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from multiprocessing.pool import ThreadPool as Pool

pool_size = 10

FOLDER = ["Classes2"]

def copy_file_into_class(folder, new_folder, class_path):
       
    new_class_path = new_folder + "/" + class_path
    
    try:
        os.makedirs(new_class_path)
    except:
        pass
    
    files = os.listdir(folder + "/" + class_path)

    for file in files :
        
        file_path = folder + "/" + class_path + "/" + file
        new_file_path = new_folder + "/" + class_path + "/" + file

        if os.path.isfile(file_path):

            if file.endswith('.ppm'):

                new_file_path = new_folder + "/" + class_path + "/" + os.path.splitext(file)[0] + ".png"
            
                img = mpimg.imread(file_path)

                img = cv2.resize(img, dsize=(75, 75), interpolation=cv2.INTER_LANCZOS4)
  
                mpimg.imsave(new_file_path, img, format="png")

            else:
                pass
                # Si ce n'est pas un fichier PPM on ne le copie pas
                # copyfile(file_path, new_file_path)

        print(class_path + " : " + file + " tranformed and copied !")



pool = Pool(pool_size)

for folder in FOLDER:

    classes_path = os.listdir(folder)

    new_folder = "PNG-RESIZED-" + folder

    try:
        os.makedirs(new_folder)
    except:
        pass

    for class_path in classes_path:
        pool.apply_async(copy_file_into_class, (folder, new_folder, class_path,))

pool.close()
pool.join()