import cv2
import os
import numpy as np
from tensorflow import keras
import model as MODEL

IMG_SIZE = (720, 480)
MODEL_SIZE = (480, 480)
VIDEO_PATH = r"F:\Road Video"
MODEL_PATH = r"J:\PROJET\IA\A-UNET\models\20210929-000046\SequenceAttentionResUNet-F16_MultiDataset_480-480_epoch-08_loss-0.30.h5"

DISPLAY_LAYER = [1, 2, 3, 8]  # Ou True pour tous ou False pour le désactiver

CATEGORIES = {
    1: {"name": "Road", "color": [128, 64, 128]},
    2: {"name": "Lane", "color": [255, 255, 255]},
    3: {"name": "Crosswalk", "color": [200, 128, 128]},
    4: {"name": "Curb", "color": [196, 196, 196]},
    5: {"name": "Sidewalk", "color": [244, 35, 232]},

    6: {"name": "Traffic Light", "color": [250, 170, 30]},
    7: {"name": "Traffic Sign", "color": [220, 220, 0]},

    8: {"name": "Person", "color": [220, 20, 60]},
    9: {"name": "Bicyclist", "color": [255, 0, 0]},
    10: {"name": "Motorcyclist", "color": [255, 0, 100]},

    11: {"name": "Bicycle", "color": [119, 11, 32]},
    12: {"name": "Bus", "color": [0, 60, 100]},
    13: {"name": "Car", "color": [0, 0, 142]},
    14: {"name": "Motorcycle", "color": [0, 0, 230]},
    15: {"name": "Truck", "color": [0, 0, 70]}
}

# Load model
model = keras.models.load_model(MODEL_PATH)

# Get video in the video folder
videos = os.listdir(VIDEO_PATH)

for video in videos:
    cap = cv2.VideoCapture(os.path.join(VIDEO_PATH, video))

    if (cap.isOpened() == False):
        print("Error opening video stream or file")
        break

    while(cap.isOpened()):

        ret, frame = cap.read()
        if ret == True:

            frame = cv2.resize(frame, IMG_SIZE, interpolation=cv2.INTER_AREA)

            img_resized = cv2.resize(frame, MODEL_SIZE, interpolation=cv2.INTER_AREA)

            background_instance = np.zeros_like(img_resized)

            result_image = model.predict(np.expand_dims(img_resized / 255., axis=0))[0]

            background_instance = cv2.cvtColor(background_instance, cv2.COLOR_RGB2BGR)

            num_categories_out = result_image.shape[2]

            if DISPLAY_LAYER is not False:
                for i in range(num_categories_out) if DISPLAY_LAYER == True else DISPLAY_LAYER:
                    cv2.imshow('Resultat ' + str(i) + '/' + str(num_categories_out) + " : " + CATEGORIES[i]["name"], cv2.applyColorMap(np.array(result_image[:, :, i] * 255., dtype = np.uint8), cv2.COLORMAP_JET))

            for key in CATEGORIES:
                if key in [4, 2]: continue
                result_image[:, :, key] = cv2.blur(result_image[:, :, key], (10,10)) 

            cv2.imshow('result_image[: , : , 1] 11 :', result_image[:, :, 1])
            cv2.imshow('result_image[: , : , 1] 22 :', cv2.applyColorMap(np.array(result_image[:, :, 1] * 255., dtype = np.uint8), cv2.COLORMAP_JET))
        
            result_image[result_image <= 0.5] = 0
            result_image[(result_image > 0) & (result_image <= 0.6)] = 0.5
            result_image[result_image > 0.6] = 1
            

            # result_image_uint = (result_image * 255.).astype(np.uint8)

            for key in CATEGORIES:
                background_instance[result_image[:, :, key] == 0.5] = (background_instance[result_image[:, :, key] == 0.5] + CATEGORIES[key]["color"]) / 2
                img_resized[result_image[:, :, key] == 0.5] = (background_instance[result_image[:, :, key] == 0.5] + CATEGORIES[key]["color"]) / 2
                background_instance[result_image[:, :, key] == 1] = CATEGORIES[key]["color"]
                img_resized[result_image[:, :, key] == 1] = CATEGORIES[key]["color"]

            # background_instance = cv2.dilate(background_instance, np.ones((5, 5), np.uint8), iterations=1)
            background_instance = cv2.resize(background_instance, IMG_SIZE, interpolation=cv2.INTER_AREA)
            img_resized = cv2.resize(img_resized, IMG_SIZE, interpolation=cv2.INTER_AREA)

            cv2.imshow('Video :', frame)
            cv2.imshow('Segmentation :', img_resized)
            cv2.imshow('Black Segmentation :', cv2.cvtColor(background_instance, cv2.COLOR_BGR2RGB))

            # IMAGE_W, IMAGE_H = IMG_SIZE
            # dst = np.float32([[int(IMAGE_W * 1 / 4), IMAGE_H], [int(IMAGE_W * 3 / 4), IMAGE_H], [0, 0], [IMAGE_W, 0]])
            # src = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [0, IMAGE_H * 4 / 5], [IMAGE_W, IMAGE_H * 4 / 5]])
            # warped_img = cv2.warpPerspective(img_resized, cv2.getPerspectiveTransform(src, dst), IMG_SIZE)
            # cv2.imshow('warped_img :', warped_img)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
