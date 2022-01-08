import os
import cv2
import imageio
import numpy as np

from model import ArgmaxMeanIOU
from tensorflow import keras, argmax

CATEGORIES = {
    1: {"name": "Road", "color": [75, 75, 75]},
    2: {"name": "Lane", "color": [255, 255, 255]},
    3: {"name": "Crosswalk", "color": [200, 128, 128]},
    4: {"name": "Curb", "color": [150, 150, 150]},
    5: {"name": "Sidewalk", "color": [244, 35, 232]},

    6: {"name": "Traffic Light", "color": [250, 170, 30]},
    7: {"name": "Traffic Sign", "color": [255, 255, 0]},

    8: {"name": "Person", "color": [255, 0, 0]},
    9: {"name": "Bicyclist", "color": [150, 150, 100]},
    10: {"name": "Motorcyclist", "color": [20, 50, 100]},

    11: {"name": "Bicycle", "color": [119, 11, 32]},
    12: {"name": "Bus", "color": [255, 15, 147]},
    13: {"name": "Car", "color": [0, 255, 142]},
    14: {"name": "Motorcycle", "color": [0, 0, 230]},
    15: {"name": "Truck", "color": [75, 10, 170]}
}

IMG_SIZE = (720, 480)
VIDEO_PATH = r"C:\Users\thena\Videos\ProCamX\VID_20211225_122657.mp4"
MODEL_PATH = r"C:\Users\thena\Desktop\AI\BiSeNet-V2\models\BiSeNet-V2_MultiDataset_480-704_epoch-25_loss-0.15_miou_0.45.h5"


if __name__ == "__main__":
    segmentation_model = keras.models.load_model(MODEL_PATH, custom_objects={'ArgmaxMeanIOU': ArgmaxMeanIOU})
    segmentation_model_size = segmentation_model.get_layer(index=0).input_shape[0][1:-1][::-1]

    video_capture = cv2.VideoCapture(VIDEO_PATH)

    while video_capture.isOpened() :
        ret, frame = video_capture.read()
        if not ret:
            continue

        img_resized = cv2.resize(frame, segmentation_model_size, interpolation=cv2.INTER_AREA)
        result_segmentation = segmentation_model.predict(np.expand_dims(img_resized / 255., axis=0))[0]

        # Argmax
        result_segmentation = argmax(result_segmentation, axis=-1)
        segmentation = np.zeros(result_segmentation.shape + (3,), dtype=np.uint8)
        for categorie in CATEGORIES.keys():
            segmentation[result_segmentation == categorie] = CATEGORIES[categorie]["color"]

        if segmentation_model_size != (640, 480):
            img_resized = cv2.resize(img_resized, (640, 480), interpolation=cv2.INTER_AREA)
            segmentation = cv2.resize(segmentation, (640, 480), interpolation=cv2.INTER_AREA)

        cv2.imshow("self.EVT_ROAD_IMAGE", cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
        cv2.imshow("self.EVT_SEGMENTATION_IMAGE", segmentation)


    output_path = os.path.dirname(os.path.abspath(VIDEO_PATH))
    imageio.mimsave('D:/downloads/video.gif', image_lst, fps=60)