import cv2
import os
import numpy as np
from tensorflow import keras
from utils import Mean_IOU_tensorflow_2

IMG_SIZE = (256, 256) # (208, 352)
VIDEO_PATH = r"J:\PROJET\IA\ROAD FOOTAGE"
MODEL_PATH = r"J:\PROJET\IA\BDD100K\models\UNET_256-256_STATIC_DRIVABLE_segmentation_20210906-172845_epoch-11_loss-0.12.h5"

# Load model
model = keras.models.load_model(MODEL_PATH, custom_objects={'Mean_IOU_tensorflow_2': Mean_IOU_tensorflow_2})

# Get video in the video folder
videos = os.listdir(VIDEO_PATH)

for video in videos:
    cap = cv2.VideoCapture(VIDEO_PATH + "/" + video)

    if (cap.isOpened() == False):
        print("Error opening video stream or file")
        break

    while(cap.isOpened()):

        ret, frame = cap.read()
        reading = True
        if ret == True:

            if reading:
                reading = False
            else:
                reading = True
                continue

            img_resized = cv2.resize(frame, (IMG_SIZE[1], IMG_SIZE[0]), interpolation=cv2.INTER_AREA)

            result_image = model.predict(np.expand_dims(keras.preprocessing.image.img_to_array(img_resized), axis=0))[0]
            # result_image = np.zeros([img_resized.shape[0], img_resized.shape[1], 3])

            result_image[:, :, 0] = np.zeros([result_image.shape[0], result_image.shape[1]])
            result_image_uint = ((result_image > 0.5) * 255).astype('uint8')

            blur = cv2.blur(result_image_uint, (5, 5))
            green_only = blur[:, :, 1]
            red_only = blur[:, :, 2]

            ######################################################################################################
            #                             Recuperation de la plus grande zone verte                              #
            ######################################################################################################
            contours_green, _ = cv2.findContours(green_only, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            img_contour_green = np.zeros_like(green_only)

            max_c_l = 0
            max_c = []
            for c in contours_green:
                area = cv2.contourArea(c)
                if area > max_c_l:
                    max_c_l = area
                    max_c = c

            if max_c_l > 0:

                cv2.fillPoly(img_contour_green, pts=[max_c], color=(255,255,255))


            cv2.imshow('GREEN NOT DILATED :', img_contour_green)
            img_contour_green = cv2.dilate(img_contour_green, (15,15), 5)
            cv2.imshow('GREEN DILATED :', img_contour_green)

            ######################################################################################################
           

            ######################################################################################################
            #                                      Filtrage des zones rouge                                      #
            ######################################################################################################
            all_contours_red, _ = cv2.findContours(red_only, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            img_contour_red = np.zeros_like(red_only)

 
            contours_red = []
            for c in all_contours_red:
                if cv2.contourArea(c) > 500:
                    contours_red.append(c)

            # cv2.putText(img_contour_red, 'Zone = ' + str(len(contours_red)), (5,25), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1, cv2.LINE_AA)
            # cv2.putText(img_contour_red, ", ".join([ str(cv2.contourArea(c)) for c in contours_red]), (5,50), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1, cv2.LINE_AA)

            cv2.fillPoly(img_contour_red, pts=contours_red, color=(255,255,255))
            ######################################################################################################

            # cv2.imshow('Image :', frame)
            cv2.imshow('UNET :', result_image_uint)

            merged_thresh = cv2.merge([np.zeros_like(img_contour_green), img_contour_green, img_contour_red])
            cv2.imshow('UNET UPGRADE :', merged_thresh)

            merged_thresh_upsized = cv2.resize(merged_thresh, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)

            cv2.imshow('Road :', cv2.addWeighted(frame, 1, merged_thresh_upsized, 0.9, 0))

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
