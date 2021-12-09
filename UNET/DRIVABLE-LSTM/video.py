import cv2
import os
import numpy as np
from tensorflow import keras
import model as MODEL

CLIP_SIZE = 5
IMG_SIZE = (256, 256)
VISUAL_SIZE = (480, 720)
NUM_CLASSES = 3
VIDEO_PATH = r"J:\PROJET\IA\ROAD FOOTAGE"
MODEL_PATH = r"J:\PROJET\IA\BDD100K\models\UNET-MID-CONVLSTM__5-256-256__2021.09.11-18.50_epoch-100_loss-0.21.h5"

DELAY = 10

# Load model
model = MODEL.UNET_MID_CONVLSTM(CLIP_SIZE, IMG_SIZE, NUM_CLASSES)
model.load_weights(MODEL_PATH)

# Get video in the video folder
videos = os.listdir(VIDEO_PATH)

for video in videos:
    cap = cv2.VideoCapture(VIDEO_PATH + "/" + video)

    if (cap.isOpened() == False):
        print("Error opening video stream or file")
        break

    last_frames = []
    frame_id = -1

    while(cap.isOpened()):

        ret, frame = cap.read()
        reading = True
        if ret == True:

            if reading:
                reading = False
            else:
                reading = True
                continue

            frame_id += 1
            small_frame = cv2.resize(frame, (IMG_SIZE[1], IMG_SIZE[0]), interpolation=cv2.INTER_AREA)

            last_frames.append(small_frame)

            prediction_tab = []

            # Affichage des petite fenetres
            for i in range((CLIP_SIZE-2), -1, -1):
                if len(last_frames) > ((CLIP_SIZE-i) * DELAY + DELAY + 1):
                    image = last_frames[-(((CLIP_SIZE-i) * DELAY) + DELAY)]
                    prediction_tab.insert(0, image)
                    cv2.imshow('Image ' + str(i + 1) + '/5', cv2.resize(image, (VISUAL_SIZE[1] // 3, VISUAL_SIZE[0] // 3), interpolation=cv2.INTER_AREA))

            prediction_tab.append(small_frame)

            if len(prediction_tab) == 5:
                data = np.expand_dims(np.array([ keras.preprocessing.image.img_to_array(f) for f in prediction_tab]), axis=0)
                result_image = model.predict(data)[0]
            else:
                result_image = np.zeros([frame.shape[0], frame.shape[1], 3])

            result_image[:, :, 0] = np.zeros([result_image.shape[0], result_image.shape[1]])
            result_image_uint = (result_image * 255).astype('uint8')

            blur = cv2.blur(result_image_uint, (5, 5))
            green_only = blur[:, :, 1]
            red_only = blur[:, :, 2]

            thresh_red = cv2.threshold(red_only, 128, 255, cv2.THRESH_BINARY)[1]
            thresh_green = cv2.threshold(green_only, 128, 255, cv2.THRESH_BINARY)[1]

            ######################################################################################################
            #                             Recuperation de la plus grande zone verte                              #
            ######################################################################################################
            contours_green, _ = cv2.findContours(thresh_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            img_contour_green = np.zeros_like(thresh_green)

            max_c_l = 0
            max_c = []
            for c in contours_green:
                area = cv2.contourArea(c)
                if area > max_c_l:
                    max_c_l = area
                    max_c = c

            if max_c_l > 0:
                cv2.fillPoly(img_contour_green, pts=[max_c], color=(255, 255, 255))
            ######################################################################################################

            ######################################################################################################
            #                                      Filtrage des zones rouge                                      #
            ######################################################################################################
            all_contours_red, _ = cv2.findContours(thresh_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            img_contour_red = np.zeros_like(thresh_red)

            contours_red = []
            for c in all_contours_red:
                if cv2.contourArea(c) > 2500:
                    contours_red.append(c)

            # cv2.putText(img_contour_red, 'Zone = ' + str(len(contours_red)), (5,25), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1, cv2.LINE_AA)
            # cv2.putText(img_contour_red, ", ".join([ str(cv2.contourArea(c)) for c in contours_red]), (5,50), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1, cv2.LINE_AA)

            cv2.fillPoly(img_contour_red, pts=contours_red, color=(255, 255, 255))
            ######################################################################################################

            # cv2.imshow('Image :', frame)
            cv2.imshow('UNET :', result_image_uint)

            merged_thresh = cv2.merge([np.zeros_like(thresh_green), img_contour_green, img_contour_red])
            cv2.imshow('UNET UPGRADE :', merged_thresh)

            merged_thresh_upsized = cv2.resize(merged_thresh, (VISUAL_SIZE[1], VISUAL_SIZE[0]), interpolation=cv2.INTER_AREA)
            frame = cv2.resize(frame, (VISUAL_SIZE[1], VISUAL_SIZE[0]), interpolation=cv2.INTER_AREA)

            cv2.imshow('Road :', cv2.addWeighted(frame, 1, merged_thresh_upsized, 0.9, 0))

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
