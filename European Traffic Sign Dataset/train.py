import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime

import wandb
from tensorflow import keras
from wandb.keras import WandbCallback

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#######################################################
#                   HYPERPARAMETERS                   #
#######################################################
EPOCHS = 20
LEARNING_RATE = 0.001
BATCH_SIZE = 64
IMG_SIZE = (75, 75)

CSV_CLASS = "data/ClassesAEntrainer.csv"

FOLDER = r"F:\European Traffic Sign Dataset\PNG-"

#######################################################
#                        MODEL                        #
#######################################################
def TrafficSignClassifier(img_size, classes):
    model = Sequential(name="TrafficSignClassifier")
    inputShape = img_size + (3,)

    model.add(Conv2D(8, (5, 5), input_shape=inputShape, activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Conv2D(75, (3, 3), padding="same", activation="relu"))
    # model.add(BatchNormalization())
    # model.add(Conv2D(75, (3, 3), padding="same", activation="relu"))
    # model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(classes, activation="softmax"))

    return model

#######################################################
#                 LOAD DATA FUNCTION                  #
#######################################################

def load_data(folder, class_ids):
    data_image = []
    data_label = []

    print("[INFO] " + str(len(class_ids)) +" classes à charger dans le dossier " + FOLDER + folder)

    for index, class_id in enumerate(class_ids):

        # On créé le path du dossier de la classe
        class_path = FOLDER + folder + "/" + str(class_id).zfill(3)

        try:
            # On récupère toutes les images
            images = os.listdir(class_path)[:250]

            compteur_image_ajoutee = 0

            for image in images:

                # On créé le path de l'image
                image_path = class_path + "/" + image

                try:
                    # On charge l'image
                    image = cv2.imread(image_path) / 255.

                    # Adaptive histogram equalization
                    # R, G, B = cv2.split(image)
                    # img_r = cv2.equalizeHist(R)
                    # img_g = cv2.equalizeHist(G)
                    # img_b = cv2.equalizeHist(B)
                    # image = cv2.merge((img_r, img_g, img_b))

                    # On ajoute dans la mémoire
                    data_image.append(image)
                    data_label.append(index) # ON UTILISE L'INDEX POUR LE ONE-HOT ENCODER
                    # data_label.append(class_id)

                    compteur_image_ajoutee+=1

                except KeyboardInterrupt:
                    print("[SHUTING DOWN] Fin du programme...")
                    exit()
                except:
                    print("[ERROR] Impossible d'ouvrir l'image suivante : " + image_path)
                

            print("[INFO] Classe " + str(class_id) + " chargée (" + str(compteur_image_ajoutee) + " images)")

        except Exception as err:
            print(err)
            print("[ERROR] Impossible d'ouvrir le dossier suivant : " + class_path)


    data_image = np.array(data_image)
    data_label = np.array(data_label)

    np.save("data/" + folder + "_data_image", data_image)
    np.save("data/" + folder + "_data_label", data_label)

    return (data_image, data_label)

#######################################################
#                      PROGRAM                        #
#######################################################

if __name__ == '__main__':

    # Weights & Biases
    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    run = wandb.init(project="Traffic Sign recognition", entity="nrocher", config={
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "image_size": IMG_SIZE,
        "dataset": "European Traffic Sign Dataset",
        "model": "TrafficSignClassifier"
    })

    # On charge le fichier des classes à utiliser
    class_file = pd.read_csv(CSV_CLASS)
    class_ids = class_file['Class'].tolist()

    (trainX, trainY) = load_data("Training", class_ids)
    (testX, testY) = load_data("Testing", class_ids)

    # One-Hot Encoding data
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)

    # Image Augmentation
    data_aug = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,
        horizontal_flip=False,
        vertical_flip=False)

    
    # Création du modèle
    model = TrafficSignClassifier(IMG_SIZE, classes=len(class_ids))
    optimizer = Adam(learning_rate=LEARNING_RATE, decay=LEARNING_RATE / EPOCHS)

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint("models/" + now_str + "/" + model.name + "_" + str(IMG_SIZE[0]) + "-" + str(IMG_SIZE[1]) + "_epoch-{epoch:02d}_loss-{val_loss:.2f}_acc_{val_accuracy:.2f}.h5"),
        keras.callbacks.TensorBoard(log_dir="models/" + now_str + "/logs/", histogram_freq=1),
        WandbCallback()
    ]

    fit = model.fit(
        data_aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(testX, testY),
        # use_multiprocessing=True,
        # workers=6,
        callbacks=callbacks)

    # Weights & Biases - END
    run.finish()