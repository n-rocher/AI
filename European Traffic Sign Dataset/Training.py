import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
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

CSV_CLASS = "ClassesAEntrainer.csv"

TRAINING_FOLDER = "PNG-RESIZED-Training"
TESTING_FOLDER = "PNG-RESIZED-Testing"
MODEL_FOLDER = "Models"

#######################################################
#                        MODEL                        #
#######################################################
def TrafficSignClassifier(width, height, depth, classes):
    model = Sequential()
    inputShape = (height, width, depth)

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

    model.add(Conv2D(75, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(75, (3, 3), padding="same", activation="relu"))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(classes, activation="softmax"))

    return model

#######################################################
#                 LOAD DATA FUNCTION                  #
#######################################################

def load_data(folder, class_ids):
    data_image = []
    data_label = []

    print("[INFO] " + str(len(class_ids)) +" classes à charger dans le dossier " + folder)

    for index, class_id in enumerate(class_ids):

        # On créé le path du dossier de la classe
        class_path = folder + "/" + str(class_id).zfill(3)

        try:
            # On récupère toutes les images
            images = os.listdir(class_path)

            compteur_image_ajoutee = 0

            for image in images:

                # On créé le path de l'image
                image_path = class_path + "/" + image

                try:
                    # On charge l'image
                    image = cv2.imread(image_path)

                    # Adaptive histogram equalization
                    R, G, B = cv2.split(image)
                    img_r = cv2.equalizeHist(R)
                    img_g = cv2.equalizeHist(G)
                    img_b = cv2.equalizeHist(B)
                    image = cv2.merge((img_r, img_g, img_b))

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

    return (np.array(data_image), np.array(data_label))

#######################################################
#                      PROGRAM                        #
#######################################################

try:
    # On charge le fichier des classes à utiliser
    class_file = pd.read_csv(CSV_CLASS)
    class_ids = class_file['Class'].tolist()
    print("\n[INFO] Liste des classe chargée")

    (trainX, trainY) = load_data(TRAINING_FOLDER, class_ids)
    print("\n[INFO] Donnée d'entrainement chargée")

    (testX, testY) = load_data(TESTING_FOLDER, class_ids)
    print("\n[INFO] Donnée de test chargée")

    print("\n[INFO] Normalizing data -> skipped")
    # trainX = trainX.astype("float32") / 255.0
    # testX = testX.astype("float32") / 255.0

    print("\n[INFO] One-Hot Encoding data")
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)

    data_aug = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,
        horizontal_flip=False,
        vertical_flip=False)

    model = TrafficSignClassifier(width=75, height=75, depth=3, classes=len(class_ids))
    optimizer = Adam(learning_rate=LEARNING_RATE, decay=LEARNING_RATE / (EPOCHS))

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    fit = model.fit_generator(
        data_aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(testX, testY),
        verbose=1)

    model_name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    model.save(MODEL_FOLDER + "/" + model_name)

    print("[INFO] Modele sauvegardé : " + model_name)

except KeyboardInterrupt:
    print("[SHUTING DOWN] Fin du programme...")
    exit()