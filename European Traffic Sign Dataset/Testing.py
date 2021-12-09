import cv2
import numpy as np
import pandas as pd
from tensorflow import keras

#######################################################
#                      PARAMETRES                     #
#######################################################
CSV_CLASS = "ClassesAEntrainer.csv"
MODEL_FOLDER = "Models"
MODEL_NAME = "21-08-2021-01-45-23"
TEST_FILE = "test.jpg"

#######################################################
#                        MODEL                        #
#######################################################
model = keras.models.load_model(MODEL_FOLDER + '/' + MODEL_NAME)


#######################################################
#                      PROGRAM                        #
#######################################################


# On charge le fichier des classes à utiliser
class_file = pd.read_csv(CSV_CLASS)

# On charge l'image a tester
image = cv2.imread(TEST_FILE)
image = cv2.resize(image, (75, 75), interpolation = cv2.INTER_AREA)

# Adaptive histogram equalization
R, G, B = cv2.split(image)
img_r = cv2.equalizeHist(R)
img_g = cv2.equalizeHist(G)
img_b = cv2.equalizeHist(B)
image = cv2.merge((img_r, img_g, img_b))

image = np.array([image])

result = model.predict(image)
result = result[0]

max_index_col = np.argmax(result, axis=0)

print("\n[INFO] Resultat : '" + class_file.iloc[max_index_col]["Name"] + "' P=" + str(result[max_index_col]))