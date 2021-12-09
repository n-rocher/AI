from tensorflow import keras
from dataset import DrivableAreaDataset, getClipImage
import model as MODEL

from datetime import datetime

CLIP_SIZE = 5
IMG_SIZE = (256, 256)
NUM_CLASSES = 3

BATCH_SIZE = 5
EPOCHS = 100

MAX_DATA_TRAINING = 500000
MAX_DATA_EVALUATION = 500

MODEL_PATH = r"J:\PROJET\IA\BDD100K\models\UNET-MID-CONVLSTM__5-256-256__2021.09.11-18.50_epoch-100_loss-0.21.h5"


# Récupération des paths des fichiers
print("\n> Récupération des fichiers")
train_input_img_paths, train_target_img_paths = getClipImage("images/seq-5-15/train/", "labels/drivable/colormaps/train/")
val_input_img_paths, val_target_img_paths = getClipImage("images/seq-5-15/val/", "labels/drivable/colormaps/val/")

print("  Training :", len(train_input_img_paths), "clips")
print("  Validation :", len(val_input_img_paths), "clips")

# Génération des datasets
print("\n> Génération des datasets")
train_gen = DrivableAreaDataset(BATCH_SIZE, CLIP_SIZE, IMG_SIZE, train_input_img_paths[:MAX_DATA_TRAINING], train_target_img_paths[:MAX_DATA_TRAINING])
val_gen = DrivableAreaDataset(BATCH_SIZE, CLIP_SIZE, IMG_SIZE, val_input_img_paths[:MAX_DATA_EVALUATION], val_target_img_paths[:MAX_DATA_EVALUATION])


# Création du modele
print("\n> Création du modèle")
keras.backend.clear_session()

model = MODEL.UNET_MID_CONVLSTM(CLIP_SIZE, IMG_SIZE, NUM_CLASSES)

if MODEL_PATH:
    model.load_weights(MODEL_PATH)

now_str = datetime.now().strftime("%Y.%m.%d-%H.%M")

MODELS_FOLDER = "models/"
LOGS_FOLDER = MODELS_FOLDER + "logs/"
MODEL_NAME = model.name + "__" + str(CLIP_SIZE) + "-" + str(IMG_SIZE[0]) + "-" + str(IMG_SIZE[1]) + "__" + now_str

callbacks = [
    keras.callbacks.ModelCheckpoint(MODELS_FOLDER + MODEL_NAME + "_epoch-{epoch:02d}_loss-{val_loss:.2f}.h5"),
    keras.callbacks.TensorBoard(log_dir=LOGS_FOLDER + MODEL_NAME, histogram_freq=1)
]

# Entrainement
print("\n> Entrainement")
model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen, callbacks=callbacks)
