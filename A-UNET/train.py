from tensorflow import keras
from tensorflow.keras.optimizers import Adam

from dataset import SequenceDataset
import model as MODEL

from datetime import datetime
from sanitycheck import SanityCheck

IMG_SIZE = (480, 480)
CLIP_SIZE = 5
BATCH_SIZE = 1
EPOCHS = 100

if __name__ == '__main__':

    # Génération des datasets
    print("\n> Génération des datasets")
    train_gen = SequenceDataset(CLIP_SIZE, BATCH_SIZE, IMG_SIZE, "F:\\BDD100K\\images\\SEQ-5-5\\train\\", "F:\\BDD100K\\labels\\drivable\\colormaps\\train\\")
    val_gen = SequenceDataset(CLIP_SIZE, BATCH_SIZE, IMG_SIZE, "F:\\BDD100K\\images\\SEQ-5-5\\val\\", "F:\\BDD100K\\labels\\drivable\\colormaps\\val\\")

    # Création du modele
    print("\n> Création du modèle")
    keras.backend.clear_session()

    model = MODEL.Sequence_Attention_ResUNet(CLIP_SIZE, IMG_SIZE, NUM_CLASSES=train_gen.classes())
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model.summary()

    # model.load_weights(r"J:\PROJET\IA\A-UNET\models\20210927-234830\AttentionResUNet-16_MultiDataset_480-480_epoch-11_loss-0.29.h5")

    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    callbacks = [
        SanityCheck(val_gen.__getitem__(0), output="models/" + now_str + "/check/", regulator=500),
        keras.callbacks.ModelCheckpoint("models/" + now_str + "/" + model.name + "_" + train_gen.name() + "_" + str(IMG_SIZE[0]) + "-" + str(IMG_SIZE[1]) + "_epoch-{epoch:02d}_loss-{val_loss:.2f}.h5"),
        keras.callbacks.TensorBoard(log_dir="models/" + now_str + "/logs/", histogram_freq=1)
    ]

    # Entrainement
    print("\n> Entrainement")
    model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        use_multiprocessing=True,
        workers=6,
        callbacks=callbacks
    )
