import os
import sys
import cv2
import wandb

import tensorflow as tf
from tensorflow.keras import layers
from wandb.keras import WandbCallback

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from model import DepthEstimationModel
from datagenerator import DataGenerator 


if __name__ == "__main__":

    tf.random.set_seed(123)
 
    annotation_folder = r"./DIODE/"
    if not os.path.exists(annotation_folder):
        annotation_zip = tf.keras.utils.get_file(
            "val.tar.gz",
            cache_subdir=os.path.abspath("."),
            origin="http://diode-dataset.s3.amazonaws.com/val.tar.gz",
            extract=True,
        )


    path = annotation_folder + "train/outdoor"

    filelist = []

    for root, dirs, files in os.walk(path):
        for file in files:
            filelist.append(os.path.join(root, file))

    filelist.sort()
    data = {
        "image": [x for x in filelist if x.endswith(".png")],
        "depth": [x for x in filelist if x.endswith("_depth.npy")],
        "mask": [x for x in filelist if x.endswith("_depth_mask.npy")],
    }

    df = pd.DataFrame(data)

    df = df.sample(frac=1, random_state=42)

    HEIGHT = 256
    WIDTH = 256
    LR = 0.0002
    EPOCHS = 30
    BATCH_SIZE = 32

    # Weights & Biases
    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    run = wandb.init(project="3D-DEPTH", entity="nrocher", config={
        "learning_rate": LR,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "image_size": (HEIGHT, WIDTH),
    })

    def visualize_depth_map(samples, test=False, model=None):
        input, target = samples
        # cmap = plt.cm.jet
        # cmap.set_bad(color="black")

        if test:
            pred = model.predict(input)
            fig, ax = plt.subplots(6, 3, figsize=(50, 50))
            for i in range(6):
                ax[i, 0].imshow((input[i].squeeze()))
                ax[i, 1].imshow((target[i].squeeze())) #, cmap=cmap)
                ax[i, 2].imshow((pred[i].squeeze())) #, cmap=cmap)

        else:
            fig, ax = plt.subplots(6, 2, figsize=(50, 50))
            for i in range(6):
                ax[i, 0].imshow((input[i].squeeze()))
                ax[i, 1].imshow((target[i].squeeze())) #, cmap=cmap)

        plt.show()




    # visualize_samples = next(
    #     iter(DataGenerator(data=df, batch_size=6, dim=(HEIGHT, WIDTH)))
    # )
    # # visualize_depth_map(visualize_samples)



    # depth_vis = np.flipud(visualize_samples[1][1].squeeze())  # target
    # img_vis = np.flipud(visualize_samples[0][1].squeeze())  # input

    # fig = plt.figure(figsize=(15, 10))
    # ax = plt.axes(projection="3d")

    # STEP = 3
    # for x in range(0, img_vis.shape[0], STEP):
    #     for y in range(0, img_vis.shape[1], STEP):
    #         ax.scatter(
    #             [depth_vis[x, y]] * 3,
    #             [y] * 3,
    #             [x] * 3,
    #             c=tuple(img_vis[x, y, :3] / 255),
    #             s=3,
    #         )
    #     ax.view_init(45, 135)


    optimizer = tf.keras.optimizers.Adam(learning_rate=LR, amsgrad=False)
    model = DepthEstimationModel(width=WIDTH)

    # Define the loss function
    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

    # Compile the model
    model.compile(optimizer, loss=cross_entropy)

    model.build(input_shape=(None, HEIGHT, WIDTH, 3))

    model.summary()

    train_loader = DataGenerator(data=df.reset_index(drop="true"), batch_size=BATCH_SIZE, dim=(HEIGHT, WIDTH))
    validation_loader = DataGenerator(data=df[:250].reset_index(drop="true"), batch_size=BATCH_SIZE, dim=(HEIGHT, WIDTH))
   
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint("models/" + now_str + "/" + model.name + "_" + str(HEIGHT) + "-" + str(WIDTH) + "_epoch-{epoch:02d}_val-loss-{val_loss:.2f}.h5"),
        tf.keras.callbacks.TensorBoard(log_dir="models/" + now_str + "/logs/", histogram_freq=1),
        WandbCallback()
    ]

    # Entrainement
    print("\n> Entrainement")

    model.fit(
        train_loader,
        epochs=EPOCHS,
        validation_data=validation_loader,
        use_multiprocessing=True,
        workers=6,
        callbacks=callbacks
    )

    # Weights & Biases - END
    run.finish()