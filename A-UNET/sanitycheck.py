from tensorflow import keras
import numpy as np
import os
from dataset import DrivableAreaDataset, getImagesAndMasksPath
import matplotlib.pyplot as plt


class SanityCheck(keras.callbacks.Callback):

    def __init__(self, test_data, output="./", regulator=200):
        super(SanityCheck, self).__init__()
        self.test_data = test_data
        self.output = output
        self.regulator = regulator

    id_batch = -1
    epoch = 0

    DEBUG = False

    def on_epoch_end(self, epoch, logs=None):
        if self.id_batch > 25 :
            self.process_test()
        
        self.id_batch = -1
        self.epoch += 1

    def on_train_batch_end(self, batch, logs=None):
        self.id_batch += 1
        if self.id_batch > 25 and self.id_batch % self.regulator == 0:
            self.process_test()

    def process_test(self):
        os.makedirs(self.output, exist_ok=True)

        imgs, masks = self.test_data

        result = []
        for img_i, mask_i in zip(imgs, masks):
            if self.DEBUG == True:
                result_image = np.zeros([img_i.shape[0], img_i.shape[1]])
            else:
                result_image = self.model.predict(np.expand_dims(img_i, axis=0))[0]

            temp = [img_i, mask_i.astype(int)]

            for i in range(1, result_image.shape[2]):
                temp.append((result_image[:, :, i] * 255.).astype(int))

            result.append(temp)

        plt.rcParams["figure.figsize"] = (7.5, 20)
        titles = ['Road', "Truth"] + list(range(1, len(result[0])-2))
        fig, axs = plt.subplots(len(titles), len(imgs))

        fig.suptitle("MODEL-NAME" if self.DEBUG else self.model.name)

        print("[SanityCheck] Result Shape :", result[0][0].shape)

        for j in range(len(imgs)):
            for i in range(len(titles)):
                axs[i, j].imshow(result[j][i], cmap='jet')
                axs[i, j].set_title(titles[i])
                axs[i, j].axis('off')

        fig.savefig("%s/%d_%d.png" % (self.output, self.epoch, self.id_batch), dpi=1000)
        plt.close()


if __name__ == "__main__":
    print("\n> Récupération des fichiers")
    train_input_img_paths, train_target_img_paths = getImagesAndMasksPath("images/100k/train/", "labels/drivable/colormaps/train/")

    # Génération des datasets
    print("\n> Génération des datasets")
    train_gen = DrivableAreaDataset(5, (256, 256), train_input_img_paths[: 150], train_target_img_paths[: 150])

    sc = SanityCheck(train_gen.__getitem__(8), output="A-UNET/sanity-check/")
    sc.DEBUG = True

    sc.process_test()
