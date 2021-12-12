import os
import cv2
import random
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img

def getImagesAndMasksPath(images_path, masks_path):

    list_masks = os.listdir(masks_path)

    list_ids = [os.path.splitext(mask_name)[0] for mask_name in list_masks]

    input_train_img_paths = [os.path.join(images_path, id + ".jpg") for id in list_ids]
    target_train_img_paths = [os.path.join(masks_path, id + ".png") for id in list_ids]

    return input_train_img_paths, target_train_img_paths

class DrivableAreaDataset(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    set_of_colors = set()

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size

        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]

        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = np.array(img) / 255.

        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            img = np.expand_dims(img, 2)

            # Conversion des couleurs des colormaps (0=0=noir, 130=1=rouge, 174=2=bleu)
            # For LANE ONLY
            img[img == 130] = 1.  # 2
            img[img == 174] = 2.  # 2

            y[j] = img

        return x, y


class MapillaryVistasDataset(keras.utils.Sequence):

    def __init__(self, batch_size, img_size, dataset_type):

        dataset_type = "training" if dataset_type == "train" else ("validation" if dataset_type == "val" else dataset_type)

        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths, self.target_img_paths = getImagesAndMasksPath("F:\\Vistas\\" + dataset_type + "\\images\\", "F:\\Vistas\\" + dataset_type + "\\v1.2\labels\\")

        self.CATEGORIES = {
            1: {"name": "Road", "color": [[128, 64, 128]]},
            2: {"name": "Lane", "color": [[255, 255, 255]]},
            3: {"name": "Crosswalk", "color": [[140, 140, 200], [200, 128, 128]]},
            4: {"name": "Curb", "color": [[196, 196, 196]]},
            5: {"name": "Sidewalk", "color": [[244, 35, 232]]},

            6: {"name": "Traffic Light", "color": [[250, 170, 30]]},
            7: {"name": "Traffic Sign", "color": [[220, 220, 0]]},

            8: {"name": "Person", "color": [[220, 20, 60]]},
            9: {"name": "Bicyclist", "color": [[255, 0, 0]]},
            10: {"name": "Motorcyclist", "color": [[255, 0, 100]]},

            11: {"name": "Bicycle", "color": [[119, 11, 32]]},
            12: {"name": "Bus", "color": [[0, 60, 100]]},
            13: {"name": "Car", "color": [[0, 0, 142]]},
            14: {"name": "Motorcycle", "color": [[0, 0, 230]]},
            15: {"name": "Truck", "color": [[0, 0, 70]]}
        }

    def classes(self):
        return len(self.CATEGORIES) + 1

    def name(self):
        return "MapillaryVistasDataset"

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):

        i = idx * self.batch_size

        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]

        # Chargement de la photo de la route
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            frame = np.array(load_img(path, target_size=self.img_size))
            x[j] = frame / 255.

        # Chargement du masque et traitement
        ins_255 = np.ones(self.img_size) * 255
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            mask = np.array(load_img(path, target_size=self.img_size, color_mode="rgb"))  # On charge le masque

            # Create blank image
            instance = np.zeros(self.img_size)

            # For every categories in the list
            for id_category in self.CATEGORIES:
                data_category = self.CATEGORIES[id_category]

                # We select pixels belonging to that category
                test = cv2.inRange(mask, tuple(data_category["color"][0]), tuple(data_category["color"][0]))

                # We copy 255 value for a white image
                res = cv2.bitwise_and(ins_255, ins_255, mask=test)

                # And we past it to the good id to the instance
                instance = instance + (res / 255 * id_category)

            instance = np.expand_dims(instance, 2)

            y[j] = instance

        return x, y


class A2D2Dataset(keras.utils.Sequence):

    def __init__(self, batch_size, img_size):

        self.dataset_folder = r"F:\\A2D2 Camera Semantic\\"
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths, self.target_img_paths = self.getData()

        self.CATEGORIES = {
            1: {"name": "Curb", "color": [[128, 128, 0]]},
            2: {"name": "Crosswalk", "color": [[210, 50, 115]]},
            3: {"name": "Road", "color": [[180, 50, 180], [255, 0, 255]]},
            4: {"name": "Sidewalk", "color": [[180, 150, 200]]},
            5: {"name": "Person", "color": [[204, 153, 255], [189, 73, 155], [239, 89, 191]]},
            6: {"name": "Bicyclist", "color": []},
            7: {"name": "Motorcyclist", "color": []},
            8: {"name": "Lane", "color": [[255, 193, 37], [200, 125, 210], [128, 0, 255]]},
            9: {"name": "Traffic Light", "color": [[0, 128, 255], [30, 28, 158], [60, 28, 100]]},
            10: {"name": "Traffic Sign", "color": [[0, 255, 255], [30, 220, 220], [60, 157, 199]]},
            11: {"name": "Bicycle", "color": [[182, 89, 6], [150, 50, 4], [90, 30, 1], [90, 30, 30]]},
            12: {"name": "Bus", "color": []},
            13: {"name": "Car", "color": [[255, 0, 0], [200, 0, 0], [150, 0, 0], [128, 0, 0]]},
            14: {"name": "Motorcycle", "color": []},
            15: {"name": "Truck", "color": [[255, 128, 0], [200, 128, 0], [150, 128, 0]]}
        }

    def classes(self):
        return len(self.CATEGORIES) + 1

    def name(self):
        return "A2D2Dataset"

    def getData(self):
        '''
        Permet de trouver le nom des fichiers du jeux de donnée A2D2
        '''
        data_image = []
        data_label = []

        camera_day_folders = [os.path.join(self.dataset_folder, item) for item in os.listdir(self.dataset_folder) if os.path.isdir(self.dataset_folder + item)]
        for folder in camera_day_folders:
            camera_files_folder = os.path.join(folder, "camera", "cam_front_center")
            label_files_folder = os.path.join(folder, "label", "cam_front_center")

            camera_files_files = [os.path.join(camera_files_folder, file) for file in os.listdir(camera_files_folder)]
            label_files_files = [os.path.join(label_files_folder, file) for file in os.listdir(label_files_folder)]

            data_image = data_image + camera_files_files
            data_label = data_label + label_files_files

        return data_image, data_label

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):

        i = idx * self.batch_size

        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]

        # Chargement de la photo de la route
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            x[j] = np.array(load_img(path, target_size=self.img_size)) / 255.

        # Chargement du masque et traitement
        ins_255 = np.ones(self.img_size) * 255
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            mask = np.array(load_img(path, target_size=self.img_size, color_mode="rgb"))

            # Create blank image
            instance = np.zeros(self.img_size)

            # For every categories in the list
            for id_category in self.CATEGORIES:
                data_category = self.CATEGORIES[id_category]

                for color in data_category["color"]:
                    color = tuple(color)

                    # We select pixels belonging to that category
                    test = cv2.inRange(mask, color, color)

                    # We copy 255 value for a white image
                    res = cv2.bitwise_and(ins_255, ins_255, mask=test)

                    # And we past it to the good id to the instance
                    instance = instance + (res / 255 * id_category)

            instance = np.expand_dims(instance, 2)

            y[j] = instance

        return x, y


class MultiDataset(keras.utils.Sequence):

    def __init__(self, batch_size, img_size, dataset_type):

        dataset_type = "training" if dataset_type == "train" else ("validation" if dataset_type == "val" else dataset_type)

        self.batch_size = batch_size
        self.img_size = img_size

        # Chargement des données
        self.vistas_dataset = self.getVistasData(dataset_type)
        self.a2d2_dataset = self.getA2D2Data(dataset_type)
        self.dataset = self.vistas_dataset + self.a2d2_dataset
        random.shuffle(self.dataset)

        self.CATEGORIES = {
            "VISTAS": {
                1: {"name": "Road", "color": [[128, 64, 128]]},
                2: {"name": "Lane", "color": [[255, 255, 255]]},
                3: {"name": "Crosswalk", "color": [[140, 140, 200], [200, 128, 128]]},
                4: {"name": "Curb", "color": [[196, 196, 196]]},
                5: {"name": "Sidewalk", "color": [[244, 35, 232]]},

                6: {"name": "Traffic Light", "color": [[250, 170, 30]]},
                7: {"name": "Traffic Sign", "color": [[220, 220, 0]]},

                8: {"name": "Person", "color": [[220, 20, 60]]},
                9: {"name": "Bicyclist", "color": [[255, 0, 0]]},
                10: {"name": "Motorcyclist", "color": [[255, 0, 100]]},

                11: {"name": "Bicycle", "color": [[119, 11, 32]]},
                12: {"name": "Bus", "color": [[0, 60, 100]]},
                13: {"name": "Car", "color": [[0, 0, 142]]},
                14: {"name": "Motorcycle", "color": [[0, 0, 230]]},
                15: {"name": "Truck", "color": [[0, 0, 70]]}
            },
            "A2D2": {
                1: {"name": "Road", "color": [[180, 50, 180], [255, 0, 255]]},
                2: {"name": "Lane", "color": [[255, 193, 37], [200, 125, 210], [128, 0, 255]]},
                3: {"name": "Crosswalk", "color": [[210, 50, 115]]},
                4: {"name": "Curb", "color": [[128, 128, 0]]},
                5: {"name": "Sidewalk", "color": [[180, 150, 200]]},

                6: {"name": "Traffic Light", "color": [[0, 128, 255], [30, 28, 158], [60, 28, 100]]},
                7: {"name": "Traffic Sign", "color": [[0, 255, 255], [30, 220, 220], [60, 157, 199]]},

                8: {"name": "Person", "color": [[204, 153, 255], [189, 73, 155], [239, 89, 191]]},
                9: {"name": "Bicyclist", "color": []},
                10: {"name": "Motorcyclist", "color": []},

                11: {"name": "Bicycle", "color": [[182, 89, 6], [150, 50, 4], [90, 30, 1], [90, 30, 30]]},
                12: {"name": "Bus", "color": []},
                13: {"name": "Car", "color": [[255, 0, 0], [200, 0, 0], [150, 0, 0], [128, 0, 0]]},
                14: {"name": "Motorcycle", "color": []},
                15: {"name": "Truck", "color": [[255, 128, 0], [200, 128, 0], [150, 128, 0]]}
            }
        }

    def classes(self):
        return len(self.CATEGORIES[list(self.CATEGORIES.keys())[0]]) + 1

    def name(self):
        return "MultiDataset"

    def getVistasData(self, dataset_type):
        data_image, data_label = getImagesAndMasksPath("F:\\Mapillary Vistas\\" + dataset_type + "\\images\\", "F:\\Mapillary Vistas\\" + dataset_type + "\\v1.2\labels\\")
        return list(zip(["VISTAS"] * len(data_image), data_image, data_label))

    def getA2D2Data(self, dataset_type):
        dataset_folder = r"F:\\A2D2 Camera Semantic\\" + dataset_type + "\\"

        data_image = []
        data_label = []

        camera_day_folders = [os.path.join(dataset_folder, item) for item in os.listdir(dataset_folder) if os.path.isdir(dataset_folder + item)]
        for folder in camera_day_folders:
            camera_files_folder = os.path.join(folder, "camera", "cam_front_center")
            label_files_folder = os.path.join(folder, "label", "cam_front_center")

            camera_files_files = [os.path.join(camera_files_folder, file) for file in os.listdir(camera_files_folder)]
            label_files_files = [os.path.join(label_files_folder, file) for file in os.listdir(label_files_folder)]

            data_image = data_image + camera_files_files
            data_label = data_label + label_files_files

        return list(zip(["A2D2"] * len(data_image), data_image, data_label))

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __getitem__(self, idx):

        i = idx * self.batch_size

        batch_input_img_paths = self.dataset[i: i + self.batch_size]

        # Initialisation des variables de résultat
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")

        # Temp
        ins_255 = np.ones(self.img_size) * 255

        for i, (dataset, image_path, mask_path) in enumerate(batch_input_img_paths):

            ####################
            # CHARGEMENT IMAGE #
            ####################
            x[i] = np.array(load_img(image_path, target_size=self.img_size)) / 255.
            ####################

            ###################
            # CHARGEMENT MASK #
            ###################
            mask = np.array(load_img(mask_path, target_size=self.img_size, color_mode="rgb"))

            # Create blank image
            instance = np.zeros(self.img_size)

            # For every categories in the list
            for id_category in self.CATEGORIES[dataset]:
                data_category = self.CATEGORIES[dataset][id_category]

                for color in data_category["color"]:
                    color = tuple(color)

                    # We select pixels belonging to that category
                    test = cv2.inRange(mask, color, color)

                    # We copy 255 value for a white image
                    res = cv2.bitwise_and(ins_255, ins_255, mask=test)

                    # And we past it to the good id to the instance
                    instance = instance + (res / 255 * id_category)

            instance = np.expand_dims(instance, 2)

            y[i] = instance

            ####################
        return x, y


class SequenceDataset(keras.utils.Sequence):

    def __init__(self, clip_size, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.clip_size = clip_size

        self.input_img_paths, self.target_img_paths = self.getImagesAndMasksPath(input_img_paths, target_img_paths)

        self.CATEGORIES = {
            1: {"name": "Main Road", "color": 130},
            2: {"name": "Secondary Road", "color": 174}
        }

    def getImagesAndMasksPath(self, images_path, masks_path):

        list_images = os.listdir(images_path) #[:20] FIX-ME: A supprimer
        list_masks = [os.path.splitext(mask_name)[0] for mask_name in os.listdir(masks_path)]

        using_id = [mask_id for mask_id in list_masks if mask_id in list_images]
        
        input_train_img_paths = [[os.path.join(images_path, mid, file) for file in os.listdir(os.path.join(images_path, mid))] for mid in using_id]
     
        target_train_img_paths = [os.path.join(masks_path, mid + ".png") for mid in using_id]

        return input_train_img_paths, target_train_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def classes(self):
        return 2 + 1

    def name(self):
        return "SequenceDataset"

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size

        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]

        x = np.zeros((self.batch_size, self.clip_size,) + self.img_size + (3,), dtype="float32")
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")

        # Temp
        ins_255 = np.ones(self.img_size) * 255

        for j, list_clip_image_path in enumerate(batch_input_img_paths):
            for i, clip_image_path in enumerate(list_clip_image_path):
                x[j, i] = load_img(clip_image_path, target_size=self.img_size, color_mode="rgb")


        for j, mask_path in enumerate(batch_target_img_paths):

            ###################
            # CHARGEMENT MASK #
            ###################
            mask = np.array(load_img(mask_path, target_size=self.img_size, color_mode="grayscale"))

            # Create blank image
            instance = np.zeros(self.img_size)

            # For every categories in the list
            for id_category in self.CATEGORIES:
                data_category = self.CATEGORIES[id_category]

                color = data_category["color"]

                # We select pixels belonging to that category
                test = cv2.inRange(mask, color-1, color+1)

                # # We copy 255 value for a white image
                res = cv2.bitwise_and(ins_255, ins_255, mask=test)

                # # And we past it to the good id to the instance
                instance = instance + (res / 255 * id_category)

            instance = np.expand_dims(instance, 2)

            y[j] = instance

        return x, y


if __name__ == "__main__":

    print("\n> Génération du dataset")
    train_gen = MultiDataset(5, (480, 720), "train")

    import matplotlib.pyplot as plt

    for id in range(len(train_gen)):
        data = train_gen.__getitem__(id)

        for img_i, mask_i in zip(data[0], data[1]):
            fig, axs = plt.subplots(2, 1)

            axs[0].imshow((img_i*255).astype(np.uint8))
            axs[1].imshow(mask_i)
            fig.show()
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            plt.show()
