import cv2
import os
import numpy as np

from tensorflow import keras, argmax

from model import ArgmaxMeanIOU

import sys
import time

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QFileDialog, QComboBox, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget)


CATEGORIES = {
    1: {"name": "Road", "color": [128, 64, 128]},
    2: {"name": "Lane", "color": [255, 255, 255]},
    3: {"name": "Crosswalk", "color": [200, 128, 128]},
    4: {"name": "Curb", "color": [196, 196, 196]},
    5: {"name": "Sidewalk", "color": [244, 35, 232]},

    6: {"name": "Traffic Light", "color": [250, 170, 30]},
    7: {"name": "Traffic Sign", "color": [220, 220, 0]},

    8: {"name": "Person", "color": [220, 20, 60]},
    9: {"name": "Bicyclist", "color": [255, 0, 0]},
    10: {"name": "Motorcyclist", "color": [255, 0, 100]},

    11: {"name": "Bicycle", "color": [119, 11, 32]},
    12: {"name": "Bus", "color": [0, 60, 100]},
    13: {"name": "Car", "color": [0, 0, 142]},
    14: {"name": "Motorcycle", "color": [0, 0, 230]},
    15: {"name": "Truck", "color": [0, 0, 70]}
}

TRAFFIC_SIGN_DATASET = {
    1: "Virage à droite", 
    100: "Sens unique (droit)", 
    107: "Zone 30", 
    108: "Fin zone 30", 
    109: "Passage pour piétons",
    11: "Ralentisseur simple",
    12: "Ralentisseur double",
    125: "Ralentisseur",
    13: "Route glissante",
    140: "Direction",
    15: "Chute de pierres",
    16: "Passage pour piétons",
    17: "Enfants (école)",
    2: "Virage à gauche",
    23: "Intersection",
    24: "Intersection avec une route",
    25: "Rond-point",
    3: "Double virage (gauche)",
    32: "Autres dangers",
    35: "Céder le passage",
    36: "Stop",
    37: "Route prioritaire",
    38: "Fin route prioritaire",
    39: "Priorité au trafic en sens inverse",
    4: "Double virage (droite)",
    40: "Priorité au trafic en sens inverse",
    41: "Sens interdit",
    51: "Virage à gauche interdit",
    52: "Virage à droite interdit",
    53: "Demi-tour interdit",
    54: "Dépassement interdit",
    55: "Dépassement interdit aux véhicules de transport de marchandises",
    57: "Vitesse maximale 20",
    59: "Vitesse maximale 30",
    60: "Vitesse maximale 40",
    61: "Vitesse maximale 50",
    62: "Vitesse maximale 60",
    63: "Vitesse maximale 70",
    64: "Vitesse maximale 80",
    65: "Vitesse maximale 90",
    66: "Vitesse maximale 100",
    67: "Vitesse maximale 110",
    68: "Vitesse maximale 120",
    7: "Rétrécissement de la chaussée",
    80: "Direction - Droit",
    81: "Direction - Droite",
    82: "Direction - Gauche",
    83: "Direction - Droite ou Droite",
    84: "Direction - Tout droit ou à gauche",
    85: "Direction - Tourner à droite",
    86: "Direction - Tourner à gauche",
    87: "Passer à droite",
    88: "Passer à gauche"}

TRAFFIC_SIGN_DATASET_VALUES = list(TRAFFIC_SIGN_DATASET.values())
TRAFFIC_SIGN_DATASET_KEYS = list(TRAFFIC_SIGN_DATASET.keys())

IMG_SIZE = (720, 480)
VIDEO_PATH = r"F:\Road Video"

BOUNDING_BOX_PADDING = 5

class Thread(QThread):
    EVT_ROAD_IMAGE = Signal(QImage)
    EVT_SEGMENTATION_IMAGE = Signal(QImage)

    segmentation_model = None
    segmentation_model_size = None

    traffic_sign_recognition_model = None
    traffic_sign_recognition_model_size = None

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.video_file = None
        self.status = True
        self.cap = None
        self.isAvailable = True

    def start_file(self, fname):
        self.video_file = os.path.join(VIDEO_PATH, fname)

        while self.isAvailable == False:
            pass

        self.start()

    def sendTo(self, evt, frame):
        # Creating and scaling QImage
        h, w, ch = frame.shape
        img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        scaled_img = img.scaled(640, 480, Qt.KeepAspectRatio)

        # Emit signal
        evt.emit(scaled_img)

    def loadSegmentationModel(self, filename):
        self.segmentation_model = None
        self.segmentation_model_size = None

        try:
            self.segmentation_model = keras.models.load_model(filename, custom_objects={'ArgmaxMeanIOU': ArgmaxMeanIOU})
            self.segmentation_model_size = self.segmentation_model.get_layer(index=0).input_shape[0][1:-1][::-1]
        except:
            print("[loadSegmentationModel] Une erreur est survenue lors de l'ouverture du h5")

    def loadTrafficSignRecognitionModel(self, filename):

        self.traffic_sign_recognition_model = None
        self.traffic_sign_recognition_model_size = None

        try:
            self.traffic_sign_recognition_model = keras.models.load_model(filename)
            self.traffic_sign_recognition_model_size = self.traffic_sign_recognition_model.get_layer(index=0).input_shape[0][1:-1][::-1]
        except Exception as e:
            print("[loadTrafficSignRecognitionModel] Une erreur est survenue lors de l'ouverture du h5")
            print(e)

    def run(self):

        global CATEGORIES

        if self.video_file is not None:

            self.ThreadActive = True

            if self.cap is not None:
                self.cap.release()

            self.cap = cv2.VideoCapture(self.video_file)

            while(self.ThreadActive and self.cap.isOpened()):
                self.isAvailable = False


                ret, frame = self.cap.read()
                if not ret:
                    continue

                img_resized = cv2.resize(frame, self.segmentation_model_size, interpolation=cv2.INTER_AREA)

                result_segmentation = self.segmentation_model.predict(np.expand_dims(img_resized / 255., axis=0))[0]

                # Argmax
                result_segmentation = argmax(result_segmentation, axis=-1)
                # kernel = np.ones((3, 3), np.uint8)
                # result = cv2.erode(np.array(result, dtype=np.uint8), kernel, iterations=3)
                segmentation = np.zeros(result_segmentation.shape + (3,), dtype=np.uint8)
                for categorie in CATEGORIES.keys():

                    # En cas de détection de "Traffic Sign", on dessine une box autour
                    if categorie == 7:
                        print("\n")
                        contours, _ = cv2.findContours(np.array(result_segmentation == categorie, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        for cnt in contours:
                            x, y, w, h = cv2.boundingRect(cnt)
                            if w > 10 and h > 10 and w * h > 200:
                                x = x - BOUNDING_BOX_PADDING
                                y = y - BOUNDING_BOX_PADDING
                                w = w + BOUNDING_BOX_PADDING * 2
                                h = h + BOUNDING_BOX_PADDING * 2

                                x = x if x >= 0 else 0
                                y = y if y >= 0 else 0
                                w = w if w <= self.segmentation_model_size[0] else self.segmentation_model_size[0]
                                h = h if h <= self.segmentation_model_size[1] else self.segmentation_model_size[1]

                                cv2.rectangle(segmentation, (x, y), (x + w, y + h), (0, 255, 0), 1)

                                if self.traffic_sign_recognition_model is not None:
                                    test_sign = cv2.resize(img_resized[y:y + h, x:x + w], self.traffic_sign_recognition_model_size, interpolation=cv2.INTER_AREA)
                                    test_sign = np.array([test_sign / 255.])
                                    result_traffic = self.traffic_sign_recognition_model.predict(test_sign)[0]
                                    max_index_col = np.argmax(result_traffic, axis=0)
                                    proba = result_traffic[max_index_col]
                                    if proba > 0.75:
                                        print("[INFO] Traffic Sign Recognition : '" + TRAFFIC_SIGN_DATASET_VALUES[max_index_col] + " (" + str(TRAFFIC_SIGN_DATASET_KEYS[max_index_col]) + ") ' P=" + str(proba))

                    segmentation[result_segmentation == categorie] = CATEGORIES[categorie]["color"]

                if self.segmentation_model_size != (640, 480):
                    img_resized = cv2.resize(img_resized, (640, 480), interpolation=cv2.INTER_AREA)
                    segmentation = cv2.resize(segmentation, (640, 480), interpolation=cv2.INTER_AREA)

                self.sendTo(self.EVT_ROAD_IMAGE, cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
                self.sendTo(self.EVT_SEGMENTATION_IMAGE, segmentation)

            self.cap.release()
            self.isAvailable = True

    def stop(self):
        self.ThreadActive = False

class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        # Titre
        self.setWindowTitle("Road Segmentation")

        # Thread in charge of updating the image
        self.thread = Thread(self)
        self.thread.EVT_ROAD_IMAGE.connect(self.setRoadImage)
        self.thread.EVT_SEGMENTATION_IMAGE.connect(self.setSegmentationImage)

        # MODEL CHOOSER LAYOUT
        self.model_chooser_layout = QGroupBox("Model chooser")
        self.model_chooser_layout.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        # Segmentation UI
        segmentation_model_chooser_layout = QHBoxLayout()
        self.segmentation_model_chooser_input = QLineEdit()
        self.segmentation_model_chooser_button = QPushButton("...")

        segmentation_model_chooser_layout.addWidget(QLabel("Segmentation :"), 10)
        segmentation_model_chooser_layout.addWidget(self.segmentation_model_chooser_input, 50)
        segmentation_model_chooser_layout.addWidget(self.segmentation_model_chooser_button)

        # Traffic Sign Recognition UI
        traffic_sign_model_chooser_layout = QHBoxLayout()
        self.traffic_sign_model_chooser_input = QLineEdit()
        self.traffic_sign_model_chooser_button = QPushButton("...")

        traffic_sign_model_chooser_layout.addWidget(QLabel("Traffic Sign Recognition :"), 10)
        traffic_sign_model_chooser_layout.addWidget(self.traffic_sign_model_chooser_input, 50)
        traffic_sign_model_chooser_layout.addWidget(self.traffic_sign_model_chooser_button)

        # Model UI def
        model_chooser_layout = QHBoxLayout()
        model_chooser_layout.addLayout(segmentation_model_chooser_layout)
        model_chooser_layout.addLayout(traffic_sign_model_chooser_layout)
        self.model_chooser_layout.setLayout(model_chooser_layout)

        # IMAGE RESULT
        self.video_layout_model = QGroupBox("Result")
        self.video_layout_model.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        image_layout = QHBoxLayout()
        self.image_seg = QLabel(self)
        self.image_road = QLabel(self)
        self.image_seg.setFixedSize(640, 480)
        self.image_road.setFixedSize(640, 480)
        image_layout.addWidget(self.image_road)
        image_layout.addWidget(self.image_seg)
        self.video_layout_model.setLayout(image_layout)

        # VIDEO FILE CHOOSER
        self.group_model = QGroupBox("Video file")
        self.group_model.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        model_layout = QHBoxLayout()
        self.combobox = QComboBox()
        for video_filename in os.listdir(VIDEO_PATH):
            self.combobox.addItem(video_filename)

        model_layout.addWidget(QLabel("Video :"), 10)
        model_layout.addWidget(self.combobox, 75)
        self.group_model.setLayout(model_layout)

        # BUTTONS
        buttons_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.start_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.stop_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        buttons_layout.addWidget(self.stop_button)
        buttons_layout.addWidget(self.start_button)

        control_layout = QHBoxLayout()
        control_layout.addWidget(self.group_model, 1)
        control_layout.addLayout(buttons_layout, 1)

        # Main layout
        main_layout = QHBoxLayout()
        layout = QVBoxLayout()
        layout.addWidget(self.model_chooser_layout)
        layout.addLayout(control_layout)
        layout.addWidget(self.video_layout_model)

        main_layout.addLayout(layout)

        # LABELS
        label_scroll = QScrollArea()
        label_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        label_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        grid = QGridLayout()

        for id_label in CATEGORIES:

            label = CATEGORIES[id_label]

            color = "rgb(" + str(label["color"][0]) + "," + str(label["color"][1]) + "," + str(label["color"][2]) + ")"

            testWidget = QFrame()
            testWidget.setFixedSize(50, 25)
            testWidget.setObjectName("myWidget")
            testWidget.setStyleSheet("#myWidget {background-color:" + color + ";}")

            grid.addWidget(testWidget, id_label, 0)
            grid.addWidget(QLabel(label["name"]), id_label, 1)

        label_widget = QWidget()
        label_widget.setLayout(grid)
        label_scroll.setWidget(label_widget)
        label_scroll.setFixedWidth(200)

        main_layout.addWidget(label_scroll)

        # Central widget
        widget = QWidget(self)
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

        # Connections
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setEnabled(False)
        self.combobox.currentTextChanged.connect(self.set_video)
        self.segmentation_model_chooser_input.returnPressed.connect(self.segmentation_loadModel_Input)
        self.segmentation_model_chooser_button.clicked.connect(self.segmentation_loadModel_Button)
        self.traffic_sign_model_chooser_input.returnPressed.connect(self.traffic_sign_loadModel_Input)
        self.traffic_sign_model_chooser_button.clicked.connect(self.traffic_sign_loadModel_Button)

    @Slot()
    def set_video(self, filename):
        cv2.destroyAllWindows()
        self.thread.stop()
        self.thread.start_file(filename)
        self.stop_button.setEnabled(True)
        self.start_button.setEnabled(False)

    @Slot()
    def start(self):
        self.stop_button.setEnabled(True)
        self.start_button.setEnabled(False)
        self.thread.start_file(self.combobox.currentText())

    @Slot()
    def stop(self):
        cv2.destroyAllWindows()
        self.thread.stop()
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(True)

    @Slot(QImage)
    def setRoadImage(self, image):
        self.image_road.setPixmap(QPixmap.fromImage(image))

    @Slot(QImage)
    def setSegmentationImage(self, image):
        self.image_seg.setPixmap(QPixmap.fromImage(image))

    def segmentation_loadModel_Input(self):
        fileName = self.segmentation_model_chooser_input.text()
        self.thread.loadSegmentationModel(fileName)

    def segmentation_loadModel_Button(self):
        fileName = QFileDialog.getOpenFileName(self, "Load model savepoint", "", "H5 file (*.h5)")
        self.segmentation_model_chooser_input.setText(fileName[0])
        self.thread.loadSegmentationModel(fileName[0])

    def traffic_sign_loadModel_Input(self):
        fileName = self.traffic_sign_model_chooser_input.text()
        self.thread.loadTrafficSignRecognitionModel(fileName)

    def traffic_sign_loadModel_Button(self):
        fileName = QFileDialog.getOpenFileName(self, "Load model savepoint", "", "H5 file (*.h5)")
        self.traffic_sign_model_chooser_input.setText(fileName[0])
        self.thread.loadTrafficSignRecognitionModel(fileName[0])


def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


if __name__ == "__main__":

    sys.excepthook = except_hook

    app = QApplication()
    w = Window()
    w.show()
    sys.exit(app.exec())


# Load model
# model = keras.models.load_model(MODEL_PATH, custom_objects={'ArgmaxMeanIOU': ArgmaxMeanIOU})

# MODEL_SIZE = model.get_layer(index=0).input_shape[0][1:-1][::-1]

# # Get video in the video folder
# videos = os.listdir(VIDEO_PATH)

# for video in videos:
#     cap = cv2.VideoCapture(os.path.join(VIDEO_PATH, video))

#     if (cap.isOpened() == False):
#         print("Error opening video stream or file")
#         break

#     while(cap.isOpened()):

#         ret, frame = cap.read()
#         if ret == True:

#             frame = cv2.resize(frame, IMG_SIZE, interpolation=cv2.INTER_AREA)
#             img_resized = cv2.resize(frame, MODEL_SIZE, interpolation=cv2.INTER_AREA)

#             cv2.imshow('Video :', img_resized)

#             result = model.predict(np.expand_dims(img_resized / 255., axis=0))[0]
#             num_categories_out = result.shape[2]

#             # On affiche des layers
#             if DISPLAY_LAYER is not False:
#                 for i in range(1, num_categories_out) if DISPLAY_LAYER == True else DISPLAY_LAYER:
#                     layer = np.array(result[:, :, i] * 255., dtype=np.uint8)
#                     cv2.imshow('Resultat ' + str(i) + '/' + str(num_categories_out) + " : " + CATEGORIES[i]["name"], cv2.applyColorMap(layer, cv2.COLORMAP_JET))

#             # Argmax
#             result = argmax(result, axis=-1)
#             kernel = np.ones((3, 3), np.uint8)
#             result = cv2.erode(np.array(result, dtype=np.uint8), kernel, iterations=1)
#             test_r = np.zeros(result.shape + (3,), dtype=np.uint8)
#             for categorie in CATEGORIES.keys():
#                 test_r[result == categorie] = CATEGORIES[categorie]["color"]

#             cv2.imshow('Segmentation :', cv2.resize(cv2.cvtColor(test_r, cv2.COLOR_RGB2BGR), IMG_SIZE, interpolation=cv2.INTER_AREA))

#             # Press Q on keyboard to  exit
#             if cv2.waitKey(15) & 0xFF == ord('q'):
#                 break

#         # Break the loop
#         else:
#             break

#     cap.release()
#     cv2.destroyAllWindows()
