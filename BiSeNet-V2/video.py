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

IMG_SIZE = (720, 480)
VIDEO_PATH = r"F:\Road Video"
MODEL_PATH = r"J:\PROJET\IA\BiSeNet-V2\models\20211203-105503\BiSeNet-V2_MultiDataset_480-704_epoch-12_loss-0.14_miou_0.47.h5"

BOUNDING_BOX_PADDING = 10

class Thread(QThread):
    EVT_ROAD_IMAGE = Signal(QImage)
    EVT_SEGMENTATION_IMAGE = Signal(QImage)

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.video_file = None
        self.status = True
        self.cap = True

    def set_file(self, fname):
        self.video_file = os.path.join(VIDEO_PATH, fname)

    def sendTo(self, evt, frame):
        # Creating and scaling QImage
        h, w, ch = frame.shape
        img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        scaled_img = img.scaled(640, 480, Qt.KeepAspectRatio)

        # Emit signal
        evt.emit(scaled_img)

    def run(self):

        global model
        global MODEL_SIZE
        global CATEGORIES

        if self.video_file is not None:

            self.cap = cv2.VideoCapture(self.video_file)

            while(self.cap.isOpened() and self.status):

                ret, frame = self.cap.read()
                if not ret:
                    continue

                img_resized = cv2.resize(frame, MODEL_SIZE, interpolation=cv2.INTER_AREA)

                result = model.predict(np.expand_dims(img_resized / 255., axis=0))[0]

                # Argmax
                result = argmax(result, axis=-1)
                # kernel = np.ones((3, 3), np.uint8)
                # result = cv2.erode(np.array(result, dtype=np.uint8), kernel, iterations=3)
                segmentation = np.zeros(result.shape + (3,), dtype=np.uint8)
                for categorie in CATEGORIES.keys():
                    
                    # En cas de détection de "Traffic Sign", on dessine une box autour
                    if categorie == 7:
                        contours, _ = cv2.findContours(np.array(result == categorie, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        for cnt in contours:
                            x, y, w, h = cv2.boundingRect(cnt)
                            if w * h > 200 : 
                                x = x - BOUNDING_BOX_PADDING
                                y = y - BOUNDING_BOX_PADDING
                                w = w + BOUNDING_BOX_PADDING * 2
                                h = h + BOUNDING_BOX_PADDING * 2
                                cv2.rectangle(segmentation, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    segmentation[result == categorie] = CATEGORIES[categorie]["color"]

                self.sendTo(self.EVT_ROAD_IMAGE, cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
                self.sendTo(self.EVT_SEGMENTATION_IMAGE, segmentation)

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        # Title and dimensions
        self.setWindowTitle("Road Segmentation")
        # self.setGeometry(0, 0, 800, 500)

        # Thread in charge of updating the image
        self.thread = Thread(self)
        self.thread.EVT_ROAD_IMAGE.connect(self.setRoadImage)
        self.thread.EVT_SEGMENTATION_IMAGE.connect(self.setSegmentationImage)

        # MODEL CHOOSER LAYOUT
        self.model_chooser_layout = QGroupBox("Model chooser")
        self.model_chooser_layout.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        model_chooser_layout = QHBoxLayout()

        self.model_chooser_dialog = QFileDialog()
        self.model_chooser_input = QLineEdit()
        self.model_chooser_button = QPushButton("...")

        model_chooser_layout.addWidget(QLabel("Model :"), 10)
        model_chooser_layout.addWidget(self.model_chooser_input, 50)
        model_chooser_layout.addWidget(self.model_chooser_button)
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
        self.model_chooser_input.returnPressed.connect(self.loadModel_Input)
        self.model_chooser_button.clicked.connect(self.loadModel_Button)

    @Slot()
    def set_video(self, filename):
        cv2.destroyAllWindows()
        self.thread.terminate()
        self.thread.wait()
        self.thread.set_file(filename)
        self.thread.start()
        self.stop_button.setEnabled(True)
        self.start_button.setEnabled(False)

    @Slot()
    def start(self):
        self.stop_button.setEnabled(True)
        self.start_button.setEnabled(False)
        self.thread.set_file(self.combobox.currentText())
        self.thread.start()

    @Slot()
    def stop(self):
        cv2.destroyAllWindows()
        self.thread.terminate()
        self.thread.wait()
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(True)

    @Slot(QImage)
    def setRoadImage(self, image):
        self.image_road.setPixmap(QPixmap.fromImage(image))

    @Slot(QImage)
    def setSegmentationImage(self, image):
        self.image_seg.setPixmap(QPixmap.fromImage(image))

    def loadModel_Input(self):
        fileName = self.model_chooser_input.text()
        self.loadModel(fileName)

    def loadModel_Button(self):
        fileName = QFileDialog.getOpenFileName(self, "Load model savepoint", "", "H5 file (*.h5)")
        self.loadModel(fileName[0])

    def loadModel(self, filename):
        global model
        global MODEL_SIZE

        try:
            model = keras.models.load_model(filename, custom_objects={'ArgmaxMeanIOU': ArgmaxMeanIOU})
            MODEL_SIZE = model.get_layer(index=0).input_shape[0][1:-1][::-1]
            self.model_chooser_input.setText(filename)
        except:
            print("Une erreur est survenue lors de l'ouverture du modele")

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
