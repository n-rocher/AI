import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K
from utils import Mean_IOU_tensorflow_2

def UNET_BEG_CONVLSTM(nbr_images, input_size, num_classes):

    # Input
    inputs = layers.Input(shape=(nbr_images,) + input_size + (3,))


    # Contraction path
    c1 = layers.ConvLSTM2D(16, (3, 3), activation='relu', return_sequences=True, kernel_initializer='he_normal', padding='same')(inputs)
    c1 = layers.Dropout(0.1)(c1)
    c1, state_c1, _ = layers.ConvLSTM2D(16, (3, 3), activation='relu', return_sequences=True, return_state=True, kernel_initializer='he_normal', padding='same')(c1)
    p1 = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(c1)

    c2 = layers.ConvLSTM2D(32, (3, 3), activation='relu', return_sequences=True, kernel_initializer='he_normal', padding='same')(p1)
    c2 = layers.Dropout(0.1)(c2)
    c2, state_c2, _ = layers.ConvLSTM2D(32, (3, 3), activation='relu', return_sequences=True, return_state=True, kernel_initializer='he_normal', padding='same')(c2)
    p2 = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(c2)

    c3 = layers.ConvLSTM2D(64, (3, 3), activation='relu', return_sequences=True, kernel_initializer='he_normal', padding='same')(p2)
    c3 = layers.Dropout(0.2)(c3)
    c3, state_c3, _ = layers.ConvLSTM2D(64, (3, 3), activation='relu', return_sequences=True, return_state=True, kernel_initializer='he_normal', padding='same')(c3)
    p3 = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(c3)

    c4 = layers.ConvLSTM2D(128, (3, 3), activation='relu', return_sequences=True, kernel_initializer='he_normal', padding='same')(p3)
    c4 = layers.Dropout(0.2)(c4)
    c4, state_c4, _ = layers.ConvLSTM2D(128, (3, 3), activation='relu', return_sequences=True, return_state=True, kernel_initializer='he_normal', padding='same')(c4)
    p4 = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(c4)


    # Middle
    c5 = layers.ConvLSTM2D(200, (3, 3), activation='relu', return_sequences=True, kernel_initializer='he_normal', padding='same')(p4)
    c5 = layers.Dropout(0.3)(c5)
    c5 = layers.ConvLSTM2D(200, (3, 3), activation='relu', return_sequences=False, kernel_initializer='he_normal', padding='same')(c5)


    # Expansive path
    u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, state_c4])
    c6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = layers.Dropout(0.2)(c6)
    c6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, state_c3])
    c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = layers.Dropout(0.2)(c7)
    c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, state_c2])
    c8 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = layers.Dropout(0.1)(c8)
    c8 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, state_c1], axis=3)
    c9 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = layers.Dropout(0.1)(c9)
    c9 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)


    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(c9)

    model = keras.Model(inputs=[inputs], outputs=[outputs], name="UNET-BEG-CONVLSTM")
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy', Mean_IOU_tensorflow_2])
    model.summary()

    return model


def UNET_MID_CONVLSTM(nbr_images, input_size, num_classes):

    inputs = layers.Input(shape=(nbr_images,) + input_size + (3,))

    # Contraction path
    c1 = layers.TimeDistributed(layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))(inputs)

    c1 = layers.TimeDistributed(layers.Dropout(0.1))(c1)
    c1 = layers.TimeDistributed(layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))(c1)
    last_c1 = c1[:, 4]
    p1 = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(c1)
    p1 = layers.BatchNormalization()(p1)
    
    c2 = layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))(p1)
    c2 = layers.TimeDistributed(layers.Dropout(0.1))(c2)
    c2 = layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))(c2)
    last_c2 = c2[:, 4]
    p2 = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(c2)
    p2 = layers.BatchNormalization()(p2)
     
    c3 = layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))(p2)
    c3 = layers.TimeDistributed(layers.Dropout(0.2))(c3)
    c3 = layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))(c3)
    last_c3 = c3[:, 4]
    p3 = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(c3)
    p3 = layers.BatchNormalization()(p3)
     
    c4 = layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))(p3)
    c4 = layers.TimeDistributed(layers.Dropout(0.2))(c4)
    c4 = layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))(c4)
    last_c4 = c4[:, 4]
    p4 = layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2)))(c4)
    p4 = layers.BatchNormalization()(p4)
     
    # Middle
    # OLD MIDDLE c5 = layers.ConvLSTM2D(200, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same', dropout=0.3)(p4)
  
    # NEW MIDDLE
    c5 = layers.ConvLSTM2D(200, (3, 3), activation='relu', return_sequences=True, kernel_initializer='he_normal', padding='same')(p4)
    c5 = layers.TimeDistributed(layers.Dropout(0.3))(c5)
    c5 = layers.ConvLSTM2D(200, (3, 3), activation='relu', return_sequences=False, kernel_initializer='he_normal', padding='same')(c5)

    # Expansive path 
    u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, last_c4])
    c6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = layers.Dropout(0.2)(c6)
    c6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
     
    u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, last_c3])
    c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = layers.Dropout(0.2)(c7)
    c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
     
    u8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, last_c2])
    c8 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = layers.Dropout(0.1)(c8)
    c8 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
     
    u9 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, last_c1], axis=3)
    c9 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = layers.Dropout(0.1)(c9)
    c9 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
     
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(c9)
     
    model = keras.Model(inputs=[inputs], outputs=[outputs], name="UNET-MID-CONVLSTM")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy', Mean_IOU_tensorflow_2])
    model.summary()

    return model