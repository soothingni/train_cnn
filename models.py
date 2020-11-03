# 모델 생성하는 get_model 함수가 정의된 코드

#DL
import tensorflow as tf
import kerastuner as kt
from tensorflow.keras import applications
from tensorflow.keras.backend import clear_session
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, BatchNormalization, Activation, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard, CSVLogger
from tensorflow.keras.models import Model, load_model

def get_model(cfg):
    pooling = cfg.hp.pooling
    dropout = float(cfg.hp.dropout)
    learning_rate = float(cfg.hp.learning_rate)
    optim = cfg.hp.optimizer

    model = Sequential()

    backbone = getattr(applications, cfg.model_name)(
        weights='imagenet',
        include_top=False,
        input_shape=(cfg.img_size, cfg.img_size, 3),
    )

    model.add(backbone)

    #pooling layer
    if pooling == "avg": model.add(AveragePooling2D(name="avg_pooling"))
    else: model.add(MaxPooling2D(name="max_pooling"))

    #flatten layer
    model.add(Flatten(name="flatten"))
    model.add(Dropout(rate=dropout))
    model.add(Dense(cfg.class_num, activation="softmax"))

    #optimizer
    optimizer = Adam(learning_rate) if optim == "adam" else SGD(learning_rate)

    #compile
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model