# 하이퍼파라미터 탐색을 수행하는
# keras tuner 객체 생성하는 코드


import os

from omegaconf import OmegaConf

import functools

import kerastuner as kt
from generator import train_generator, val_generator
from models import get_model
from utils import get_callbacks

import tensorflow as tf
import kerastuner as kt
from tensorflow.keras import applications
from tensorflow.keras.backend import clear_session
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, BatchNormalization, Activation, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam

from tensorflow.keras.models import Model, load_model

def build_model(hp, cfg):
    # Builds a convolutional model
    # layers
    model = Sequential()

    backbone = getattr(applications, cfg.model_name)(
        weights='imagenet',
        include_top=False,
        input_shape=(cfg.img_size, cfg.img_size, 3),
    )

    model.add(backbone)

    if hp.Choice("pooling", ['max', 'avg']) == 'max':
        model.add(MaxPooling2D())
    else:
        model.add(AveragePooling2D())

    model.add(Flatten(name="flatten"))
    model.add(Dropout(rate=hp.Float("dropout_rate", 0.2, 0.8)))
    model.add(Dense(cfg.class_num, activation="softmax"))

    # optimizer
    lr = hp.Float('learning_rate', 1e-5, 1e-2, sampling="log")

    if hp.Choice('optimizer', ['adam', 'sgd']) == 'adam':
        optimizer = Adam(lr=lr)
    else:
        optimizer = SGD(lr=lr)

    # compile
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# hpo 후 config 파일 overwrite 하는 함수
def _tune(cfg, tg, vg):
    """
    Arguments:
        cfg: main configuration file (hydra config object)
        tg: train generator (keras image data generator)
        vg: val generator (keras image data generator)

    Returns:
        None
    """

    #wrap build_model function
    wrapped_build_func = functools.partial(build_model, cfg=cfg)

    #path
    save_path = os.path.join(cfg.root, f'hpo_results/{cfg.model_name}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # define tuner
    tuner = kt.Hyperband(
        hypermodel=wrapped_build_func,
        objective='val_accuracy',
        max_epochs=cfg.tune.max_epochs,
        factor=2,
        hyperband_iterations=cfg.tune.hyperband_iterations,
        distribution_strategy=tf.distribute.MirroredStrategy(),
        # directory=save_path
        directory=os.path.normpath('C:/')
    )

    # hpo
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        tuner.search(tg,
                     validation_data=vg,
                     callbacks=[tf.keras.callbacks.EarlyStopping('val_accuracy')]
                     )

    best_hps = tuner.get_best_hyperparameters()[0]

    # get best hps
    pooling = best_hps.get("pooling")
    dropout_rate = best_hps.get("dropout_rate")
    learning_rate = best_hps.get("learning_rate")
    optimizer = best_hps.get("optimizer")

    # overwrite config file
    config_path = os.path.join(cfg.root, f"config/backbone/{cfg.model_name}.yaml")
    config = OmegaConf.load(config_path)
    config.hp.pooling = pooling
    config.hp.dropout = dropout_rate
    config.hp.learning_rate = learning_rate
    config.hp.optimizer = optimizer
    OmegaConf.save(config, config_path)

    print(f"Config for {cfg.model_name} overwritten")


def _train(cfg, tg, vg):
    # training 실행 함수
    callbacks = get_callbacks(cfg)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = get_model(cfg)
        model.fit(tg, validation_data=vg, callbacks=callbacks, epochs=cfg.train.epochs)


# def _continue_train(cfg):
#     # 이어서 학습하는 함수
#
#     tg = train_generator(cfg)
#     vg = val_generator(cfg)
#
#     return None