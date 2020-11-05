# 하이퍼파라미터 탐색을 수행하는
# keras tuner 객체 생성하고
# hpo를 수행하는 코드

import os

import hydra
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


# config overwrite 하는 함수
def overwrite_cfg(cfg, tuner):
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

# # search epoch마다 config 파일 overwrite 하는 callback
# class CustomCallback(BaseTuner, TunerCallback):
#     def __init__(self, cfg):
#         self.cfg = cfg
#
#     def on_epoch_begin(self, epoch, logs=None):
#         print("epoch successfully started")
#         print(self.oracle)
#         print(self.tuner)
#         print(self.trial)
#
#     def on_epoch_end(self, epoch, logs=None):
#         print(self.oracle)
#         print(self.tuner)
#         print(self.trial)
#         best_trials = self.oracle.get_best_trials()
#         if len(best_trials) > 0 and self.trial.score > best_trials[0].score:
#             overwrite_cfg(self.cfg, self.tuner)


# hpo 하는 함수
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

    # hpo
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
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

        tuner.search(tg,
                     validation_data=vg)

    # callback에서 overwrite 안 할 경우 아래 실행
    overwrite_cfg(cfg, tuner)

@hydra.main(config_path="config", config_name="cfg")
def tune(cfg):

    # get generator
    tg = train_generator(cfg)
    vg = val_generator(cfg)

    # hpo
    print()
    print(f"[TUNING] {cfg.model_name} against {cfg.data.dataset} for {cfg.tune.max_epochs} epochs")
    print()

    _tune(cfg, tg, vg)

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    tune()