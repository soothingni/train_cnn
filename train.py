# 모델 학습시키는 코드

import hydra
import os
import glob
import re
import tensorflow as tf
from tensorflow.keras.models import load_model
from generator import train_generator, val_generator
from generator import train_generator, val_generator
from models import get_model
from utils import get_callbacks

# 초기 학습 함수
def _train(cfg, tg, vg):
    callbacks = get_callbacks(cfg)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = get_model(cfg)
        model.fit(tg, validation_data=vg, callbacks=callbacks, epochs=cfg.train.epochs)

# 학습시킨 pretrained weight 있는지 확인하고
# 있을 경우 최고 성능의 모델 path를 리턴하고
# 없을 경우 False를 리턴하는 함수
def get_pretrained_path(cfg):
    checkpoint_dir = os.path.join(cfg.root, "checkpoints", cfg.model_name)
    weights = glob.glob(checkpoint_dir + '/*.hdf5')
    if weights != []:
        acc_regex = re.compile('\d+\.\d{3}.hdf5')
        accs = [float(acc_regex.search(w).group().strip('.hdf5')) for w in weights]
        max_acc = max(accs)
        best_model = weights[accs.index(max_acc)]
        best_model_path = os.path.join(checkpoint_dir, best_model)
        return best_model_path
    else:
        return None

# 이어서 학습하는 함수
def _continue_train(cfg, best_model_path):
    tg = train_generator(cfg)
    vg = val_generator(cfg)

    callbacks = get_callbacks(cfg)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = load_model(best_model_path)

        print()
        print(f"Loaded {best_model_path.split('/')[-1]} \nContinuing training from this weight file.")
        print()

        model.fit(tg, validation_data=vg, callbacks=callbacks, epochs=cfg.train.epochs)

@hydra.main(config_path="config", config_name="cfg")
def train(cfg):

    model_name = cfg.model_name

    print()
    print(f"MODEL: {model_name}")
    print()

    # get generator
    tg = train_generator(cfg)
    vg = val_generator(cfg)

    #train
    print()
    print(f"[TRAINING] {model_name} against {cfg.data.dataset} for {cfg.train.epochs} epochs")
    print()

    # pretrained weight 있는지 확인
    best_model_path = get_pretrained_path(cfg)
    if best_model_path != None:
        print(f"Found pretrained weights for {cfg.model_name}")
        print()
        _continue_train(cfg, best_model_path)

    else:
        _train(cfg, tg, vg)

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
    train()







