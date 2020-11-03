# 데이터 제너레이터 환경설정 후
# 제너레이터 객체 생성하는 코드

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def train_generator(cfg):
    if cfg.data.aug == True:
        img_size = cfg.img_size
        batch_size = cfg.batch_size
        train_gen = ImageDataGenerator(rescale=1. / 255,  # pixel normalization
                                       rotation_range=15,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       brightness_range=[0.3, 0.7],
                                       shear_range=0.5,
                                       zoom_range=0.3,
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       fill_mode="nearest")
    else:
        train_gen = ImageDataGenerator(rescale=1. / 255)

    train_dir = os.path.join(cfg.root, cfg.data.train_dir)

    train_generator = train_gen.flow_from_directory(train_dir,
                                                    batch_size=batch_size,
                                                    target_size=(img_size, img_size))
    return train_generator

def val_generator(cfg):
    img_size = cfg.img_size
    batch_size = cfg.batch_size
    val_gen = ImageDataGenerator(rescale=1. / 255)  # pixel normalization

    test_dir = os.path.join(cfg.root, cfg.data.test_dir)

    val_generator = val_gen.flow_from_directory(test_dir,
                                                batch_size=batch_size,
                                                target_size=(img_size, img_size))
    return val_generator