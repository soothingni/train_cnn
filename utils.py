# 학습용 잡동사니 함수들 정의한 코드

import os
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard, CSVLogger

#learning rate scheduler
def get_scheduler(epochs, lr):
    class Schedule:
        def __init__(self, epochs, lr):
            self.epochs = epochs
            self.initial_lr = lr

        def __call__(self, epoch_idx):
            if epoch_idx < self.epochs * 0.25:
                return self.initial_lr
            elif epoch_idx < self.epochs * 0.50:
                return self.initial_lr * 0.2
            elif epoch_idx < self.epochs * 0.75:
                return self.initial_lr * 0.04
            return self.initial_lr * 0.008
    return Schedule(epochs, lr)

#callbacks
def get_callbacks(cfg):
    checkpoint_dir = os.path.join(cfg.root, "checkpoints", cfg.model_name)
    filename = "{epoch:02d}-{val_loss:.3f}-{val_accuracy:.3f}.hdf5"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    callbacks = [LearningRateScheduler(schedule=get_scheduler(cfg.train.epochs, cfg.hp.learning_rate)),
                 ModelCheckpoint(str(checkpoint_dir) + "/" + filename,
                                 monitor="val_accuracy",
                                 verbose=1,
                                 save_best_only=True,
                                 mode="auto"),
                 TensorBoard(log_dir=f'./logs/{cfg.model_name}')]
    return callbacks