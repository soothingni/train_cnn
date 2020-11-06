# 하이퍼파라미터 탐색을 수행하는
# keras tuner 객체 생성하고
# hpo를 수행하는 코드

import os

import hydra
from omegaconf import OmegaConf

import functools

import kerastuner as kt
from kerastuner.tuners.hyperband import Hyperband, HyperbandOracle
from kerastuner.engine import trial as trial_module
from generator import train_generator, val_generator
from models import get_model
from utils import get_callbacks

import sys
from io import StringIO

import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.backend import clear_session
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, BatchNormalization, Activation, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping

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
def overwrite_cfg(tuner, cfg, best_score):
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
    config.hp.best_value = float(best_score)
    OmegaConf.save(config, config_path)

    print()
    print(f"Config for {cfg.model_name} overwritten")
    print()

class MyHyperband(Hyperband):
    """Variation of HyperBand algorithm.
    Reference:
        Li, Lisha, and Kevin Jamieson.
        ["Hyperband: A Novel Bandit-Based
         Approach to Hyperparameter Optimization."
        Journal of Machine Learning Research 18 (2018): 1-52](
            http://jmlr.org/papers/v18/16-558.html).
    # Arguments
        hypermodel: Instance of HyperModel class
            (or callable that takes hyperparameters
            and returns a Model instance).
        objective: String. Name of model metric to minimize
            or maximize, e.g. "val_accuracy".
        max_epochs: Int. The maximum number of epochs to train one model. It is
          recommended to set this to a value slightly higher than the expected time
          to convergence for your largest Model, and to use early stopping during
          training (for example, via `tf.keras.callbacks.EarlyStopping`).
        factor: Int. Reduction factor for the number of epochs
            and number of models for each bracket.
        hyperband_iterations: Int >= 1. The number of times to iterate over the full
          Hyperband algorithm. One iteration will run approximately
          `max_epochs * (math.log(max_epochs, factor) ** 2)` cumulative epochs
          across all trials. It is recommended to set this to as high a value
          as is within your resource budget.
        seed: Int. Random seed.
        hyperparameters: HyperParameters class instance.
            Can be used to override (or register in advance)
            hyperparamters in the search space.
        tune_new_entries: Whether hyperparameter entries
            that are requested by the hypermodel
            but that were not specified in `hyperparameters`
            should be added to the search space, or not.
            If not, then the default value for these parameters
            will be used.
        allow_new_entries: Whether the hypermodel is allowed
            to request hyperparameter entries not listed in
            `hyperparameters`.
        **kwargs: Keyword arguments relevant to all `Tuner` subclasses.
            Please see the docstring for `Tuner`.
    """

    def __init__(self,
                 hypermodel,
                 objective,
                 max_epochs,
                 factor=3,
                 hyperband_iterations=1,
                 seed=None,
                 hyperparameters=None,
                 tune_new_entries=True,
                 allow_new_entries=True,
                 cfg = None,
                 **kwargs):
        oracle = HyperbandOracle(
            objective,
            max_epochs=max_epochs,
            factor=factor,
            hyperband_iterations=hyperband_iterations,
            seed=seed,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries)
        super(Hyperband, self).__init__(
            oracle=oracle,
            hypermodel=hypermodel,
            **kwargs)
        self.cfg = cfg

    def on_trial_end(self, trial):
        """A hook called after each trial is run.
        # Arguments:
            trial: A `Trial` instance.
        """
        # Send status to Logger
        if self.logger:
            self.logger.report_trial_state(trial.trial_id, trial.get_state())

        self.oracle.end_trial(
            trial.trial_id, trial_module.TrialStatus.COMPLETED)
        self.oracle.update_space(trial.hyperparameters)
        # Display needs the updated trial scored by the Oracle.
        self._display.on_trial_end(self.oracle.get_trial(trial.trial_id))
        self.save()

        #overwrite check
        best_trials = self.oracle.get_best_trials()
        config_path = os.path.join(self.cfg.root, f"config/backbone/{self.cfg.model_name}.yaml")
        # 새로운 값으로 다시 불러오기 위해 config 매번 새로 로드
        config_best_value = OmegaConf.load(config_path).hp.best_value
        config_best_value = float(self.cfg.hp.best_value)
        current_value = trial.score
        current_vale = float(current_value)
        # print(f"config_best_value: {type(config_best_value)}, {config_best_value}")
        # print(f"current_value: {type(current_value)}, {current_value}")
        if len(best_trials) > 0:
          if current_value > config_best_value:
            overwrite_cfg(self, self.cfg, current_value)
          else:
            print()
            print(f"Current value({current_value}) is lower than config best_value({config_best_value}). Not updating config")
            print()

    def search(self, *fit_args, **fit_kwargs):
        """Performs a search for best hyperparameter configuations.
        # Arguments:
            *fit_args: Positional arguments that should be passed to
              `run_trial`, for example the training and validation data.
            *fit_kwargs: Keyword arguments that should be passed to
              `run_trial`, for example the training and validation data.
        """
        if 'verbose' in fit_kwargs:
            self._display.verbose = fit_kwargs.get('verbose')
        self.on_search_begin()
        while True:
            trial = self.oracle.create_trial(self.tuner_id)
            if trial.status == trial_module.TrialStatus.STOPPED:
                # Oracle triggered exit.
                tf.get_logger().info('Oracle triggered exit')
                break
            if trial.status == trial_module.TrialStatus.IDLE:
                # Oracle is calculating, resend request.
                continue

            self.on_trial_begin(trial)
            self.run_trial(trial, *fit_args, **fit_kwargs)
            self.on_trial_end(trial)
        self.on_search_end()

    def run_trial(self, trial, *fit_args, **fit_kwargs):
        hp = trial.hyperparameters
        if 'tuner/epochs' in hp.values:
            fit_kwargs['epochs'] = hp.values['tuner/epochs']
            fit_kwargs['initial_epoch'] = hp.values['tuner/initial_epoch']
        super(MyHyperband, self).run_trial(trial, *fit_args, **fit_kwargs)

    def _build_model(self, hp):
        model = super(MyHyperband, self)._build_model(hp)
        if 'tuner/trial_id' in hp.values:
            trial_id = hp.values['tuner/trial_id']
            history_trial = self.oracle.get_trial(trial_id)
            # Load best checkpoint from this trial.
            model.load_weights(self._get_checkpoint_fname(
                history_trial.trial_id, history_trial.best_step))
        return model


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
        tuner = MyHyperband(
            hypermodel=wrapped_build_func,
            objective='val_accuracy',
            max_epochs=cfg.tune.max_epochs,
            factor=2,
            hyperband_iterations=cfg.tune.hyperband_iterations,
            distribution_strategy=tf.distribute.MirroredStrategy(),
            # directory=save_path
            directory=os.path.normpath('C:/'),
            cfg = cfg
        )

        tuner.search(tg,
                     validation_data=vg,
                     callbacks = [EarlyStopping(monitor='val_accuracy', mode='max', baseline=1.0)]
                     )

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