# 모델 학습시키는 코드

import hydra
from tuner import _tune, _train
from generator import train_generator, val_generator

@hydra.main(config_path="config", config_name="cfg")
def train(cfg):

    model_name = cfg.model_name
    print(f"MODEL: {model_name}")

    # get generator
    tg = train_generator(cfg)
    vg = val_generator(cfg)

    # hpo
    if cfg.tune.tf == True:
        print("=====================================================================================")
        print(f"[TUNING] {model_name} against {cfg.data.dataset} for {cfg.tune.max_epochs} epochs")
        print("=====================================================================================")

        _tune(cfg, tg, vg)

    #train
    print("=====================================================================================")
    print(f"[TRAINING] {model_name} against {cfg.data.dataset} for {cfg.train.epochs} epochs")
    print("=====================================================================================")

    _train(cfg, tg, vg)

    print("=====================================================================================")
    print(f"All processes for {model_name} ended successfully")
    print("=====================================================================================")

if __name__ == "__main__":
    train()







