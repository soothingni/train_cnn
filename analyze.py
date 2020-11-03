# 다양한 모델 checkpoint
# 비교 분석 및 최고 성능 모델
# 리턴

import os
import re
import hydra

@hydra.main(config_path="config", config_name="cfg")
def find_best(cfg):
    """
    weight 파일 디렉토리에서
    loss와 accuracy 비교 후
    최고 모델과 loss, accuracy를
    출력해주는 함수
    """
    models = list(cfg.backbone_options)

    loss = 10000
    loss_model = ''
    loss_epoch = 0

    acc = 0
    acc_model = ''
    acc_epoch = 0

    for backbone in models:
        try:
            path = os.path.join(cfg.root, 'checkpoints', backbone)
            weights = os.listdir(path)
            loss_regex = re.compile("-\d+\.\d{3}-")
            acc_regex = re.compile('\d+\.\d{3}.hdf5')
            losses = [float(loss_regex.search(w).group().strip('-')) for w in weights]
            accs = [float(acc_regex.search(w).group().strip('.hdf5')) for w in weights]
            min_loss = min(losses)
            min_loss_epoch = os.listdir(path)[losses.index(min_loss)].split('-')[0]
            max_acc = max(accs)
            max_acc_epoch = os.listdir(path)[accs.index(max_acc)].split('-')[0]

            if min_loss < loss:
                loss = min_loss
                loss_model = backbone
                loss_epoch = min_loss_epoch
            if max_acc > acc:
                acc = max_acc
                acc_model = backbone
                acc_epoch = max_acc_epoch

        except:
            continue


    script = f"""
                [ANALYSIS RESULT] \n
                -Lowest LOSS model is {loss_model} epoch {loss_epoch} at {loss} \n
                -Highest ACCURACY model is {acc_model} epoch {acc_epoch} at {acc}"""

    print()
    print(script)
    print()

if __name__ == "__main__":
    find_best()