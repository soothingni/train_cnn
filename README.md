# train_cnn
automated training using various cnn architectures (hpo + training)

```
pip install -r requirements.txt
```

## 1. hpo

```
python tune.py backbone=DenseNet169,EfficientNetB3,InceptionV3 -m
```

## 2. train

```
python train.py backbone=DenseNet169,EfficientNetB3,InceptionV3 -m
```


**important** there should be no space after the commas between backbone options!
