# train_cnn
automated training using various cnn architectures (hpo + training)

```
pip install -r requirements.txt
```

```
python train.py backbone=DenseNet169,EfficientNetB3,InceptionV3 -m
```

**important** there should be no space after the commas between backbone options!
