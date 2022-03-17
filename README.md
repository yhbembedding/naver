# Naver Project
Ptit x Naver
## About the project
Indoor Segmantic segmentation
pip install -r requirements.txt

### set up env
python >=3.6

```
pip install -r requirements.txt
```
### Dataset
You can download the datasets in the [DRIVER](https://drive.google.com/drive/folders/1r0-Hu0WxwZBnL_WaNbg6S7uoD_4Ml9xi?usp=sharing)
```
To use dataset, your dataset must be organized as follow:
```
```
data5
├── image
    ├──── 40027089_1575338780779869.jpg
    ├──── 40027089_1575338780779869.jpg
      .......
├── mask
    ├──── 40027089_1575338780779869.png
    ├──── 40027089_1575338780779869.png
    .....
```
The pixel of the label has a value from 0 to 23 corresponding to the list
```
['background','air_conditioner','bicycle','cabinet','celling','chair','door','floor','guard_rail','light','monitor','person',
'placard','seat','sign','stairs','telephone','trash_can','unknown','vending_machine','wall','window','metro','shelf']
```

### Training

```
python train.py --model deeplabv3 --loss focal
```
### predict

```
python predict.py
```
### eval

```
python test.py
```
