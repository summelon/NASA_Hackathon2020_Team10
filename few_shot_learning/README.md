# Few-shot Learning

## Prerequisites
- Python == 3.6
- Pytorch == 1.5.0
- Install dependencies rapidly
```shell
pip3 install -r requirements.txt
```

## How to reproduce
- For the reproducibility of our experiments, we provide our random seed `41608` as default
- Dump meta-data json file
```shell
python3 ./data/split_dataset.py --dataset_pt YOUR_DATASET_PATH --json_pt TARGET_JSON_FILE
```
### Training
- Train [day | night] view model
```shell
python3 train.py --name PROJECT_NAME --dataset TARGET_JSON_FILE --train_aug [  | --night]
```
- Evaluate [day | night] view model
```shell
python3 test.py --name PROJECT_NAME --dataset TARGET_JSON_FILE [  | --night]
```

## Experimental Results
- [Few-shot Learning test](https://hackmd.io/VFJFaVocS5-4lMsAhypkFg)

## Reference
- The code is built upon the implementation from [DropOut](https://github.com/hytseng0509/DropGrad)
