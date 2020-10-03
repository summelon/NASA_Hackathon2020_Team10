# Naive Classification

## Environment
* Pytorch 1.6 docker container
  `docker run -it --gpus all -v {inaturalist path}:/train-data -v $(pwd):/workspace pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime /bin/bash`
* Data processing package
  `pip install pandas scikit-learn`

### iNaturalist Path
```
/train-data
├── inaturalist-2019
│   ├── train2019.json
│   ├── train_val2019
│   │   └── Birds
│   │       ├── 000 (class folder)
│   │       ├── 001 (class folder)
│   │       ├── ... (class folder)
│   │       └── 100 (class folder)
│   └── val2019.json
```

## Usage
* `python main.py`

### Argument
* `-c`, `--config`: config file path, default is `config.yaml` in same directory

