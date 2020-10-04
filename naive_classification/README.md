# Naive Classification
The naive classification using `resnet18` to classify iNaturalist data.

## Environment
* Pytorch 1.6 docker container
  ```
  docker run -it --gpus all -v {inaturalist path}:/train-data -v $(pwd):/workspace pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime /bin/bash
  ```
* Data processing package
  ```
  pip install pandas scikit-learn
  ```

### iNaturalist Path
```
/train-data
└── inaturalist-2019
    ├── train2019.json
    ├── train_val2019
    │   └── Birds
    │       ├── 000 (class folder)
    │       ├── 001 (class folder)
    │       ├── ... (class folder)
    │       └── 100 (class folder)
    └── val2019.json
```

### Json Format
```
{
  "info" : info,
  "images" : [image],
  "categories" : [category],
  "annotations" : [annotation],
  "licenses" : [license]
}

info {
  "year" : int,
  "version" : str,
  "description" : str,
  "contributor" : str,
  "url" : str,
  "date_created" : datetime,
}

image {
  "id" : int,
  "width" : int,
  "height" : int,
  "file_name" : str,
  "license" : int,
  "rights_holder" : str
}

category {
  "id" : int,
  "name" : str,
  "kingdom" : str,
  "phylum" : str,
  "class" : str,
  "order" : str,
  "family" : str,
  "genus" : str
}

annotation {
  "id" : int,
  "image_id" : int,
  "category_id" : int
}

license {
  "id" : int,
  "name" : str,
  "url" : str
}
```

## Usage
* copy `config_example.yaml` to `config.yaml`
* update information in `config.yaml`
* `python main.py`

### Argument
* `-c`, `--config`: config file path, default is `config.yaml` in same directory
* `-m`, `--mode`: model mode, will transform data, should be `day` or `night`, default is `day`
