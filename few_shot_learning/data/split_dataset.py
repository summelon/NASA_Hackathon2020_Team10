import glob
import random
import pandas as pd


def dump_json(data_dict: dict, json_name: str) -> bool:
    labels = [i for i, pts in enumerate(list(data_dict.values()))
              for n in range(len(pts))]
    meta_dataframe = pd.DataFrame.from_dict(
            {'label_names': [list(data_dict.keys())],
             'image_names': [[p for pts in data_dict.values() for p in pts]],
             'image_labels': [labels]})
    meta_dataframe.to_json(json_name, orient='records', lines=True)

    return True


def main():
    dataset_pt = "/home/data/inaturalist-2019/train_val2019/Birds"
    ratio = 0.9
    all_class = glob.glob(dataset_pt+'/*')
    new_dict = dict()
    for class_dir in all_class:
        imgs = glob.glob(class_dir+'/*')
        if len(imgs) > 500:
            new_dict[class_dir.split('/')[-1]] = imgs

    train_dict, val_dict = dict(), dict()
    for cls, pts in new_dict.items():
        train_pts = random.sample(pts, int(ratio*len(pts)))
        train_dict[cls] = train_pts
        val_dict[cls] = list(set(pts) - set(train_pts))

    _ = dump_json(train_dict, 'train/base.json')
    _ = dump_json(val_dict, 'train/val.json')


if __name__ == "__main__":
    main()
