import os
import tqdm
import glob
import shutil
import numpy as np
from PIL import Image
import torchvision.transforms as trans
from argparse import ArgumentParser


def npy_trans_func(img):
    img = trans.Resize((240, 240))(img)
    img = np.array(img, dtype=np.float32)
    img = img / 255
    img = img.transpose(2, 0, 1)
    img = img.reshape(1, *img.shape)

    return img


def bin_trans_func(img, dtype):
    img = trans.Resize((240, 240))(img)
    img = np.array(img, np.uint8 if dtype == 'uint8' else np.float32)
    img = img.transpose(2, 0, 1)
    img = img.reshape(1, *img.shape)

    return img


def select_data(cls_list):
    dataset_dir = "/home/data/inaturalist-2019/train_val2019/Birds"
    img_pts = list()
    for cls in cls_list:
        cls_img_pts = glob.glob(os.path.join(dataset_dir, str(cls), '*.jpg'))
        img_pts += cls_img_pts

    return img_pts


def main():
    parser = ArgumentParser()
    parser.add_argument("--ftype", choices=['bin', 'npy'],
                        default='bin', help="File type to dump")
    parser.add_argument("--dtype", choices=['uint8', 'float32'],
                        default='uint8', help="Data type for bin file")
    args, _ = parser.parse_known_args()

    if args.ftype == 'bin':
        if args.dtype == 'uint8':
            data_path = './data/bird/bin_uint8'
        elif args.dtype == 'float32':
            data_path = './data/bird/bin_fp32'
        else:
            raise ValueError(f"Your dtype {args.dtype} is not correct!")
        class_dict = {230: 0, 234: 1, 247: 2, 272: 3, 285: 4}
        image_paths = select_data(class_dict.keys())
    elif args.ftype == 'npy':
        class_list = [202, 203, 204, 205, 209, 211, 212, 213, 218, 219,
                      222, 225, 231, 232, 233, 234, 235, 236, 238, 240,
                      244, 246, 247, 248, 249, 254, 255, 259, 262, 263,
                      266, 268, 270, 272, 273, 275, 276, 277, 278, 279,
                      280, 281, 282, 283, 284, 285, 288, 294, 296, 297,
                      301, 306, 208, 311, 313, 314, 318, 319, 320, 321,
                      322, 323, 327]
        data_path = './data/bird/npy'
        image_paths = select_data(class_list)
    else:
        raise ValueError(f"Your ftype {args.ftype} is not correct!")
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    os.makedirs(data_path)
    pbar = tqdm.tqdm(image_paths)
    for idx, path in enumerate(pbar):
        image = Image.open(path).convert('RGB')
        if args.ftype == 'bin':
            new_name = os.path.join(
                    data_path,
                    str(class_dict[int(path.split('/')[-2])])+'_'+str(idx))
            image = bin_trans_func(image, args.dtype)
            image.tofile(new_name+'.bin')
        elif args.ftype == 'npy':
            new_name = os.path.join(
                    data_path, path.split('/')[-2]+'_'+str(idx))
            image = npy_trans_func(image)
            np.save(new_name, image)


if __name__ == "__main__":
    main()
