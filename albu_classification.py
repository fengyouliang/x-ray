from urllib.request import urlopen
import os
import os.path as osp
from pathlib import Path
import cv2
import numpy as np
from albumentations import (
    CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose
)
from albumentations import *
from matplotlib import pyplot as plt


def download_image(url):
    data = urlopen(url).read()
    data = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_image(image_path):
    assert Path(image_path).is_file(), image_path
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def augment_and_show(aug, image):
    aug_image = aug(image=image)['image']
    plt.figure(figsize=(10, 10))
    plt.imshow(aug_image)
    plt.show()
    return aug_image


def augment_flips_color(p=.5):
    return Compose([
        CLAHE(),
        RandomRotate90(),
        Transpose(),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
        Blur(blur_limit=3),
        OpticalDistortion(),
        GridDistortion(),
        HueSaturationValue()
    ], p=p)


def strong_aug(p=.5):
    return Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=0.3),
        HueSaturationValue(p=0.3),
    ], p=p)


def main():
    # image = download_image('https://d177hi9zlsijyy.cloudfront.net/wp-content/uploads/sites/2/2018/05/11202041'
    #                        '/180511105900-atlas-boston-dynamics-robot-running-super-tease.jpg')
    import config
    image_root = config.image_root_path
    image_id = 100001
    image = load_image(f"{image_root}{image_id}.jpg")
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.show()

    aug = RandomBrightnessContrast(0.5, 0.5, p=1)
    augment_and_show(aug, image)

    # aug = HorizontalFlip(p=1)
    # augment_and_show(aug, image)
    #
    # aug = IAAPerspective(scale=0.2, p=1)
    # augment_and_show(aug, image)
    #
    # aug = ShiftScaleRotate(p=1)
    # augment_and_show(aug, image)
    #
    # aug = augment_flips_color(p=1)
    # augment_and_show(aug, image)

    # aug = strong_aug(p=1)
    # augment_and_show(aug, image)


if __name__ == '__main__':
    main()
