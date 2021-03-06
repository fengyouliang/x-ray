import cv2 as cv
import numpy as np
# from pycocotools.coco import COCO
from coco import COCO
import config
from albumentations import *
from pathlib import Path


def my_coco_vis(coco, cats, image_id=None, augs=None, rotate=None):
    assert rotate in [None, 0, 1, 2]
    if image_id is None:
        imgIds = coco.getImgIds()
        image_id = imgIds[np.random.randint(0, len(imgIds))]
    assert image_id in coco.getImgIds()
    coco_img = coco.loadImgs(ids=image_id)[0]
    img_id = coco_img['id']
    image_path = f"{config.image_root_path}/{coco_img['file_name']}"
    assert Path(image_path).is_file(), image_path

    image = cv.imread(image_path)
    assert image is not None, f'{image_path}'
    ori_image = image.copy()

    if augs is not None:
        for aug in augs:
            image = aug(image=image)['image']

    annIds = coco.getAnnIds(imgIds=img_id, iscrowd=None)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns, draw_bbox=True)

    for ann in anns:
        label_id = ann['category_id']
        label_name = cats[label_id]['name']
        if cats[label_id].get('color'):
            color = cats[label_id]['color']
        else:
            color = [0, 0, 0]
        x, y, w, h = list(map(int, ann['bbox']))

        cv.rectangle(ori_image, (x, y), (x + w, y + h), color=color, thickness=2)
        cv.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=2)

        cv.putText(ori_image, label_name, (x, y), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=2, color=color)
        cv.putText(image, label_name, (x, y), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=2, color=color)

    if rotate is not None:
        ori_image = cv.rotate(ori_image, rotate)
        image = cv.rotate(image, rotate)

    h, w = ori_image.shape[:2]
    if h > w or 0.7 < h / w < 1.3:
        concat_image = np.concatenate([ori_image, image], axis=1)  # w
    else:
        concat_image = np.concatenate([ori_image, image], axis=0)  # h

    window_name = coco_img['file_name'] + '  ori_aug'
    cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)
    cv.imshow(window_name, concat_image)
    cv.waitKey()
    cv.destroyAllWindows()


def main():
    x_ray = f'E:/iFLYTEK/x_ray/train\coco/annotations/fold0/all.json'
    # x_ray = f'E:/iFLYTEK/x_ray/train\coco/annotations/all.json'
    # x_ray = f'E:/download/iFYTEK/video_ad/train/train_seg.json'

    coco = COCO(x_ray)
    cats = coco.cats

    # for idx, (k, v) in enumerate(cats.items()):
    #     v['color'] = (np.random.random((1, 3)) * 255).astype(np.int).tolist()[0]
    #     v['color'] = config.colors[idx % len(config.colors)]

    aug_hue = HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1)
    aug_bc = RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=1)
    aug_clahe = CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1)
    aug_inv = InvertImg(p=1)
    aug_cut_out = CoarseDropout(max_holes=20, max_height=8, max_width=8, fill_value=0, p=1)
    aug_gamma = RandomGamma(gamma_limit=(80, 120), eps=None, p=1)
    aug_od = OpticalDistortion(distort_limit=0.05, shift_limit=0.05, interpolation=cv.INTER_CUBIC, value=None, mask_value=None, p=1)

    augs = [aug_hue, ]

    while True:
        my_coco_vis(coco, cats=cats, image_id=None, augs=augs, rotate=None)


if __name__ == '__main__':
    main()
