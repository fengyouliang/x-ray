from pycocotools.coco import COCO
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
import os.path as osp
import config
import sys
import albumentations


x_ray = f'{config.root_dir}/coco/annotations/fold0/all.json'
image_root_path = f'{config.root_dir}/coco/images'

coco = COCO(x_ray)
cats = coco.cats
colors = [
    [0, 139, 139],
    [28, 28, 28],
    [139, 0, 0],
    [0, 0, 139],
    [85, 26, 139],
    [139, 34, 82],
    [205, 0, 0],
    [255, 165, 0],
    [255, 255, 0],
    [0, 255, 0],
]
for idx, (k, v) in enumerate(cats.items()):
    v['color'] = (np.random.random((1, 3)) * 255).astype(np.int).tolist()[0]
    v['color'] = colors[idx]


def vis_image(image_id=None):
    # catIds = coco.getCatIds()
    if image_id == None:
        imgIds = coco.getImgIds()
        image_id = imgIds[np.random.randint(0, len(imgIds))]
    img = coco.loadImgs(ids=image_id)[0]
    id = img['id']
    image_path = f"{image_root_path}/{img['filename']}"
    assert osp.exists(image_path)
    I = cv.imread(image_path)
    assert I is not None
    I = cv.cvtColor(I, cv.COLOR_RGB2BGR)
    plt.axis('off')
    plt.imshow(I)
    # plt.show()

    annIds = coco.getAnnIds(imgIds=id, iscrowd=None)
    anns = coco.loadAnns(annIds)
    print(f'anns: {anns}')
    coco.showAnns(anns, draw_bbox=True)
    plt.show()


def my_coco_vis(image_id=None):
    if image_id == None:
        imgIds = coco.getImgIds()
        image_id = imgIds[np.random.randint(0, len(imgIds))]
    assert image_id in coco.getImgIds()
    coco_img = coco.loadImgs(ids=image_id)[0]
    img_id = coco_img['id']
    image_path = f"{image_root_path}/{coco_img['filename']}"
    image = cv.imread(image_path)

    annIds = coco.getAnnIds(imgIds=img_id, iscrowd=None)
    anns = coco.loadAnns(annIds)

    for ann in anns:
        label_id = ann['category_id']
        label_name = cats[label_id]['name']
        color = cats[label_id]['color']
        x, y, w, h = list(map(int, ann['bbox']))

        cv.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=2)
        cv.putText(image, label_name, (x, y), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=2, color=color)
    cv.imshow('demo', image)
    cv.waitKey()
    cv.destroyAllWindows()


def main():
    while True:
        my_coco_vis()


if __name__ == '__main__':
    main()
