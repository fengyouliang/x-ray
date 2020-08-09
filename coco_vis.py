import cv2 as cv
import numpy as np
# from pycocotools.coco import COCO
from coco import COCO
import config


def my_coco_vis(coco, cats, image_id=None, rotate=None):
    assert rotate in [None, 0, 1, 2]
    if image_id is None:
        imgIds = coco.getImgIds()
        image_id = imgIds[np.random.randint(0, len(imgIds))]
    assert image_id in coco.getImgIds()
    coco_img = coco.loadImgs(ids=image_id)[0]
    img_id = coco_img['id']
    image_path = f"{config.image_root_path}/{coco_img['filename']}"
    image = cv.imread(image_path)
    assert image is not None, f'{image_path}'

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

        cv.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=2)
        cv.putText(image, label_name, (x, y), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=2, color=color)
    if rotate is not None:
        image = cv.rotate(image, rotate)
    cv.imshow(coco_img['filename'], image)
    cv.waitKey()
    cv.destroyAllWindows()


def main():
    x_ray = f'E:/iFLYTEK/x_ray/train\coco/annotations/all.json'
    # x_ray = f'E:/download/iFYTEK/video_ad/train/train_seg.json'

    coco = COCO(x_ray)
    cats = coco.cats

    # for idx, (k, v) in enumerate(cats.items()):
    #     v['color'] = (np.random.random((1, 3)) * 255).astype(np.int).tolist()[0]
    #     v['color'] = config.colors[idx % len(config.colors)]

    while True:
        my_coco_vis(coco, cats=cats, image_id=100001, rotate=None)


if __name__ == '__main__':
    main()
