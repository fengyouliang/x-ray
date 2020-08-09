import cv2 as cv
# from pycocotools.coco import COCO
from coco import COCO
import json
import config
import glob
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, init_shape, meter_info='AverageMeter'):
        self.shape = init_shape
        self.meter_info = meter_info
        self.val = np.zeros(shape=self.shape)
        self.avg = np.zeros(shape=self.shape)
        self.sum = np.zeros(shape=self.shape)
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        assert list(val.shape) == self.shape, f"{list(val.shape)}-{self.shape} not matched!"

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"{self.meter_info}: {self.avg}"


def compute_iou(box1, box2, wh=False):
    """
    compute the iou of two boxes.
    Args:
        box1, box2: [xmin, ymin, xmax, ymax] (wh=False) or [xcenter, ycenter, w, h] (wh=True)
        wh: the format of coordinate.
    Return:
        iou: iou of box1 and box2.
    """
    if not wh:
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
    else:
        xmin1, ymin1 = int(box1[0] - box1[2] / 2.0), int(box1[1] - box1[3] / 2.0)
        xmax1, ymax1 = int(box1[0] + box1[2] / 2.0), int(box1[1] + box1[3] / 2.0)
        xmin2, ymin2 = int(box2[0] - box2[2] / 2.0), int(box2[1] - box2[3] / 2.0)
        xmax2, ymax2 = int(box2[0] + box2[2] / 2.0), int(box2[1] + box2[3] / 2.0)

    # 获取矩形框交集对应的左上角和右下角的坐标（intersection）
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])

    # 计算两个矩形框面积
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))
    iou = inter_area / (area1 + area2 - inter_area + 1e-6)

    return iou


def coco_merge(coco_a, coco_b):
    a = json.load(open(coco_a, 'r', encoding='utf-8'))
    b = json.load(open(coco_b, 'r', encoding='utf-8'))
    assert a['cata']


def test_iou():
    x_ray = f'{config.root_dir}/coco/annotations/fold0/all.json'
    coco = COCO(x_ray)

    image_id = 600037
    coco_img = coco.loadImgs(ids=image_id)[0]
    img_id = coco_img['id']
    image_path = f"{config.image_root_path}/{coco_img['filename']}"

    I = cv.imread(image_path)
    assert I is not None
    I = cv.cvtColor(I, cv.COLOR_RGB2BGR)
    plt.axis('off')
    plt.imshow(I)

    annIds = coco.getAnnIds(imgIds=img_id, iscrowd=None)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns, draw_bbox=True)
    plt.show()
    rects = [ann['bbox'] for ann in anns if ann['category_id'] == 1]
    print(rects)
    iou = compute_iou(rects[0], rects[1], wh=True)
    print(iou)


def compute_mean_std():

    all_images = glob.glob('E:/iFLYTEK/x_ray/train/all_train' + '/**/*.jpg', recursive=True)
    mean_meter = AverageMeter([3], meter_info='image RGB mean average meter')
    std_meter = AverageMeter([3], meter_info='image RGB std average meter')
    pbar = tqdm(all_images, total=len(all_images))
    for image_path in pbar:
        pbar.set_description(image_path)
        p = Path(image_path)
        image = cv.imread(p.as_posix())
        mean = np.mean(image, axis=(0, 1))
        std = np.std(image, axis=(0, 1))

        mean_meter.update(mean)
        std_meter.update(std)

    print(mean_meter)  # [191.85634426 207.56213923 202.32689929]
    print(std_meter)  # [59.52199661 52.59761332 66.76140793]


if __name__ == '__main__':
    compute_mean_std()
