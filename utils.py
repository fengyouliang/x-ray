import pickle
from pycocotools.coco import COCO
import config


def load_pkl():
    with open(f'ann_count/count.pkl', 'rb') as fid:
        d = pickle.load(fid)
    return d


def load_coco():
    x_ray = f'{config.root_dir}/coco/annotations/fold0/all.json'
    coco = COCO(x_ray)
    return coco


def check_coco():
    ann_count = load_pkl()
    coco = load_coco()

    ann_sum = sum(list(ann_count.values()))
    ann_id = coco.getAnnIds()

    for name, v in ann_count.items():
        annIds = coco.getAnnIds(imgIds=int(name), iscrowd=None)
        anns = coco.loadAnns(annIds)
        if len(anns) != v:
            print(name)

    print(ann_sum, len(ann_id))


def main():
    check_coco()


if __name__ == '__main__':
    main()
