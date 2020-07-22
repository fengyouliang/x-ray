from pycocotools.coco import COCO
import numpy as np
import pandas as pd
import config
import cv2 as cv
from matplotlib import pyplot as plt
import os


class coco_eda:
    def __init__(self, coco_json_path):
        self.coco = COCO(coco_json_path)
        print('-' * 50)
        self.cats_with_color = self.coco.cats

        for idx, (k, v) in enumerate(self.cats_with_color.items()):
            v['color'] = (np.random.random((1, 3)) * 255).astype(np.int).tolist()[0]
            # v['color'] = config.colors[idx % len(config.colors)]

    def get_image_size(self):
        coco = self.coco
        all_images = coco.imgs
        heights = [item['height'] for k, item in all_images.items()]
        widths = [item['width'] for k, item in all_images.items()]

        return heights, widths

    def statistic_image_size(self):
        heights, widths = self.get_image_size()
        total = [(h, w) for h, w in zip(heights, widths)]
        unique = set(total)
        for k in unique:
            print('高宽为(%d,%d)的图片数量为：' % k, total.count(k))

    def show_wh(self, save_path=None):
        heights, widths = self.get_image_size()
        plt.scatter(heights, widths)
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(f"{save_path}/wh.jpg")
        else:
            plt.show()

    def show_hist_image_wh(self):
        heights, widths = self.get_image_size()
        maxrange = max(max(heights), max(widths))

        total_width = widths
        total_height = heights

        gap = 100
        group_names = [i for i in range(0, maxrange, gap)]
        width_cuts = pd.cut(total_width, bins=group_names, labels=group_names[:-1])  # ,labels=group_names
        height_cuts = pd.cut(total_height, bins=group_names, labels=group_names[:-1])  # ,labels=group_names

        total_df = pd.DataFrame({'width': width_cuts.value_counts(), 'height': height_cuts.value_counts()})
        total_df.plot(kind='bar')
        plt.show()

    def get_ann_wh(self):
        anns = self.coco.anns
        bbox = [item['bbox'] for k, item in anns.items()]
        heights = [item[2] for item in bbox]
        widths = [item[3] for item in bbox]

        return heights, widths

    def show_ann_wh(self, save_path=None):

        heights, widths = self.get_ann_wh()
        max_size = max(max(heights), max(widths)) + 10
        plt.figure(figsize=(10, 10))
        plt.xlim(0, max_size)
        plt.ylim(0, max_size)
        x = np.linspace(0, max_size)
        plt.plot(x, x, color='red', linestyle='--')
        plt.scatter(heights, widths)

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(f"{save_path}/wh.jpg")
        else:
            plt.show()

    def show_hist_ann_wh(self):
        heights, widths = self.get_ann_wh()
        maxrange = max(max(heights), max(widths))

        total_width = widths
        total_height = heights

        gap = 20
        group_names = [i for i in range(0, int(maxrange), gap)]
        width_cuts = pd.cut(total_width, bins=group_names, labels=group_names[:-1])  # ,labels=group_names
        height_cuts = pd.cut(total_height, bins=group_names, labels=group_names[:-1])  # ,labels=group_names

        total_df = pd.DataFrame({'width': width_cuts.value_counts(), 'height': height_cuts.value_counts()})
        total_df.plot(kind='bar')
        plt.show()

    def count_images_anns(self):
        images = self.coco.imgs
        annotations = self.coco.anns
        print(f'#image: {len(images)}')
        print(f'#annotations: {len(annotations)}')

    def count_annotations(self):
        annotations = self.coco.anns
        id2category = self.coco.cats
        counts_label = {v['name']: 0 for k, v in id2category.items()}
        for k, v in annotations.items():
            cat = id2category[v['category_id']]['name']
            counts_label[cat] += 1

        sorted_count_label = sorted(counts_label.items(), key=lambda x: x[1], reverse=True)

        # string show
        name_max_length = 15
        num_max_length = 6
        print_string = [f"{item[0]:{name_max_length}s}:{item[1]:{num_max_length}d} |" for item in sorted_count_label]
        tab_max_length = max([len(item) for item in print_string])
        print('-' * tab_max_length)
        print('statistic annotations: \n')
        print('-' * tab_max_length)
        print('\n'.join(print_string))
        print('-' * tab_max_length)

        # pie image show
        keys = list(counts_label.keys())
        values = list(counts_label.values())
        plt.pie(x=values, labels=keys, autopct='%.2f%%')
        plt.show()

        # hist show
        name, count = zip(*sorted_count_label)
        plt.barh(range(len(sorted_count_label)), count, height=0.7, color='steelblue', alpha=0.8)      # 从下往上画
        plt.yticks(range(len(sorted_count_label)), name)
        plt.xlim(30, 3000)
        for x, y in enumerate(count):
            plt.text(y + 0.2, x - 0.1, '%s' % y)
        plt.show()

    def coco_vis(self, image_id=None):
        coco = self.coco
        cats = self.cats_with_color
        if image_id is None:
            imgIds = coco.getImgIds()
            image_id = imgIds[np.random.randint(0, len(imgIds))]
        assert image_id in coco.getImgIds()
        coco_img = coco.loadImgs(ids=image_id)[0]
        img_id = coco_img['id']
        image_path = f"{config.image_root_path}/{coco_img['file_name']}"
        image = cv.imread(image_path)
        assert image is not None, f'{image_path}'

        annIds = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = coco.loadAnns(annIds)
        coco.showAnns(anns, draw_bbox=True)

        for ann in anns:
            label_id = ann['category_id']
            label_name = cats[label_id]['name']
            if cats is None:
                color = [0, 0, 255]
            else:
                color = cats[label_id]['color']
            x, y, w, h = list(map(int, ann['bbox']))

            cv.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=2)
            cv.putText(image, label_name, (x, y), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=2,
                       color=color)
        cv.imshow(coco_img['file_name'], image)
        cv.waitKey()
        cv.destroyAllWindows()


def main():
    coco_json_path = f'E:/download/iFYTEK/x_ray/dataset/coco/annotations/fold0/all.json'

    eda = coco_eda(coco_json_path)

    # while True:
    #     eda.coco_vis()

    # eda.show_wh()
    # eda.show_ann_wh()
    # eda.statistic_image_size()
    # eda.count_images_anns()
    # eda.count_annotations()
    # eda.show_hist_image_wh()
    eda.show_hist_ann_wh()


if __name__ == '__main__':
    main()
