"""
COCO 格式的数据集转化为 YOLO 格式的数据集
--json_path 输入的json文件路径
--save_path 保存的文件夹名字，默认为当前目录下的labels。
python coco2yolo.py --json_path $json_path --input_img_dir $input_img_dir --save_path $save_path --out_img_dir $out_img_dir
"""

import os
import json
import cv2
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--json_path', default='./annotations/train.json', type=str, help="input: coco format(json)")
parser.add_argument('--input_img_dir', default='./images', type=str, help="specify where to save the output dir of labels")
parser.add_argument('--save_path', default='./labels/train', type=str, help="specify where to save the output dir of labels")
parser.add_argument('--out_img_dir', default='./images/train', type=str, help="specify where to save the output dir of labels")
arg = parser.parse_args()


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


if __name__ == '__main__':
    json_file = arg.json_path  # COCO Object Instance 类型的标注
    input_img_dir = arg.input_img_dir   # 输入图像路径
    labels_save_path = arg.save_path    # 保存标注的路径
    out_img_dir = arg.out_img_dir       # 保存图像路径

    data = json.load(open(json_file, 'r'))
    if not os.path.exists(labels_save_path):
        os.makedirs(labels_save_path)
    if not os.path.exists(out_img_dir):
        os.makedirs(out_img_dir)
    id_map = {}  # coco数据集的id不连续！重新映射一下再输出！
    with open(os.path.join(labels_save_path, 'classes.txt'), 'w') as f:
        # 写入classes.txt
        for i, category in enumerate(data['categories']):
            f.write(f"{category['name']}\n")
            id_map[category['id']] = i
    # print(id_map)

    for img in tqdm(data['images']):
        filename = img["file_name"]
        img_width = img["width"]
        img_height = img["height"]
        img_id = img["id"]
        head, tail = os.path.splitext(filename)
        ana_txt_name = head + ".txt"  # 对应的txt名字，与jpg一致
        f_txt = open(os.path.join(labels_save_path, ana_txt_name), 'w')
        for ann in data['annotations']:
            if ann['image_id'] == img_id:
                box = convert((img_width, img_height), ann["bbox"])
                f_txt.write("%s %s %s %s %s\n" % (id_map[ann["category_id"]], box[0], box[1], box[2], box[3]))
        image = cv2.imread(os.path.join(input_img_dir,filename))
        cv2.imwrite(os.path.join(out_img_dir,filename),image)
        f_txt.close()