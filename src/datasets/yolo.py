import os
import subprocess

import numpy as np
import skimage.io

from datasets.base import BaseDataset
from utils.boxes import generate_anchors
from PIL import Image
from torchvision.datasets.folder import default_loader
from pathlib import Path
import glob
import random
IMG_FORMATS = ['jpg', 'jpeg', 'png']  # acceptable image suffixes


class YOLO(BaseDataset):
    def __init__(self, phase, cfg):
        super(YOLO, self).__init__(phase, cfg)
        self.cfg = cfg
        self.input_size = cfg.input_size  # (height, width), both dividable by 16
        self.resized_image_size = cfg.resized_image_size
        self.data_root = '/home/hazen/workspace/datasets/yolo_format'
        data_dict = {
            'train':{
                    # 'alhajjcam0_yolo/data/train.txt': -1,
                    # 'alhajjcam1_yolo/data/train.txt': -1,
                    # 'detrac_yolo/data/train.txt': -1
                    # 'idd_yolo/data/train.txt': 5000,
                    # 'bdd_yolo/data/train.txt': 5000,
                    # 'riyad_yolo/data/train.txt': -1,
                    # 'kitti_yolo/data/train.txt': -1,
                    'nuimages_yolo/data/train.txt': -1,
                    # 'karachi_yolo/data/train.txt': -1,

                    },
            'val':{
                    'detrac_yolo/data/val.txt': 50
                },
        }

        
        self.all_class_names =  [
            'person', 'bicycle', 'car', 'motorcycle', 'bus',
            'train', 'truck', 'trailer', 'autorickshaw', 'trafficlight',
            'trafficsign', 'animal', 'van', 'rider', 'not_a_class'
        ]

        self.map_class_names = [
            'person', 'bike', 'car', 'bike', 'bus',
            'not_a_class', 'truck', 'truck', 'not_a_class', 'not_a_class',
            'not_a_class', 'not_a_class', 'bus', 'person', 'not_a_class'
        ]
        self.class_names = ['background', 'person', 'bike', 'car', 'bus', 'truck']
        # Kitti mean
        self.rgb_mean = np.array([93.877, 98.801, 95.923], dtype=np.float32).reshape(1, 1, 3)
        self.rgb_std = np.array([78.782, 80.130, 81.200], dtype=np.float32).reshape(1, 1, 3)

        # real_filtered mean and std
        # self.rgb_mean = np.array([94.87347, 96.89165, 94.70493], dtype=np.float32).reshape(1, 1, 3)
        # self.rgb_std = np.array([53.869507, 53.936283, 55.2807], dtype=np.float32).reshape(1, 1, 3)
        
        # real_filtered plus all_sites_seatbelt mean and std
        # self.rgb_mean = np.array([104.90631, 105.41336, 104.70162], dtype=np.float32).reshape(1, 1, 3)
        # self.rgb_std = np.array([50.69564, 49.60443, 50.158844], dtype=np.float32).reshape(1, 1, 3)

        # real+synthv2+synthv2_SB>h8_w8
        # self.rgb_mean = np.array([95.6651, 93.45838, 78.97777], dtype=np.float32).reshape(1, 1, 3)
        # self.rgb_std = np.array([64.098885, 61.599213, 56.8366  ], dtype=np.float32).reshape(1, 1, 3)

        self.num_classes = len(self.class_names)
        self.class_ids_dict = {cls_name: cls_id for cls_id, cls_name in enumerate(self.class_names)}

        # self.data_dir = os.path.join(cfg.data_dir, 'kitti')
        self.sample_ids= self.get_sample_ids(data_dict)

        self.grid_size = tuple(x //cfg.stride  for x in self.resized_image_size)  # anchors grid 
        self.anchors_seed = np.array([[ 29, 17], [46, 32], [69, 52],
                                        [109, 68], [84, 127], [155, 106], 
                                        [255, 145], [183, 215], [371, 221]], dtype=np.float32) ## real_filtered anchors
        
        # self.anchors_seed = np.array( [[ 32, 20], [ 61, 42], [ 59, 97],
        #                                 [103, 66], [122, 114], [183, 96],
        #                                 [160, 152], [211, 201], [343, 205]], dtype=np.float32) ## real_filtered plus all_sites_seatbelt anchors

        # self.anchors_seed = np.array( [[ 20, 16], [ 52, 24], [ 33, 58],
        #                                 [92, 44], [76, 110], [146, 76],
        #                                 [231, 109], [163, 187], [377, 170]], dtype=np.float32) ## real+synthv2+synthv2_SB>h8_w8

        self.anchors = generate_anchors(self.grid_size, self.resized_image_size, self.anchors_seed)
        self.anchors_per_grid = self.anchors_seed.shape[0]
        self.num_anchors = self.anchors.shape[0]

        self.results_dir = os.path.join(cfg.save_dir, 'results')

    def get_sample_ids(self, data_dict):
        
        sample_set_name = 'train' if self.phase == 'train' \
            else 'val' if self.phase == 'val' \
            else 'train' if self.phase == 'trainval' \
            else None
        path = data_dict[sample_set_name]
        # sample_ids_path = os.path.join(self.data_dir, 'image_sets', sample_set_name)
        # with open(sample_ids_path, 'r') as fp:
        #     sample_ids = fp.readlines()
        # sample_ids = tuple(x.strip() for x in sample_ids)
        image_paths = self.get_image_paths(path)
        return image_paths

    def load_image(self, index):
        image_path = self.sample_ids[index]
        image_id = image_path.split('/')[-1].split('.')[0]
        site_name = image_path.split('/')[-4]
        image_id = '_'.join([site_name, image_id])
        image = default_loader(image_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        image = np.array(image).astype(np.float32)
        # image = skimage.io.imread(image_path).astype(np.float32)
        return image, image_id, image_path

    def load_annotations(self, index):
        ann_path = self.sample_ids[index].split('.')[0] + '.txt'
        with open(ann_path, 'r') as fp:
            annotations = fp.readlines()

        annotations = [ann.strip().split(' ') for ann in annotations]
        class_ids, boxes = [], []
        for ann in annotations:
            # orig_class = self.all_class_names[int(ann[0])]
            mapped_class = self.map_class_names[int(ann[0])]
            if mapped_class not in self.class_names:
                continue
            class_id = self.class_names.index(mapped_class)
            box = [float(x) for x in ann[1:5]]
            if (box[2]*448 > self.cfg.object_size_thresh[1]) and (box[3]*256 > self.cfg.object_size_thresh[0]):
                boxes.append(box)
                class_ids.append(class_id)

        class_ids = np.array(class_ids, dtype=np.int16)
        boxes = np.array(boxes, dtype=np.float32)
        if len(boxes):
            return class_ids, boxes
        boxes = None
        return class_ids, boxes

    def get_image_paths(self, path):
        f = []  # image files
        for p, no_samples in path.items():
            p = os.path.join(self.data_root,p)
            p = Path(p)  # os-agnostic
            if p.is_file:
                with open(p) as t:
                    t = t.read().strip().splitlines()
                    if no_samples != -1:
                        t = random.sample(t, no_samples)
                    parent = str(p.parent.parent) + os.sep
                    f += [os.path.join(parent, x) for x in t]  # local to global path
            else:
                raise Exception(f'{p} does not exist')
        img_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
        return img_files