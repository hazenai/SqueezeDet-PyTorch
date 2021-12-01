import os
import subprocess

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
import skimage.io

from datasets.base import BaseDataset
from utils.boxes import generate_anchors
import torch
import torchvision
from torchvision.datasets.folder import default_loader


class LPR(BaseDataset):
    def __init__(self, phase, cfg):
        super(LPR, self).__init__(phase, cfg)

        #self.input_size = (512, 512)  # (height, width), both dividable by 16
        # self.input_size = (256, 256)
        self.input_size = (128, 128)

        self.class_names = ('0')
        self.rgb_mean = np.array([97.631615, 98.70732, 98.41285], dtype=np.float32).reshape(1, 1, 3)
        self.rgb_std = np.array([52.766678, 52.63513, 52.348827], dtype=np.float32).reshape(1, 1, 3)

        self.num_classes = len(self.class_names)
        self.class_ids_dict = {cls_name: cls_id for cls_id, cls_name in enumerate(self.class_names)}

        self.data_dir = os.path.join(cfg.data_dir, 'lpr_crop/merged_data')
        self.sample_ids, self.sample_set_path = self.get_sample_ids()

        self.grid_size = tuple(x // 16 for x in self.input_size)  # anchors grid
        #self.grid_size = tuple(x // 8 for x in self.input_size)
        #self.grid_size = tuple(x // 32 for x in self.input_size)

        #self.anchors_seed = np.array([[12, 10], [22, 19], [35, 20], [36, 35], [39, 47], [60, 31]], dtype=np.float32)
        self.anchors_seed = np.array([[6, 5], [12, 10], [18, 10], [18, 18], [20, 24], [30, 15]], dtype=np.float32)
        #self.anchors_seed = np.array([[3, 3], [6, 5], [9, 5], [10, 9], [10, 12], [15, 8]], dtype=np.float32)

        #self.anchors_seed = np.array([[14, 11], [20, 13], [33, 11], [30, 6], [20, 22], [26, 44]], dtype=np.float32) # learned anchors
        
 
 


        # self.anchors_seed = np.array([[45, 22], [80, 40], [110, 42], [149, 45], [142, 82], [224, 72]], dtype=np.float32)

        self.anchors = generate_anchors(self.grid_size, self.input_size, self.anchors_seed)
        self.anchors_per_grid = self.anchors_seed.shape[0]
        self.num_anchors = self.anchors.shape[0]

        #torch.save(self.anchors, '/home/urwa/Documents/squeeze_det/anchors/anchors_256_exp_8.pt')
        
        

        self.results_dir = os.path.join(cfg.save_dir, 'results')

    def get_sample_ids(self):
        sample_set_name = 'train.txt' if self.phase == 'train' \
            else 'val.txt' if self.phase == 'val' \
            else 'trainval.txt' if self.phase == 'trainval' \
            else None

        sample_ids_path = os.path.join(self.data_dir, sample_set_name)
        with open(sample_ids_path, 'r') as fp:
            sample_ids = fp.readlines()
        sample_ids = tuple(x.strip() for x in sample_ids)

        return sample_ids, sample_ids_path

    def load_image(self, index):
        image_id = self.sample_ids[index]
        image_path = os.path.join(self.data_dir, 'images', image_id + '.png')
        #print(">>>>>",image_path)
        image = default_loader(image_path)
        # image = skimage.io.imread(image_path).astype(np.float32)
        return image, image_id

    def load_annotations(self, index):
        ann_id = self.sample_ids[index]
        ann_path = os.path.join(self.data_dir, 'labels', ann_id + '.txt')
        with open(ann_path, 'r') as fp:
            annotations = fp.readlines()

        annotations = [ann.strip().split(' ') for ann in annotations]
        class_ids, boxes = [], []
        for ann in annotations:
            if ann[0] not in self.class_names:
                continue
            if abs(float(ann[6]) - float(ann[4])) > 32 or abs(float(ann[7]) - float(ann[5])) > 16:
            	boxes.append([float(x) for x in ann[4:8]])
            	class_ids.append(self.class_ids_dict[ann[0]])

        class_ids = np.array(class_ids, dtype=np.int16)
        boxes = np.array(boxes, dtype=np.float32)
        if boxes.size == 0:
            boxes = None

        #print("loaded: ",boxes)
        return class_ids, boxes

    # ========================================
    #                evaluation
    # ========================================

    def save_results(self, results):
        #print("::::::::::::::: Save Results ::::::::::::::::::::")
        txt_dir = os.path.join(self.results_dir, 'data')
        os.makedirs(txt_dir, exist_ok=True)

        for res in results:
            txt_path = os.path.join(txt_dir, res['image_meta']['image_id'] + '.txt')
            if 'class_ids' not in res:
                with open(txt_path, 'w') as fp:
                    fp.write('')
                continue

            num_boxes = len(res['class_ids'])
            with open(txt_path, 'w') as fp:
                for i in range(num_boxes):
                    class_name = self.class_names[res['class_ids'][i]].lower()
                    score = res['scores'][i]
                    bbox = res['boxes'][i, :]
                    line = '{} -1 -1 0 {:.2f} {:.2f} {:.2f} {:.2f} 0 0 0 0 0 0 0 {:.3f}\n'.format(
                            class_name, *bbox, score)
                    fp.write(line)

    def evaluate(self):
        #print("::::::::::::::: EValuating ::::::::::::::::::::")
        kitti_eval_tool_path = os.path.join(self.cfg.root_dir, 'src/utils/kitti-eval/cpp/evaluate_object')
        cmd = '{} {} {} {} {}'.format(kitti_eval_tool_path,
                                      #os.path.join(self.data_dir, 'training'),
                                      self.data_dir,
                                      self.sample_set_path,
                                      self.results_dir,
                                      len(self.sample_ids))

        status = subprocess.call(cmd, shell=True)

        aps = {}
        for class_name in self.class_names:
            map_path = os.path.join(self.results_dir, 'stats_{}_ap.txt'.format(class_name.lower()))
            if os.path.exists(map_path):
                with open(map_path, 'r') as f:
                    lines = f.readlines()
                _aps = [float(line.split('=')[1].strip()) for line in lines]
            else:
                _aps = [0., 0., 0.]

            aps[class_name + '_easy'] = _aps[0]
            aps[class_name + '_moderate'] = _aps[1]
            aps[class_name + '_hard'] = _aps[2]

        aps['mAP'] = sum(aps.values()) / len(aps)

        return aps
