import os
import subprocess

import numpy as np
import skimage.io   
import pybboxes as pbx               

from datasets.base import BaseDataset
from utils.boxes import generate_anchors
from PIL import Image
from torchvision.datasets.folder import default_loader    

from utils.image import whiten, drift, flip, resize, crop_or_pad
from utils.boxes import compute_deltas, visualize_boxes
import torch




class YOLO(BaseDataset):
    def __init__(self, phase, cfg):
        super(YOLO, self).__init__(phase, cfg)

        # self.input_size = (256, 448)  # (height, width), both dividable by 16
        self.input_size = (256, 256)  # (height, width), changed above with this one for alpr det
        self.class_names = ['licenseplate']     # used for LPD                                                                          
        # self.class_names = ('cyclist', 'car', 'pedestrian')     # used for kitti                                                           
        #self.class_names = ('bike', 'car', 'bus')            
        # real_filtered mean and std
        # self.rgb_mean = np.array([94.87347, 96.89165, 94.70493], dtype=np.float32).reshape(1, 1, 3)
        # self.rgb_std = np.array([53.869507, 53.936283, 55.2807], dtype=np.float32).reshape(1, 1, 3)
        
        # real_filtered plus all_sites_seatbelt mean and std
        # self.rgb_mean = np.array([104.90631, 105.41336, 104.70162], dtype=np.float32).reshape(1, 1, 3)
        # self.rgb_std = np.array([50.69564, 49.60443, 50.158844], dtype=np.float32).reshape(1, 1, 3)
        self.rgb_mean = np.array([0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 1, 3)
        self.rgb_std = np.array([40.0, 40.0, 40.0], dtype=np.float32).reshape(1, 1, 3)

        self.num_classes = len(self.class_names)
        self.class_ids_dict = {cls_name: cls_id for cls_id, cls_name in enumerate(self.class_names)}

        self.data_dir = os.path.join(cfg.data_dir, 'kitti')
        # self.data_dir = os.path.join(cfg.data_dir, 'all_real_plus_synth_8sites_plus_SVsynth_plus_seatbelt_plus_new_trajectory_data_kitti_format_5percentofwidth_filtered')
        self.sample_ids, self.sample_set_path = self.get_sample_ids()

        self.grid_size = tuple(x //cfg.stride  for x in self.input_size)  # anchors grid 
        # self.anchors_seed = np.array([[ 10, 5], [5, 10], [6, 6],
        #                                 [25, 13], [60, 30], [90, 43], 
        #                                 [55, 15], [350, 180], [20, 43]], dtype=np.float32) ## Anchors used for LPD                    
        
        # self.anchors_seed = np.array( [[ 115, 95], [ 61, 42], [ 59, 97],                      # [ 32, 20] remove from first location [width, height]          
        #                                 [103, 66], [122, 114], [183, 96],                        # Anchors used for kitti training           
        #                                 [160, 152], [211, 201], [343, 205]], dtype=np.float32) ## real_filtered plus all_sites_seatbelt anchors

        self.anchors_seed = np.array(
            [[6, 5], [12, 10], [18, 10], [18, 18], [20, 24], [30, 15]],
            dtype=np.float32,
        )  # ALPR Detector Anchor boxes
        self.anchors = generate_anchors(self.grid_size, self.input_size, self.anchors_seed)
        self.anchors_per_grid = self.anchors_seed.shape[0]
        self.num_anchors = self.anchors.shape[0]

        self.results_dir = os.path.join(cfg.save_dir, 'results')

    def get_sample_ids(self):
        print('pase is: {}'.format(self.phase))
        if self.cfg.oneimage:
            sample_set_name = 'train_oneimage.txt' if self.phase == 'train' \
            else 'val_oneimage.txt' if self.phase == 'val' \
            else 'trainval.txt' if self.phase == 'trainval' \
            else None

        else:
            sample_set_name = 'train.txt' if self.phase == 'train' \
                else 'val.txt' if self.phase == 'val' \
                else 'trainval.txt' if self.phase == 'trainval' \
                else None

        sample_ids_path = os.path.join(self.data_dir, 'image_sets', sample_set_name)
        print(sample_ids_path)
        with open(sample_ids_path, 'r') as fp:
            sample_ids = fp.readlines()
        sample_ids = tuple(x.strip() for x in sample_ids)
        return sample_ids, sample_ids_path

    def load_image(self, index):
        image_id = self.sample_ids[index]
        # image_path = os.path.join(self.data_dir, 'training/image_2', image_id + '.png')  #Add line for kitti 
        image_path = os.path.join(self.data_dir, 'training/image_2', image_id + '.jpg')
        image = default_loader(image_path)
        if image.mode == 'L':
            image = image.convert('RGB')
        image = np.array(image).astype(np.float32)
        # image = skimage.io.imread(image_path).astype(np.float32)
        return image, image_id

    def load_annotations(self, index):
        ann_id = self.sample_ids[index]
        ann_path = os.path.join(self.data_dir, 'training/label_2', ann_id + '.txt')
        with open(ann_path, 'r') as fp:
            annotations = fp.readlines()

        annotations = [ann.strip().split(' ') for ann in annotations]
        class_ids, boxes = [], []
        for ann in annotations:
            if ann[0].lower() not in self.class_names:
            # if ann[0] not in self.class_names:
                continue
            class_ids.append(self.class_ids_dict[ann[0].lower()])
            # class_ids.append(self.class_ids_dict[ann[0]])
            box = [float(x) for x in ann[4:8]]
            # if box[2] <= 0:
            #     box[2] = 0.00001
            # if box[3] <= 0:
            #     box[3] = 0.00001
            boxes.append(box)

        class_ids = np.array(class_ids, dtype=np.int16)
        boxes = np.array(boxes, dtype=np.float32)
        if len(boxes):
            return class_ids, boxes
        boxes = None
        return class_ids, boxes
    
    def load_annotations_comma_and_Space_format(self, index):
        ann_id = self.sample_ids[index]
        ann_path = os.path.join(self.data_dir, 'training/label_2', ann_id + '.txt')
        with open(ann_path, 'r') as fp:
            annotations = fp.readlines()


        # ls = 
        # annotations = [an.strip() for an in [ann.split(',') for ann in annotations][0]]
        class_ids, boxes, res = [], [], []
        for an in [ann.split(',') for ann in annotations]:
            tempBox = [float(aa.strip()) for aa in an]
            # modified according to given gt_labels in the form of y1,x1,y2,x2 and converting them to x1,y1,x2,y2
            res.append([tempBox[1],tempBox[0], tempBox[3], tempBox[2]])
            class_ids.append(self.class_ids_dict['licenseplate'])
        annotations_processed = res.copy()
        boxes = annotations_processed.copy()

        # box = [float(x) for x in annotations[:]]
        # # true_boxes.append([name, 'licenseplate', box[0], box[1], box[2], box[3]])
        # for i in range(len(box)):
        #     class_ids.append(self.class_ids_dict['licenseplate'])
        #     
        # annotations = [ann.strip().split(' ') for ann in annotations]
        # class_ids, boxes = [], []
        # for ann in annotations:
        #     if ann[0].lower() not in self.class_names:
        #     # if ann[0] not in self.class_names:
        #         continue
        #     class_ids.append(self.class_ids_dict['licenseplate'])
        #     # class_ids.append(self.class_ids_dict[ann[0]])
        #     box = [float(x) for x in ann[4:8]]
        #     # if box[2] <= 0:
        #     #     box[2] = 0.00001
        #     # if box[3] <= 0:
        #     #     box[3] = 0.00001
        #     boxes.append(box)

        class_ids = np.array(class_ids, dtype=np.int16)
        boxes = np.array(boxes, dtype=np.float32)
        if len(boxes):
            return class_ids, boxes
        boxes = None
        return class_ids, boxes

    # ========================================
    #                preprocess yolo
    # ========================================
    def preprocess(self, image, image_meta, boxes=None, class_ids=None):
        # print('Preprocess from child of baseDataset: yolo is called')
        image, image_meta = whiten(image, image_meta, self.rgb_mean, self.rgb_std)
        # resize the image
        image, image_meta, boxes = resize(image, image_meta, self.input_size, boxes=boxes)
        image = (image * 2) - 1
        image = torch.from_numpy(image.transpose(2, 0, 1))
        image_visualize = image
        
        return image, image_visualize, image_meta, boxes, class_ids

        # return super().preprocess(image, image_meta, boxes, class_ids)

    # ========================================
    #                evaluation
    # ========================================

    def save_results(self, results):
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
        kitti_eval_tool_path = os.path.join(self.cfg.root_dir, 'src/utils/kitti-eval/cpp/evaluate_object')
        cmd = '{} {} {} {} {}'.format(kitti_eval_tool_path,
                                      os.path.join(self.data_dir, 'training'),
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
