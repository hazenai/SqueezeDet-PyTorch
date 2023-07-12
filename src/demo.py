import os
import glob
import tqdm

import numpy as np
import skimage.io
import torch
import torch.utils.data

from datasets.kitti import KITTI

from datasets.yolo import YOLO
from engine.detector import Detector
from model.squeezedet import SqueezeDet, SqueezeDetWithLoss
from utils.config import Config
from utils.model import load_model
from PIL import Image
import cv2

def resize(image, image_meta, target_size, boxes=None):
    height, width = image.shape[:2]            
    scales = np.array([target_size[0] / height, target_size[1] / width], dtype=np.float32)
    image = cv2.resize(image, (target_size[1], target_size[0]))

    if boxes is not None:
        boxes[:, [0, 2]] *= scales[1]
        boxes[:, [1, 3]] *= scales[0]

    image_meta.update({'scales': scales})

    return image, image_meta, boxes

def whiten(image, image_meta, mean=0., std=1.):
    """
    :param image:
    :param image_meta:
    :param mean: float or np.ndarray(1, 1, 3)
    :param std: float or np.ndarray(1, 1, 3)
    :return:
    """
    image = (image - mean) / std
    image_meta.update({'rgb_mean': mean, 'rgb_std': std})
    return image, image_meta


def demo(cfg):
    input_size = (256, 256)  # (height, width), changed above with this one for alpr det
    rgb_mean = np.array([0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 1, 3)
    rgb_std = np.array([40.0, 40.0, 40.0], dtype=np.float32).reshape(1, 1, 3)
    # prepare configurations
    # cfg.load_model = '../exp/real_filtered_3class/model_best.pth'
    # cfg.load_model = '/workspace/SqueezeDet-PyTorch_simple_bypass/models/squeezedet_kitti_epoch280.pth'
    # cfg.load_model = '/workspace/SqueezeDet-PyTorch_simple_bypass/models/model_5040.pth'
    cfg.load_model = '/workspace/SqueezeDet-PyTorch_simple_bypass/models/alpr_det.pth'

    # cfg.load_model = '/workspace/SqueezeDet-PyTorch_simple_bypass/models/all_real_plus_synth_8sites_plus_SVsynth_plus_seatbelt_plus_new_trajectory_data_kitti_format_5percentofwidth_filtered_cont.pth'
    cfg.gpus = [0]  # -1 to use CPU
    cfg.debug = 2  # to visualize detection boxes
    dataset = YOLO('val', cfg)
    cfg = Config().update_dataset_info(cfg, dataset)

    # # preprocess image to match model's input resolution
    # preprocess_func = dataset.preprocess
    # del dataset

    # dataset = KITTI('val', cfg)
    # cfg = Config().update_dataset_info(cfg, dataset)

    # preprocess image to match model's input resolution
    preprocess_func = dataset.preprocess
    del dataset

    # prepare model & detector
    # model = SqueezeDet(cfg)
    model = SqueezeDetWithLoss(cfg, detectFlag=True)
    # model = load_model(model, cfg.load_model)
    model = load_model(model, cfg.load_model, cfg)
    detector = Detector(model.to(cfg.device), cfg)

    # prepare images
    # sample_images_dir = '/home/hazen/workspace/datasets/redspeed/image_2'
    # sample_image_paths = glob.glob(os.path.join(sample_images_dir, '*.jpg'))
    sample_images_dir = '/workspace/SqueezeDet-PyTorch/data/kitti/training/image_2'
    # sample_image_paths = glob.glob(os.path.join(sample_images_dir, '*.png'))
    # sample_images_dir = '/workspace/SqueezeDet-PyTorch_simple_bypass/data/Synthetic_LP_Det_Datasetdbe3d82f/images'
    sample_image_paths = glob.glob(os.path.join(sample_images_dir, '*.jpg'))

    base_images_dir_data = '/workspace/SqueezeDet-PyTorch/data/kitti/training/image_2'
    sample_ids_path='/workspace/SqueezeDet-PyTorch_simple_bypass/data/kitti/image_sets/val_oneimage.txt'
    

    with open(sample_ids_path, 'r') as fp:
        sample_ids = fp.readlines()
    sample_ids = tuple(x.strip() for x in sample_ids)
    # detection
    for img_name in tqdm.tqdm(sample_ids):
        path=os.path.join(base_images_dir_data, img_name+'.jpg')
        image = skimage.io.imread(path).astype(np.float32)
        
        # image = Image.open(path)
        # if image.mode == 'L':
        #     image = image.convert('RGB')
        # image = np.array(image).astype(np.float32)

        image_meta = {'image_id': os.path.basename(path)[:-4],
                      'orig_size': np.array(image.shape, dtype=np.int32)}

        # image, _ , image_meta, _, _= preprocess_func(image, image_meta)
        # # image, image_meta, _ = preprocess_func(image, image_meta)
        
        # whiten the image
        image, image_meta = whiten(image, image_meta, rgb_mean, rgb_std)
        # resize the image
        image, image_meta, boxes = resize(image, image_meta, input_size, boxes=None)
        image = (image * 2) - 1
        image = torch.from_numpy(image.transpose(2, 0, 1))
        image = image.unsqueeze(0).to(cfg.device)
        image_meta = {k: torch.from_numpy(v).unsqueeze(0).to(cfg.device) if isinstance(v, np.ndarray)
                      else [v] for k, v in image_meta.items()}

        inp = {'image': image,
               'image_meta': image_meta}
        
        results = detector.detect(inp)
        
        
        for res in results:
            num_boxes = len(res['class_ids'])
            for i in range(num_boxes):
                score = res['scores'][i]
                bbox = res['boxes'][i, :]
                line = '{} -1 -1 0 {:.2f} {:.2f} {:.2f} {:.2f} 0 0 0 0 0 0 0 {:.3f}\n'.format(
                'class_name', *bbox, score)
                print(line)
