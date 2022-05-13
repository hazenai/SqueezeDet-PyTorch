import os
import glob
import tqdm

import numpy as np
import skimage.io
import torch
import torch.utils.data

from datasets.yolo import YOLO
from engine.detector import Detector
from model.squeezedet import SqueezeDet, SqueezeDetWithLoss
from utils.config import Config
from utils.model import load_model
from PIL import Image


def demo(cfg):
    # prepare configurations
    cfg.load_model = '../exp/yolt_with_resnet_base_and_mobv2_cls_9sites_cont5/model_best.pth'
    cfg.gpus = [0]  # -1 to use CPU
    cfg.debug = 2  # to visualize detection boxes
    dataset = YOLO('val', cfg)
    cfg = Config().update_dataset_info(cfg, dataset)

    # preprocess image to match model's input resolution
    preprocess_func = dataset.preprocess
    del dataset

    # prepare model & detector
    model = SqueezeDetWithLoss(cfg)
    model = load_model(model, cfg.load_model, cfg)
    model.detect = True
    detector = Detector(model.to(cfg.device), cfg)

    # prepare images
    sample_images_dir = '/data/datasets/trajectory_training_data/yolo_format/delta_united_yolo/data/obj_train_data'
    sample_image_paths = glob.glob(os.path.join(sample_images_dir, '*.png'))

    # detection
    for path in tqdm.tqdm(sample_image_paths):
        # image = skimage.io.imread(path).astype(np.float32)
        image = Image.open(path)
        if image.mode == 'L':
            image = image.convert('RGB')
        image = np.array(image).astype(np.float32)

        image_meta = {'image_path': path,
                      'image_id': os.path.basename(path)[:-4],
                      'orig_size': np.array(image.shape, dtype=np.int32)}

        image, _ , image_meta, _, _= preprocess_func(image, image_meta)
        image = torch.from_numpy(image).unsqueeze(0).to(cfg.device)
        # print(image.shape)
        # raise()
        image_meta = {k: torch.from_numpy(v).unsqueeze(0).to(cfg.device) if isinstance(v, np.ndarray)
                      else [v] for k, v in image_meta.items()}

        inp = {'image': image,
               'image_meta': image_meta}

        _ = detector.detect(inp)
