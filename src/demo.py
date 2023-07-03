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


def demo(cfg):
    # prepare configurations
    # cfg.load_model = '../exp/real_filtered_3class/model_best.pth'
    # cfg.load_model = '/workspace/SqueezeDet-PyTorch_simple_bypass/models/squeezedet_kitti_epoch280.pth'
    cfg.load_model = '/workspace/SqueezeDet-PyTorch_simple_bypass/models/model_5040.pth'
    # cfg.load_model = '/workspace/SqueezeDet-PyTorch_simple_bypass/models/alpr_det.pth'

    # cfg.load_model='/workspace/SqueezeDet-PyTorch_simple_bypass/models/all_real_plus_synth_8sites_plus_SVsynth_plus_seatbelt_plus_new_trajectory_data_kitti_format_5percentofwidth_filtered_cont.pth'
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

    # detection
    for path in tqdm.tqdm(sample_image_paths):
        image = skimage.io.imread(path).astype(np.float32)
        # image = Image.open(path)
        # if image.mode == 'L':
        #     image = image.convert('RGB')
        # image = np.array(image).astype(np.float32)

        image_meta = {'image_id': os.path.basename(path)[:-4],
                      'orig_size': np.array(image.shape, dtype=np.int32)}

        image, _ , image_meta, _, _= preprocess_func(image, image_meta)
        # image, image_meta, _ = preprocess_func(image, image_meta)
        image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(cfg.device)
        image_meta = {k: torch.from_numpy(v).unsqueeze(0).to(cfg.device) if isinstance(v, np.ndarray)
                      else [v] for k, v in image_meta.items()}

        inp = {'image': image,
               'image_meta': image_meta}
        _ = detector.detect(inp)
