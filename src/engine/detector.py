import os
import time

import numpy as np
import torch
import torch.utils.data
from torchvision.ops import nms

from utils.image import image_postprocess
from utils.boxes import boxes_postprocess, visualize_boxes
from utils.misc import MetricLogger
from torchvision.datasets.folder import default_loader


class Detector(object):
    def __init__(self, model, cfg):
        self.model = model.to(cfg.device)
        self.model.eval()
        self.cfg = cfg
        # if self.cfg.dataset=='lpr':
        #     self.data_dir = os.path.join(self.cfg.data_dir, 'lpr_crop/merged_data')
        # elif self.cfg.dataset=='yolo':
        #     self.data_dir = os.path.join(self.cfg.data_dir, 'all_real_plus_synth_8sites_plus_SVsynth_plus_seatbelt_plus_new_trajectory_data_kitti_format_5percentofwidth_filtered')
    
    def detect(self, batch):
        dets = self.model(batch)

        results = []
        batch_size = dets['class_ids'].shape[0]
        for b in range(batch_size):
            image_meta = {k: v[b].cpu().numpy() if not isinstance(v, list) else v[b]
                          for k, v in batch['image_meta'].items()}

            det = {k: v[b] for k, v in dets.items()}
            det = self.filter(det)

            if det is None:
                results.append({'image_meta': image_meta})
                continue

            det = {k: v.cpu().detach().numpy() for k, v in det.items()}

            boxes = det['boxes']
            boxes[:,0] = (boxes[:,0]/self.cfg.resized_image_size[1])*self.cfg.input_size[1]
            boxes[:,1] = (boxes[:,1]/self.cfg.resized_image_size[0])*self.cfg.input_size[0]
            boxes[:,2] = (boxes[:,2]/self.cfg.resized_image_size[1])*self.cfg.input_size[1]
            boxes[:,3] = (boxes[:,3]/self.cfg.resized_image_size[0])*self.cfg.input_size[0]
            det['boxes'] = boxes
    
            det['boxes'] = boxes_postprocess(det['boxes'], image_meta)
            det['image_meta'] = image_meta
            results.append(det)
            if self.cfg.debug == 2:
                image_path = os.path.join(self.cfg.data_dir, 'training/image_2', image_meta['image_id'] + '.png')
                # image_path = os.path.join(self.data_dir, 'images' if self.cfg.dataset=='lpr' else 'training/image_2', image_meta['image_id'] + '.png'  if self.cfg.dataset=='lpr' else image_meta['image_id'] +'.jpg')
                image_visualize = load_image(image_path)
                save_path = os.path.join(self.cfg.debug_dir, image_meta['image_id'] + '.png')
                visualize_boxes(image_visualize, det['class_ids'], det['boxes'], det['scores'],
                                class_names=self.cfg.class_names,
                                save_path=save_path,
                                show=False) #self.cfg.mode == 'demo'

        return results

    def detect_dataset(self, dataset, cfg):
        start_time = time.time()

        data_loader = torch.utils.data.DataLoader(DataWrapper(dataset),
                                                  batch_size=self.cfg.batch_size,
                                                  num_workers=self.cfg.num_workers,
                                                  pin_memory=True)
        num_iters = len(data_loader)
        data_timer, net_timer = MetricLogger(), MetricLogger()
        end = time.time()

        results = []
        for iter_id, batch in enumerate(data_loader):
            for k in batch:
                if 'image_meta' not in k:
                    batch[k] = batch[k].to(device=self.cfg.device, non_blocking=True)
            data_timer.update(time.time() - end)
            end = time.time()
            results.extend(self.detect(batch))

            net_timer.update(time.time() - end)
            end = time.time()
            if iter_id % self.cfg.print_interval == 0:
                msg = 'eval: [{0}/{1}] | data {2:.3f}s | net {3:.3f}s'.format(
                    iter_id, num_iters, data_timer.val, net_timer.val)
                print(msg)
                with open(cfg.log_file, 'a+') as file:
                    file.write(msg + '\n')

        total_time = time.time() - start_time
        tpi = total_time / len(dataset)
        msg = 'Elapsed {:.2f}min ({:.1f}ms/image, {:.1f}frames/s)'.format(
            total_time / 60., tpi * 1000., 1 / tpi)
        print(msg)
        print('-' * 80)
        with open(cfg.log_file, 'a+') as file:
            file.write(msg + '\n')
            file.write('-' * 80 + '\n')

        return results

    def filter(self, det):
        # orders = torch.argsort(det['scores'], descending=True)[:self.cfg.keep_top_k]
        class_ids = det['class_ids']
        class_scores = det['class_scores']
        scores = det['scores']
        boxes = det['boxes']
        
        ## obj_score_threshold
        # print(scores)
        # & (class_ids != 0)
        keeps = (scores > self.cfg.score_thresh) & (class_scores > self.cfg.class_score_thresh)
        class_ids = class_ids[keeps]
        class_scores = class_scores[keeps]
        scores = scores[keeps]
        boxes = boxes[keeps]

        if torch.sum(keeps) == 0:
            det = None
        else:
            det = {'class_ids': class_ids,
                   'scores': scores,
                   'boxes': boxes}

        return det


class DataWrapper(torch.utils.data.Dataset):
    """ A wrapper of Dataset class that bypasses loading annotations """

    def __init__(self, dataset):
        super(DataWrapper, self).__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        image, image_id = self.dataset.load_image(index)
        image_meta = {'index': index,
                      'image_id': image_id,
                      'orig_size': np.array(image.shape, dtype=np.int32)}

        image, _, image_meta, gt_boxes, gt_class_ids = self.dataset.preprocess(image, image_meta)

        batch = {'image': image,
                 'image_meta': image_meta}
        return batch

    def __len__(self):
        return len(self.dataset)


def load_image(image_path):
    image = default_loader(image_path)
    if image.mode == 'L':
        image = image.convert('RGB')
    image = np.array(image).astype(np.float32)
    # image = skimage.io.imread(image_path).astype(np.float32)
    return image
