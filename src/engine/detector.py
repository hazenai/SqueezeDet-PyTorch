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
from tqdm import tqdm
from utils.detector_utils import ConfusionMatrix, process_batch, save_one_txt, ap_per_class
from utils.detector_utils import LOGGER


class Detector(object):
    def __init__(self, model, cfg):
        self.model = model.to(cfg.device)
        self.model.eval()
        self.cfg = cfg

    def detect(self, batch):
        dets = self.model(batch)
        outs = []
        batch_size = dets['class_ids'].shape[0]
        for b in range(batch_size):
            image_meta = {k: v[b].cpu().numpy() if not isinstance(v, list) else v[b]
                          for k, v in batch['image_meta'].items()}

            det = {k: v[b] for k, v in dets.items()}
            det = self.filter(det)

            if det is None:
                outs.append(torch.empty(0, 6))
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
            if self.cfg.debug == 2:
                image_path = image_meta['image_path']
                image_visualize = load_image(image_path)
                save_path = os.path.join(self.cfg.debug_dir, image_meta['image_id'] + '.png')
                height, width = image_visualize.shape[:2]
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0., width - 1.)
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0., height - 1.)
                
                visualize_boxes(image_visualize, det['class_ids'], det['boxes'], det['scores'],
                                class_names=self.cfg.class_names,
                                save_path=save_path,
                                show=False) #self.cfg.mode == 'demo'
            out = np.concatenate((det['boxes'], np.expand_dims(det['scores'], axis=1), np.expand_dims(det['class_ids'], axis=1)), axis=1)
            out = torch.tensor(out, device = batch['image'].device)
            outs.append(out)
        return outs

    def detect_dataset(self, dataset, cfg):
        start_time = time.time()

        data_loader = torch.utils.data.DataLoader(DataWrapper(dataset, cfg),
                                                  batch_size=self.cfg.batch_size,
                                                  num_workers=self.cfg.num_workers,
                                                  pin_memory=True,
                                                  collate_fn=collate_fn)

        seen = 0
        stats, ap, ap_class = [], [], []

        single_cls = True if self.cfg.classagnostic_map else False
        nc = 1 if single_cls else self.cfg.num_classes # number of classes
        names = {i:k for i,k in enumerate(self.cfg.class_names)}
        save_dir = self.cfg.save_dir
        plots = self.cfg.plots
        save_txt=self.cfg.save_txt
        save_conf= self.cfg.save_conf
        verbose = True
        if save_txt:
            os.makedirs(os.path.join(save_dir,'labels'), exist_ok=True)
        confusion_matrix = ConfusionMatrix(nc=nc)
        iouv = torch.linspace(0.5, 0.95, 10).to(self.cfg.device)  # iou vector for mAP@0.5:0.95
        niou = iouv.numel()
        s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
        with open(cfg.log_file, 'a+') as file:
            file.write(s + '\n')
        pbar = tqdm(data_loader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        for iter_id, batch in enumerate(pbar):
            for k in batch:
                if 'image_meta' not in k:
                    batch[k] = batch[k].to(device=self.cfg.device, non_blocking=True)
            out = self.detect(batch)
            targets = batch['targets']
            # Metrics
            for si, pred in enumerate(out):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                if single_cls:
                    tcls = [0 for i in range(nl)]
                    labels[:, 0:1] = 0
                    pred[:, 5] = 0
                seen += 1

                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue

                # Predictions
                predn = pred.clone()
                # Evaluate
                if nl:
                    tbox = labels[:, 1:5]
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                    correct = process_batch(predn, labelsn, iouv)
                    if plots:
                        confusion_matrix.process_batch(predn, labelsn)
                else:
                    correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)

                # Save/log
                shape = batch['image_meta']['orig_size'][si][:2].tolist()
                image_id = batch['image_meta']['image_id'][si]
                if save_txt:
                    save_one_txt(predn, save_conf, shape, file=os.path.join(save_dir,'labels',(image_id + '.txt')))
        # Compute metrics
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
            pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
            info = pf % ('all', seen, nt.sum(), mp, mr, map50, map)
            with open(cfg.log_file, 'a+') as file:
                file.write(info + '\n')
            LOGGER.info(info)

        else:
            nt = torch.zeros(1)
        # Print results per class
        if (verbose or (nc < 50)) and nc > 1 and len(stats):
            for i, c in enumerate(ap_class):
                info = pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i])
                with open(cfg.log_file, 'a+') as file:
                    file.write(info + '\n')
                LOGGER.info(info)

        total_time = time.time() - start_time
        tpi = total_time / len(dataset)
        msg = 'Elapsed {:.2f}min ({:.1f}ms/image, {:.1f}frames/s)'.format(
            total_time / 60., tpi * 1000., 1 / tpi)
        print(msg)
        print('-' * 80)
        with open(cfg.log_file, 'a+') as file:
            file.write(msg + '\n')
            file.write('-' * 80 + '\n')
        if len(stats) and stats[0].any():
            return {'map@50':map50, 'map':map}
        else:
            return {'map@50':0, 'map':0}

    def filter(self, det):
        class_ids = det['class_ids']
        class_scores = det['class_scores']
        scores = det['scores']
        boxes = det['boxes']
       
        #class-wise nms
        filtered_class_ids, filtered_class_scores, filtered_scores, filtered_boxes = [], [], [], []
        for cls_id in range(self.cfg.num_classes):
            idx_cur_class = (class_ids == cls_id)
            if torch.sum(idx_cur_class) == 0:
                continue

            class_ids_cur_class = class_ids[idx_cur_class]
            class_scores_cur_class = class_scores[idx_cur_class]
            scores_cur_class = scores[idx_cur_class]
            boxes_cur_class = boxes[idx_cur_class, :]

            keeps = nms(boxes_cur_class, scores_cur_class, self.cfg.nms_thresh_test)

            filtered_class_ids.append(class_ids_cur_class[keeps])
            filtered_class_scores.append(class_scores_cur_class[keeps])
            filtered_scores.append(scores_cur_class[keeps])
            filtered_boxes.append(boxes_cur_class[keeps, :])

        filtered_class_ids = torch.cat(filtered_class_ids)
        filtered_class_scores = torch.cat(filtered_class_scores)
        filtered_scores = torch.cat(filtered_scores)
        filtered_boxes = torch.cat(filtered_boxes, dim=0)

        keeps = (filtered_scores > self.cfg.score_thresh) & (filtered_class_ids != 0) & (filtered_class_scores > self.cfg.class_score_thresh)

        if torch.sum(keeps) == 0:
            filtered_det = None
        else:
            filtered_det = {'class_ids': filtered_class_ids[keeps],
                   'class_scores': filtered_class_scores[keeps],
                   'scores': filtered_scores[keeps],
                   'boxes': filtered_boxes[keeps]}

        return filtered_det


class DataWrapper(torch.utils.data.Dataset):
    """ A wrapper of Dataset class that bypasses loading annotations """

    def __init__(self, dataset, cfg):
        super(DataWrapper, self).__init__()
        self.dataset = dataset 
        self.cfg = cfg

    def __getitem__(self, index):
        image, image_id, image_path= self.dataset.load_image(index)
        gt_class_ids, gt_boxes = self.dataset.load_annotations(index)
        if self.cfg.dataset=='yolo':
            h,w = image.shape[:2]
            if gt_boxes is not None:
                # Denormalze
                gt_boxes[:, 0] = gt_boxes[:, 0] * w
                gt_boxes[:, 1] = gt_boxes[:, 1] * h
                gt_boxes[:, 2] = gt_boxes[:, 2] * w
                gt_boxes[:, 3] = gt_boxes[:, 3] * h
                #xywh to xyxy
                gt_boxes = xywh_to_xyxy(gt_boxes)
        image_meta = {'index': index,
                      'image_id': image_id,
                      'image_path': image_path,
                      'orig_size': np.array(image.shape, dtype=np.int32)}

        image, _, image_meta, _, _ = self.dataset.preprocess(image, image_meta)
        if gt_boxes is not None:
            indx = np.zeros(gt_class_ids.shape)
            targets = np.concatenate((np.expand_dims(indx,axis=1), np.expand_dims(gt_class_ids, axis=1), gt_boxes), axis=1)
        else:
            targets = np.empty((0,6))
        batch = {'image': torch.tensor(image),
                 'image_meta': image_meta,
                 }
        return batch, torch.tensor(targets)

    def __len__(self):
        return len(self.dataset)


def load_image(image_path):
    image = default_loader(image_path)
    if image.mode == 'L':
        image = image.convert('RGB')
    image = np.array(image).astype(np.float32)
    # image = skimage.io.imread(image_path).astype(np.float32)
    return image


def collate_fn(batch):
        b, targets = zip(*batch)  # transposed
        for i, l in enumerate(targets):
            l[:, 0] = i  # add target image index for build_targets()
        
        images = torch.stack([b[i]['image'] for i in range(len(b))])
        image_meta = {}
        for i in range(len(b)):
            for k,v in b[i]['image_meta'].items():
                if k not in image_meta.keys():
                    image_meta[k] = [v]
                else:
                    image_meta[k].extend([v])

        targets = torch.cat(targets, 0)
        gt = {
            'image': images,
            'image_meta':image_meta,
            'targets': targets
        }
        return gt

def xywh_to_xyxy(boxes_xywh):
    assert np.ndim(boxes_xywh) == 2
    # assert np.all(boxes_xywh > 0)

    return np.concatenate([
        boxes_xywh[:, [0]] - 0.5 * (boxes_xywh[:, [2]]),
        boxes_xywh[:, [1]] - 0.5 * (boxes_xywh[:, [3]]),
        boxes_xywh[:, [0]] + 0.5 * (boxes_xywh[:, [2]]),
        boxes_xywh[:, [1]] + 0.5 * (boxes_xywh[:, [3]])
    ], axis=1)