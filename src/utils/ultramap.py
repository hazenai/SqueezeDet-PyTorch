"""
Description: This modules holds the implementation of mAP computation as per
in ultralytics repo, https://github.com/ultralytics/yolov5.
Note: This metric expects following entries in the data batch (last index).
    'batch_ultramap_input': list(tensor[numboxes, label, cfd, x1, y1, x2, y2])
    'batch_target': list(tensor[numboxes, label, x, y, w, h] normalized boxes)
Author: shahzaib@hazen.ai
"""
import copy
import os
import numpy as np
import torch
import logging
from logging import debug, info
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import matplotlib.pyplot as plt
plt.switch_backend('agg')


class UltraMap(object):
    def __init__(self, config):
        self.device = config['device']
        self.stats = list()
        self.batchcount = 0
        self.classnames = config['metric']['classnames']
        iou_info = config['metric']['iou_info']
        self.iouv = torch.linspace(
            iou_info[0], iou_info[1], iou_info[2]
        ).to(self.device)

    @staticmethod
    def clip_coords(boxes, img_shape):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        boxes[:, 0].clamp_(0, img_shape[1])  # x1
        boxes[:, 1].clamp_(0, img_shape[0])  # y1
        boxes[:, 2].clamp_(0, img_shape[1])  # x2
        boxes[:, 3].clamp_(0, img_shape[0])  # y2

    @staticmethod
    def xywh2xyxy(x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]
        #  where xy1=top-left, xy2=bottom-right
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor)\
             else np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    @staticmethod
    def box_iou(box1, box2):
        # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """

        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)

        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
                 torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
        return inter / (area1[:, None] + area2 - inter)

    @staticmethod
    def compute_ap(recall, precision):
        """ Compute the average precision, given the recall and precision curves.
        Source: https://github.com/rbgirshick/py-faster-rcnn.
        # Arguments
            recall:    The recall curve (list).
            precision: The precision curve (list).
        # Returns
            The average precision as computed in py-faster-rcnn.
        """

        # Append sentinel values to beginning and end
        mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
        mpre = np.concatenate(([0.], precision, [0.]))

        # Compute the precision envelope
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

        # Integrate area under curve
        method = 'interp'  # methods: 'continuous', 'interp'
        if method == 'interp':
            x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
            ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
        else:  # 'continuous'
            # points where x axis (recall) changes
            i = np.where(mrec[1:] != mrec[:-1])[0]
            # area under curve
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        return ap

    @staticmethod
    def ap_per_class(tp, conf, pred_cls, target_cls, fname):
        """ Compute the average precision, given the recall and precision curves.
        Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
        # Arguments
            tp:  True positives (nparray, nx1 or nx10).
            conf:  Objectness value from 0-1 (nparray).
            pred_cls:  Predicted object classes (nparray).
            target_cls:  True object classes (nparray).
            plot:  Plot precision-recall curve at mAP@0.5
            fname:  Plot filename
        # Returns
            The average precision as computed in py-faster-rcnn.
        """

        # Sort by objectness
        i = np.argsort(-conf)
        tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

        # Find unique classes
        unique_classes = np.unique(target_cls)

        # Create Precision-Recall curve and compute AP for each class
        px, py = np.linspace(0, 1, 1000), []  # for plotting
        # score to evaluate
        # P and R https://github.com/ultralytics/yolov3/issues/898
        pr_score = 0.1
        s = [unique_classes.shape[0], tp.shape[1]]
        # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
        ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)
        for ci, c in enumerate(unique_classes):
            i = pred_cls == c
            n_gt = (target_cls == c).sum()  # Number of ground truth objects
            n_p = i.sum()  # Number of predicted objects

            if n_p == 0 or n_gt == 0:
                continue
            else:
                # Accumulate FPs and TPs
                fpc = (1 - tp[i]).cumsum(0)
                tpc = tp[i].cumsum(0)

                # Recall
                recall = tpc / (n_gt + 1e-16)  # recall curve
                # r at pr_score, negative x, xp because xp decreases
                r[ci] = np.interp(-pr_score, -conf[i], recall[:, 0])

                # Precision
                precision = tpc / (tpc + fpc)  # precision curve
                # p at pr_score
                p[ci] = np.interp(-pr_score, -conf[i], precision[:, 0])

                # AP from recall-precision curve
                # precision at mAP@0.5
                py.append(np.interp(px, recall[:, 0], precision[:, 0]))
                for j in range(tp.shape[1]):
                    ap[ci, j] = UltraMap.compute_ap(recall[:, j],
                                                    precision[:, j])

        # Compute F1 score (harmonic mean of precision and recall)
        f1 = 2 * p * r / (p + r + 1e-16)

        py = np.stack(py, axis=1)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot(px, py, linewidth=0.5, color='grey')  # plot(recall, precision)
        ax.plot(px, py.mean(1), linewidth=2, color='blue', label='all classes')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.legend()
        fig.tight_layout()
        fig.savefig(fname, dpi=200)
        plt.close('all')

        return p, r, ap, f1, unique_classes.astype('int32')

    @staticmethod
    def getmetricpath(config):
        metricfolder = config['metric']['output_path']
        expid = config['exp_id']
        _ce = config.get('current_epoch', 0)
        if not os.path.exists(metricfolder):
            os.mkdir(metricfolder)
        if not os.path.exists(f"{metricfolder}/{expid}"):
            os.mkdir(f"{metricfolder}/{expid}")
        if not os.path.exists(f"{metricfolder}/{expid}/{_ce}"):
            os.mkdir(f"{metricfolder}/{expid}/{_ce}")
        return f"{metricfolder}/{expid}/{_ce}"

    @staticmethod
    def log_metrics(config, mAP50):
        _epoch = config.get('current_epoch', 0)
        _metrics = {'final': {'mAP @ 0.5': mAP50}}
        info({'epoch': _epoch, 'metrics': _metrics})

    def __call__(self, data, config):
        output = data[-1]['batch_ultramap_input']
        targets = data[-1]['batch_target']
        numbatches = config['dataset'][config['data_mode']]['numbatches']
        ap, ap_class = [], []
        classnameslist = self.classnames
        # Remove 'not_a_class' Class.
        if 'not_a_class' in classnameslist:
            classnameslist.remove('not_a_class')

        nc = len(classnameslist)
        niou = self.iouv.numel()
        for si, pred in enumerate(output):
            width = data[si]['infersize'][0]
            height = data[si]['infersize'][1]
            whwh = torch.Tensor([width, height, width, height]).to(self.device)
            labels = targets[si]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class

            if pred is None:
                if nl:
                    self.stats.append(
                        (torch.zeros(0, niou, dtype=torch.bool),
                         torch.Tensor(), torch.Tensor(), tcls)
                    )
                continue

            # Clip boxes to image bounds
            self.clip_coords(pred, (height, width))

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool,
                                  device=self.device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = self.xywh2xyxy(labels[:, 1:5]) * whwh

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    # prediction indices
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)
                    # target indices
                    pi = (cls == pred[:, 0]).nonzero(as_tuple=False).view(-1)

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        # best ious, indices
                        ious, i = self.box_iou(pred[pi, 2:], tbox[ti]).max(1)

                        # Append detections
                        detected_set = set()
                        for j in (ious > self.iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                # iou_thres is 1xn
                                correct[pi[j]] = ious[j] > self.iouv
                                # all targets already located in image
                                if len(detected) == nl:
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            self.stats.append((correct.cpu(), pred[:, 1].cpu(),
                               pred[:, 0].cpu(), tcls))
        self.batchcount += 1
        if self.batchcount >= numbatches:
            # to numpy
            statsc = [np.concatenate(x, axis=0) for x in zip(*self.stats)]
            p, r, f1, mp, mr, map50, _map = 0., 0., 0., 0., 0., 0., 0.
            if len(statsc) and statsc[0].any():
                metricpath = self.getmetricpath(config)
                p, r, ap, f1, ap_class = self.ap_per_class(
                    *statsc, fname=f"{metricpath}/precision-recall_curve.png"
                    )
                # [P, R, AP@0.5, AP@0.5:0.95]
                p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)
                mp, mr, map50, _map = p.mean(), r.mean(), \
                    ap50.mean(), ap.mean()
                # number of targets per class
                nt = np.bincount(statsc[3].astype(np.int64), minlength=nc)
            else:
                nt = torch.zeros(1)
            # Log results
            s = ('%25s' + '%15s' * 6) % (
                'Class', 'Batches', 'Targets',
                'P', 'R',
                f'mAP@{round(float(self.iouv[0]), 2)}',
                f'mAP@{round(float(self.iouv[0]), 2)}:'
                f'{round(float(self.iouv[-1]), 2)}'
                )
            debug(s)
            pf = '%25s' + '%15.3g' * 6
            debug(pf % ('all', self.batchcount, nt.sum(), mp, mr, map50, _map))
            # Log results per class
            if nc > 1 and len(statsc):
                for i, c in enumerate(ap_class):
                    debug(pf % (classnameslist[c], self.batchcount,
                                nt[c], p[i], r[i], ap50[i], ap[i]))
            self.log_metrics(config, map50)
            self.batchcount = 0
            self.stats = list()