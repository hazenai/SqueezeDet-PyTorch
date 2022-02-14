import tqdm
import numpy as np
import torch.utils.data
from scipy.cluster.vq import kmeans2

from utils.config import Config
from utils.misc import load_dataset
from utils.image import resize


class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super(DatasetWrapper, self).__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        # image, image_id, image_path = self.dataset.load_image(index)
        gt_class_ids, gt_boxes = self.dataset.load_annotations(index)
        h,w = 256, 448
        if gt_boxes is not None:
            # Denormalze
            gt_boxes[:, 0] = gt_boxes[:, 0] * w
            gt_boxes[:, 1] = gt_boxes[:, 1] * h
            gt_boxes[:, 2] = gt_boxes[:, 2] * w
            gt_boxes[:, 3] = gt_boxes[:, 3] * h
            #xywh to xyxy
            gt_boxes = xywh_to_xyxy(gt_boxes)

        if gt_boxes is None:
            gt_boxes = np.empty([0, 4]).astype(np.float32)
        return gt_boxes

    def __len__(self):
        return len(self.dataset)


def compute_dataset_anchors_seed(dataset, anchors_per_grid=9,
                                 max_num_samples=20000, num_workers=4):
    """
    :param dataset: instance of torch.utils.data.Dataset class
    :param anchors_per_grid: number of anchors at each grid
    :param max_num_samples: randomly select N samples instead of whole dataset
    :param num_workers:
    """
    max_num_samples = min(max_num_samples, len(dataset))
    dataset.sample_ids = np.random.permutation(dataset.sample_ids)[:max_num_samples]
    dataloader = torch.utils.data.DataLoader(DatasetWrapper(dataset),
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True, 
                                             shuffle=True)

    dataset_boxes = []
    for boxes in tqdm.tqdm(dataloader):
        dataset_boxes.append(boxes)

    dataset_boxes = torch.cat(dataset_boxes, dim=1).squeeze(0).cpu().numpy()  # xyxy format
    box_shapes = dataset_boxes[:, [2, 3]] - dataset_boxes[:, [0, 1]]
    anchors_seed = kmeans2(box_shapes, anchors_per_grid, minit='++', iter=25)[0]

    anchors_seed = anchors_seed[np.argsort(anchors_seed[:, 0] * anchors_seed[:, 1]), :]

    return anchors_seed.astype(np.int32)


def xywh_to_xyxy(boxes_xywh):
    assert np.ndim(boxes_xywh) == 2
    # assert np.all(boxes_xywh > 0)

    return np.concatenate([
        boxes_xywh[:, [0]] - 0.5 * (boxes_xywh[:, [2]]),
        boxes_xywh[:, [1]] - 0.5 * (boxes_xywh[:, [3]]),
        boxes_xywh[:, [0]] + 0.5 * (boxes_xywh[:, [2]]),
        boxes_xywh[:, [1]] + 0.5 * (boxes_xywh[:, [3]])
    ], axis=1)


def main():
    cfg = Config().parse('eval --dataset yolo'.split(' '))
    dataset = load_dataset(cfg.dataset)('trainval', cfg)

    anchors_seed = compute_dataset_anchors_seed(dataset)

    print('Dataset\'s anchors seed: ', anchors_seed)


if __name__ == '__main__':
    main()

