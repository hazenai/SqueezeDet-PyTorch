import os
from operator import itemgetter 

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import nms

import cv2


def whiten(image, mean=0.0, std=1.0):
    """
    :param image:
    :param mean: float or np.ndarray(1, 1, 3)
    :param std: float or np.ndarray(1, 1, 3)
    :return:
    """
    image = (image - mean) / std
    return image


def pad(image, padding):
    """
    :param image:
    :param padding: (top_pad, bottom_pad, left_pad, right_pad)
    :return:
    """
    if not np.all(padding == 0):
        padding = (padding[:2], padding[2:], [0, 0])
        image = np.pad(image, padding, mode="constant")

    return image


def crop(image, crops):
    """
    :param image:
    :param crops: (top_crop, bottom_crop, left_crop, right_crop)
    :return:
    """
    if not np.all(crops == 0):
        image = (
            image[crops[0] : -crops[1], :, :]
            if crops[1] > 0
            else image[crops[0] :, :, :]
        )
        image = (
            image[:, crops[2] : -crops[3], :]
            if crops[3] > 0
            else image[:, crops[2] :, :]
        )

    return image


def crop_or_pad(image, target_size):
    """
    :param image:
    :param image_meta:
    :param target_size: (height, width)
    :return:
    """
    padding, crops = np.zeros(4, dtype=np.int16), np.zeros(
        4, dtype=np.int16
    )  # (top, bottom, left, right) format

    height, width = image.shape[:2]
    target_height, target_width = target_size

    if height < target_height:
        padding[0] = (target_height - height) // 2
        padding[1] = (target_height - height) - padding[0]
    elif height > target_height:
        crops[0] = (height - target_height) // 2
        crops[1] = (height - target_height) - crops[0]

    if width < target_width:
        padding[2] = (target_width - width) // 2
        padding[3] = (target_width - width) - padding[2]
    elif width > target_width:
        crops[2] = (width - target_width) // 2
        crops[3] = (width - target_width) - crops[2]

    image = pad(image, padding)
    image = crop(image, crops)

    return image, padding, crops


def resize(image, target_size):
    height, width = image.shape[:2]
    scales = np.array(
        [target_size[0] / height, target_size[1] / width], dtype=np.float32
    )
    image = cv2.resize(image, (target_size[1], target_size[0]))

    return image, scales


def filter(det, objectness_thresh=0.5, nms_thresh=0.2, keep_top_k=1024):

    orders = torch.argsort(det[:, 4], descending=True)[:keep_top_k]
    class_ids = det[orders, 5]
    scores = det[orders, 6]
    objectness = det[orders, 4]
    boxes = det[orders, :4]
    objectness = det[orders, 4]

    # class-agnostic nms
    keeps = nms(boxes, objectness, nms_thresh)

    filtered_class_ids = class_ids[keeps]
    filtered_objectness = objectness[keeps]
    filtered_boxes = boxes[keeps, :]
    filtered_scores = scores[keeps]

    keeps = torch.logical_and(
        filtered_objectness > objectness_thresh, filtered_scores > objectness_thresh
    )

    if torch.sum(keeps) == 0:
        det = None
    else:
        det = torch.zeros(torch.sum(keeps), 7)
        det[:, :4] = filtered_boxes[keeps, :]
        det[:, 4] = filtered_objectness[keeps]
        det[:, 5] = filtered_class_ids[keeps]
        det[:, 6] = filtered_scores[keeps]
    return det


def boxes_postprocess(boxes, scales=[1, 1], padding=[0, 0, 0, 0], crops=[0, 0, 0, 0]):
    """
    remap processed boxes back into original image coordinates
    :param boxes: xyxy format
    :return:
    """
    # Adjust for scales
    boxes[:, [0, 2]] /= scales[1]
    boxes[:, [1, 3]] /= scales[0]

    # Adjust for padding
    boxes[:, [0, 2]] -= padding[2]
    boxes[:, [1, 3]] -= padding[0]

    # Adjust for cropping
    boxes[:, [0, 2]] += crops[2]
    boxes[:, [1, 3]] += crops[0]

    return boxes


class_colors = (
    (
        255.0
        * np.array(
            [
                0.850,
                0.325,
                0.098,
                0.466,
                0.674,
                0.188,
                0.098,
                0.325,
                0.850,
                0.301,
                0.745,
                0.933,
                0.635,
                0.078,
                0.184,
                0.300,
                0.300,
                0.300,
                0.600,
                0.600,
                0.600,
                1.000,
                0.000,
                0.000,
                1.000,
                0.500,
                0.000,
                0.749,
                0.749,
                0.000,
                0.000,
                1.000,
                0.000,
                0.000,
                0.000,
                1.000,
                0.667,
                0.000,
                1.000,
                0.333,
                0.333,
                0.000,
                0.333,
                0.667,
                0.000,
                0.333,
                1.000,
                0.000,
                0.667,
                0.333,
                0.000,
                0.667,
                0.667,
                0.000,
                0.667,
                1.000,
                0.000,
                1.000,
                0.333,
                0.000,
                1.000,
                0.667,
                0.000,
                1.000,
                1.000,
                0.000,
                0.000,
                0.333,
                0.500,
                0.000,
                0.667,
                0.500,
                0.000,
                1.000,
                0.500,
            ]
        )
    )
    .astype(np.uint8)
    .reshape((-1, 3))
)


def lp_visualize_boxes(image, det, lps, class_names=None, save_path=None, show=False):
    boxes = det[:, :4]
    bbox_scores = det[:, 4]
    ocr_score = det[:, 6]

    image = image.astype(np.uint8)
    if boxes is not None:
        num_boxes = boxes.shape[0]
        for i in range(num_boxes):
            class_id = 1
            bbox = boxes[i].astype(np.int32).tolist()
            image = cv2.rectangle(
                image,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                class_colors[class_id].tolist(),
                2,
            )

            class_name = (
                lps[i]
                if lps is not None
                else "class_{}".format(class_id)
            )
            text = (
                "{} {:.2f} {:.2f}".format(class_name, bbox_scores[i], ocr_score[i])
                if bbox_scores is not None
                else class_name
            )
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            text_size = cv2.getTextSize(text, font, fontScale=fontScale, thickness=1)[0]
            image = cv2.rectangle(
                image,
                (bbox[0], bbox[1] - text_size[1] - 8),
                (bbox[0] + text_size[0] + 8, bbox[1]),
                class_colors[class_id].tolist(),
                -1,
            )
            image = cv2.putText(
                image,
                text,
                (bbox[0] + 4, bbox[1] - 4),
                font,
                fontScale=fontScale,
                color=(255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        if show:
            title = "{} (press any key to continue)".format(os.path.basename(save_path))
            cv2.imshow(title, image[:, :, ::-1])
            cv2.waitKey()
            cv2.destroyWindow(title)
        if not (save_path is None):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, image[:, :, ::-1])
        return image


def generate_anchors(grid_size, input_size, anchors_seed):
    """
    :param grid_size: (grid_height, grid_width), shape of the output of ConvDet layer
    :param input_size: (height, width), shape of input image
    :param anchors_seed: np.ndarray(N, 2), where N is #anchors per grid
    :return: np.ndarray(A, 4), in xyxy format, where A = N * grid_height * grid_width
    """
    assert anchors_seed.shape[1] == 2

    anchors_per_grid = anchors_seed.shape[0]
    grid_height, grid_width = grid_size

    anchors_shape = np.reshape(
        grid_width * grid_height * [anchors_seed],
        (grid_height, grid_width, anchors_per_grid, 2),
    )

    input_height, input_width = input_size
    anchors_center_x, anchors_center_y = np.meshgrid(
        input_width * (1 / (grid_width * 2) + np.linspace(0, 1, grid_width + 1)[:-1]),
        input_height
        * (1 / (grid_height * 2) + np.linspace(0, 1, grid_height + 1)[:-1]),
    )
    anchors_center = np.stack((anchors_center_x, anchors_center_y), axis=2)
    anchors_center = np.repeat(
        np.reshape(anchors_center, (grid_height, grid_width, 1, 2)),
        anchors_per_grid,
        axis=2,
    )
    anchors_xywh = np.concatenate((anchors_center, anchors_shape), axis=3)

    return np.reshape(anchors_xywh, (-1, 4))


def xywh_to_xyxy(boxes_xywh):
    # assert torch.all(boxes_xywh[..., [2, 3]] > 0)
    return torch.cat(
        [
            boxes_xywh[..., [0]] - 0.5 * (boxes_xywh[..., [2]]),
            boxes_xywh[..., [1]] - 0.5 * (boxes_xywh[..., [3]]),
            boxes_xywh[..., [0]] + 0.5 * (boxes_xywh[..., [2]]),
            boxes_xywh[..., [1]] + 0.5 * (boxes_xywh[..., [3]]),
        ],
        dim=-1,
    )


def deltas_to_boxes_tflite(deltas, anchors, input_size):
    """
    :param deltas: dxdydwdh format
    :param anchors: xywh format
    :param input_size: input image size in hw format
    :return: boxes in xyxy format
    """
    boxes_xywh = torch.cat(
        [
            anchors[..., [0]] + anchors[..., [2]] * deltas[..., [0]],
            anchors[..., [1]] + anchors[..., [3]] * deltas[..., [1]],
            anchors[..., [2]] * torch.exp(deltas[..., [2]]),
            anchors[..., [3]] * torch.exp(deltas[..., [3]]),
        ],
        dim=2,
    )

    boxes_xyxy = xywh_to_xyxy(boxes_xywh)
    return boxes_xyxy


class Fire(nn.Module):
    def __init__(self, inplanes, sqz_planes, exp1x1_planes, exp3x3_planes):
        super(Fire, self).__init__()
        self.squeeze = nn.Conv2d(inplanes, sqz_planes, kernel_size=1)
        self.expand1x1 = nn.Conv2d(sqz_planes, exp1x1_planes, kernel_size=1)
        self.expand3x3 = nn.Conv2d(sqz_planes, exp3x3_planes, kernel_size=3, padding=1)
        self.act_1 = nn.ReLU(inplace=True)
        self.act_2 = nn.ReLU(inplace=True)
        self.act_3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act_1(self.squeeze(x))
        x = torch.cat(
            [
                self.act_2(self.expand1x1(x)),
                self.act_3(self.expand3x3(x)),
            ],
            dim=1,
        )
        return x


class SqueezeDet(nn.Module):
    def __init__(self, num_classes, anchors_per_grid, num_anchors):
        super(SqueezeDet, self).__init__()
        self.num_classes = num_classes
        self.anchors_per_grid = anchors_per_grid
        self.num_anchors = num_anchors
        self.out_channels = 512

        # Network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.f1 = Fire(64, 16, 64, 64)
        self.f2 = Fire(128, 16, 64, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.f3 = Fire(128, 32, 128, 128)
        self.f4 = Fire(256, 32, 128, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.f5 = Fire(256, 48, 192, 192)
        self.f6 = Fire(384, 48, 192, 192)
        self.f7 = Fire(384, 64, 256, 256)
        self.f8 = Fire(512, 64, 256, 256)
        self.convdet = nn.Conv2d(
            self.out_channels,
            self.anchors_per_grid * (self.num_classes + 5),
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.f1(x)
        x = self.f2(x)
        x = self.pool2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.pool3(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        x = self.f8(x)
        x = self.convdet(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(-1, self.num_anchors, self.num_classes + 5)
        return x

    def load_weights(self, filepath):
        chkpt = torch.load(filepath, map_location=torch.device("cpu"))
        state_dict = chkpt["state_dict"]
        dt = {}
        for k, v in state_dict.items():
            k = k.replace("base.features.0", "conv1")
            # k = k.replace("base.features.3", "base.f1")
            # k = k.replace("base.features.4", "base.f2")
            # k = k.replace("base.features.6", "base.f3")
            # k = k.replace("base.features.7", "base.f4")
            # k = k.replace("base.features.9", "base.f5")
            # k = k.replace("base.features.10", "base.f6")
            # k = k.replace("base.features.11", "base.f7")
            # k = k.replace("base.features.12", "base.f8")
            # k = k.replace("base.convdet", "base.convdet")
            k = k.replace("base.features.3", "f1")
            k = k.replace("base.features.4", "f2")
            k = k.replace("base.features.6", "f3")
            k = k.replace("base.features.7", "f4")
            k = k.replace("base.features.9", "f5")
            k = k.replace("base.features.10", "f6")
            k = k.replace("base.features.11", "f7")
            k = k.replace("base.features.12", "f8")
            k = k.replace("base.convdet", "convdet")
            
            dt[k] = v

        self.load_state_dict(dt, strict=True)
        return self


class PredictionResolverSingleClass(nn.Module):
    def __init__(self, input_size, num_classes, anchors, anchors_per_grid):
        super(PredictionResolverSingleClass, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.anchors = anchors
        self.anchors_per_grid = anchors_per_grid
        self.num_anchors = self.anchors.size()[0]

    def forward(self, x):
        pred_class_probs = F.softmax(x[..., : self.num_classes].contiguous(), dim=-1)

        pred_scores = torch.sigmoid(
            x[..., self.num_classes : self.num_classes + 1].contiguous()
        )

        pred_deltas = x[..., self.num_classes + 1 :].contiguous()
        pred_boxes = deltas_to_boxes_tflite(
            pred_deltas, self.anchors.to(pred_deltas.device), input_size=self.input_size
        )

        boxes = torch.zeros(x.size()[0], self.num_anchors, 6 + self.num_classes)
        boxes[..., :4] = pred_boxes
        boxes[..., 4] = pred_scores.squeeze(-1)
        boxes[..., 5] = torch.argmax(pred_class_probs, dim=2).int()
        boxes[..., 6] = torch.max(pred_class_probs, 2).values

        return boxes


####################################################################################


class ChannelsToLinear(nn.Linear):
    """Flatten a Variable to 2d and apply Linear layer"""

    def forward(self, x):
        b = x.size(0)
        return super().forward(x.view(b, -1))


class OCREncoder(nn.Module):
    def __init__(self, input_size=(32, 128), z_dim=64, device='cpu'):
        super(OCREncoder, self).__init__()
        self.device = device
        self.n_filters = 64
        self.z_dim = z_dim
        self.input_size = input_size
        self.conv1 = nn.Conv2d(1, self.n_filters, 5, (1, 2), 2)
        self.m1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(self.n_filters, self.n_filters * 2, 5, 1, 2)
        self.m2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(self.n_filters * 2, self.n_filters * 4, 5, 1, 2)
        self.m3 = nn.MaxPool2d(2, stride=2)
        self.conv4 = nn.Conv2d(self.n_filters * 4, self.n_filters * 8, 5, 1, 2)
        self.toLinear1 = ChannelsToLinear(self.n_filters * 8 * 4 * 8, 512)
        self.fc1 = nn.Linear(512, self.z_dim)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        x = self.m1(self.lrelu(self.conv1(x)))
        x = self.m2(self.lrelu(self.conv2(x)))
        x = self.m3(self.lrelu(self.conv3(x)))
        x = self.lrelu(self.conv4(x))
        x = self.lrelu(self.toLinear1(x))
        x = self.fc1(x)
        return x

    def load_weights(self, filepath):
        self.load_state_dict(
            torch.load(filepath, map_location=torch.device("cpu"))["encodermodel"],
            strict=True,
        )
        return self

    def preprocess(self, im):
        if len(im.shape) > 2:
            im  = im[:, :, 0]
        x, scales = resize(im, self.input_size)
        x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor)
        x = x / 255.
        
        return im, x, scales


def linear_convert(x):
    b = x.size(0)
    return x.view(b, 7, 35)


class OCRClassifier(nn.Module):
    def __init__(self, z_dim=64, device='cpu'):
        super(OCRClassifier, self).__init__()
        self.device = device
        self.z_dim = z_dim
        self.fc1 = nn.Linear(self.z_dim, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 245)

        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.labels_list = [
            "*",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z"
        ]

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        x = linear_convert(x)
        return x

    def load_weights(self, filepath):
        self.load_state_dict(
            torch.load(filepath, map_location=torch.device("cpu"))["classifier1model"],
            strict=True,
        )
        return self

    def postprocess(self, x):
        if self.device == 'cuda':
            x = x.cpu()
        pred = torch.argmax(x, dim=2)
        regid = ''.join(itemgetter(*pred.numpy()[0])(self.labels_list))
        score = torch.min(torch.max(x, dim=2).values, 1).values.detach().numpy()
        # score = x.detach().numpy()
        return regid, score


class OCR(nn.Module):
    def __init__(self, input_size=(32, 128), device='cpu'):
        super(OCR, self).__init__()
        self.device = device
        self.enc = OCREncoder(input_size, z_dim=64, device=device).load_weights("./utils/LPD/weights/autoencoder_uk_ver2_231021.pt")
        self.cls = OCRClassifier(z_dim=64, device=device).load_weights("./utils/LPD/weights/classifier_uk_ver2_231021.pt") # Vehicles, LPs, 7 positions, 37 possibilities 

    def forward(self, x):
        _, x, _ = self.enc.preprocess(x)
        x = x.to(torch.device(self.device))
        x = self.enc(x)
        x = self.cls(x)
        lp, score = self.cls.postprocess(x)
        return lp, score


####################################################################################
class ObjectDetector(nn.Module):
    def __init__(
        self,
        input_size=(256, 256),
        objectness_thresh=0.5,
        nms_thresh=0.2,
        keep_top_k=1024,
        device='cpu',
        model_path=None
    ):
        super(ObjectDetector, self).__init__()
        self.model_path=model_path
        self.device = device
        self.stride = 16  # SqueezeDet end to end stride. This is fixed for given arch.
        self.input_size = input_size  # (height, width)
        self.num_classes = 1
        self.anchors_per_grid = 6
        self.objectness_thresh = objectness_thresh
        self.nms_thresh = nms_thresh
        self.keep_top_k = keep_top_k
        self.grid_size = tuple(x // self.stride for x in self.input_size)
        self.anchors_seed = np.array(
            [[6, 5], [12, 10], [18, 10], [18, 18], [20, 24], [30, 15]],
            dtype=np.float32,
        )  # ALPR Detector Anchor boxes

        # self.anchors_seed = np.array(
        #     [[3, 2], [6, 5], [9, 5], [9, 9], [10, 12], [15, 8]],
        #     dtype=np.float32,
        # )  # ALPR Detector Anchor boxes
        self.anchors = generate_anchors(
            self.grid_size, self.input_size, self.anchors_seed
        )
        self.anchors = torch.from_numpy(self.anchors)
        self.num_anchors = self.grid_size[0] * self.grid_size[1] * self.anchors_per_grid

        # Preprocessing params - will remain fixed for current weight file.
        # ALPR Detector mean and std
        # self.rgb_mean = np.array(
        #     [76.7466, 76.11476, 76.241066], dtype=np.float32
        # ).reshape(1, 1, 3)
        # self.rgb_std = np.array(
        #     [45.27897, 44.897312, 42.433716], dtype=np.float32
        # ).reshape(1, 1, 3)
        self.rgb_mean = np.array([0.0, 0.0, 0.0], dtype=np.float32).reshape(1, 1, 3)
        self.rgb_std = np.array([40.0, 40.0, 40.0], dtype=np.float32).reshape(1, 1, 3)

        # Setup network
        
        self.net = SqueezeDet(
            self.num_classes, self.anchors_per_grid, self.num_anchors
        ).load_weights(self.model_path)
        self.pr = PredictionResolverSingleClass(
            self.input_size, self.num_classes, self.anchors, self.anchors_per_grid
        )

    def preprocess(self, im):
        x = whiten(im, self.rgb_mean, self.rgb_std)
        # x, padding, crops = crop_or_pad(x, self.input_size)
        x, scales = resize(x, self.input_size)
        x = (x * 2) - 1
        x = x.transpose(2, 0, 1)
        x = torch.from_numpy(x).unsqueeze(0)
        return im, x, scales

    def postprocess(self, boxes, scales):
        boxes = filter(boxes, self.objectness_thresh, self.nms_thresh, self.keep_top_k)
        if boxes is None:
            return None
        boxes = boxes_postprocess(boxes, scales)
        boxes = boxes.detach().numpy()
        return boxes

    def forward(self, im):
        im, x, scales = self.preprocess(im)
        x = x.to(torch.device(self.device))
        pred = self.net(x)
        boxes = self.pr(
            pred
        )  # num_anchors x [top, left, bottom, right, objectness, class_id, class_scores...]
        
        boxes = boxes.squeeze(0)
        boxes = self.postprocess(boxes, scales)
        return im, boxes


# od = ObjectDetector(input_size=(256, 256))
# basepath = './utils/images/T02'
# basepath = os.path.realpath(basepath)
# outdir = os.path.join(basepath, 'detections')
# if not os.path.isdir(outdir):
#     os.mkdir(outdir)

# from glob import glob
# imlist = sorted(glob(basepath + '/image-set/*.pgm'))

# for impath in imlist:
#     savepath = os.path.join(outdir, os.path.splitext(os.path.basename(impath))[0] + '.jpg')
#     print(savepath)
#     im, boxes = od(impath)
#     if not (boxes is None):
#         visualize_boxes(im, boxes, ('LP'), savepath, show=False)
