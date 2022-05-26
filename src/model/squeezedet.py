import torch
import torch.nn as nn
from model.mobilenetv2 import _make_divisible, ConvBNReLU, InvertedResidual, MobileNetV2
from model.resnet import Bottleneck, BasicBlock, conv1x1, conv3x3
from model.modules import deltas_to_boxes, deltas_to_boxes_tflite, compute_overlaps, safe_softmax
import torch.nn.functional as F
from torchvision.ops import nms
import torch.nn.functional as F
from utils.roialign import RoIAlign
from utils.selecttrainingsamples import SelectTrainingSamples
from torchvision.ops import boxes as box_ops
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
import random
import gc

EPSILON = 1E-10


class Fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes, qat):
        super(Fire, self).__init__()
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.activation_1 = nn.ReLU(inplace=True)
        self.activation_2 = nn.ReLU(inplace=True)
        self.activation_3 = nn.ReLU(inplace=True)
        self.qat = qat
        self.float_functional_simple = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        x = self.activation_1(self.squeeze(x))
        if self.qat:
            x = self.float_functional_simple.cat([
                self.activation_2(self.expand1x1(x)),
                self.activation_3(self.expand3x3(x))
            ], dim=1)
        else:
            x = torch.cat([
                self.activation_2(self.expand1x1(x)),
                self.activation_3(self.expand3x3(x))
            ], dim=1)
        return x


class SqueezeDetBase(nn.Module):
    def __init__(self, cfg):
        super(SqueezeDetBase, self).__init__()
        self.num_classes = cfg.num_classes
        self.num_anchors = cfg.num_anchors
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.arch = cfg.arch
        self.qat = cfg.qat
        if self.arch == 'squeezedet':
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
            self.relu1 = nn.ReLU(inplace=True)
            self.features = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64, self.qat),
                Fire(128, 16, 64, 64, self.qat),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128, self.qat),
                Fire(256, 32, 128, 128, self.qat),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192, self.qat),
                Fire(384, 48, 192, 192, self.qat),
                Fire(384, 64, 256, 256, self.qat),
                Fire(512, 64, 256, 256, self.qat),
                Fire(512, 96, 384, 384, self.qat),
                Fire(768, 96, 384, 384, self.qat)
            )
            out_channels = 768
            self.all_layers = [self.conv1, self.relu1, self.features]

        elif self.arch == 'mobilenet_v2':
            width_mult=1.0
            round_nearest=8
            block = InvertedResidual
            input_channel = 32
            last_channel = 1280
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1, self.qat],
                [6, 24, 2, 2, self.qat],
                [6, 32, 3, 2, self.qat],
                [6, 64, 4, 2, self.qat],
                [6, 96, 3, 1, self.qat],
                [6, 160, 3, 1, self.qat],
                [6, 320, 1, 1, self.qat],
            ]

            if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 5:
                raise ValueError("inverted_residual_setting should be non-empty "
                                "or a 4-element list, got {}".format(inverted_residual_setting))

            # building first layer
            input_channel = _make_divisible(input_channel * width_mult, round_nearest)
            self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
            features = [ConvBNReLU(3, input_channel, stride=2)]
            # building inverted residual blocks
            for t, c, n, s, q in inverted_residual_setting:
                output_channel = _make_divisible(c * width_mult, round_nearest)
                for i in range(n):
                    stride = s if i == 0 else 1
                    features.append(block(input_channel, output_channel, stride, expand_ratio=t, qat=q))
                    input_channel = output_channel
            # building last several layers
            features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
            # make it nn.Sequential
            self.features = nn.Sequential(*features)
            out_channels = last_channel
            self.all_layers = [self.features]

        elif self.arch == 'resnet50':
            out_channels=2048
            block = Bottleneck
            layers = [3, 4, 6, 3]
            self.zero_init_residual = False
            groups = 1
            width_per_group = 64
            replace_stride_with_dilation = None
            norm_layer = None
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            self._norm_layer = norm_layer

            self.inplanes = 64
            self.dilation = 1

            if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
                replace_stride_with_dilation = [False, False, True]
            if len(replace_stride_with_dilation) != 3:
                raise ValueError("replace_stride_with_dilation should be None "
                                "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
            self.groups = groups
            self.base_width = width_per_group
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                        dilate=replace_stride_with_dilation[0])
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                        dilate=replace_stride_with_dilation[1])
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                        dilate=replace_stride_with_dilation[2])
            # self.convl = nn.Conv2d(2048, out_channels, kernel_size=1, stride=1, bias=False)
            self.all_layers = [self.conv1, self.bn1, self.relu, self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4]
        else:
            raise ValueError('Invalid architecture.')

        # self.base_forward = self._forward_impl
        self.dropout = nn.Dropout(cfg.dropout_prob, inplace=True) \
            if cfg.dropout_prob > 0 else None
        self.convdet = nn.Conv2d(out_channels,
                                 cfg.anchors_per_grid * (5),
                                 kernel_size=3, padding=1)
        # self.convdet = nn.Conv2d(out_channels,
        #                          cfg.anchors_per_grid * (cfg.num_classes + 5),
        #                          kernel_size=3, padding=1)

        self.init_weights()

    def forward(self, x):
        if self.qat:
            x = self.quant(x)
        for layer in self.all_layers:
            x = layer(x)
        # x = self.self.base_forard(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.convdet(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(-1, self.num_anchors, 5)
        # x = x.view(-1, self.num_anchors, self.num_classes + 5)
        if self.qat:
            x = self.dequant(x)
        return x
    
    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def init_weights(self):
        if self.arch=='squeezedet':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if m is self.convdet:
                        nn.init.normal_(m.weight, mean=0.0, std=0.002)
                    else:
                        nn.init.normal_(m.weight, mean=0.0, std=0.005)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        elif self.arch=='mobilenet_v2':
            # weight initialization
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if m is self.convdet:
                        nn.init.normal_(m.weight, mean=0.0, std=0.002)
                    else:
                        nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        
        elif self.arch=='resnet50':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            # Zero-initialize the last BN in each residual branch,
            # so that the residual branch starts with zeros, and each residual block behaves like an identity.
            # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                    elif isinstance(m, BasicBlock):
                        nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
class PredictionResolver(nn.Module):
    def __init__(self, cfg, log_softmax=False):
        super(PredictionResolver, self).__init__()
        self.log_softmax = log_softmax
        self.infer_size = cfg.resized_image_size
        self.num_classes = cfg.num_classes
        self.anchors = torch.from_numpy(cfg.anchors).unsqueeze(0).float()
        self.anchors_per_grid = cfg.anchors_per_grid

    def forward(self, pred):
        # pred_class_probs = safe_softmax(pred[..., :self.num_classes].contiguous(), dim=-1)
        # pred_log_class_probs = None if not self.log_softmax else \
        #     torch.log_softmax(pred[..., :self.num_classes].contiguous(), dim=-1)

        # pred_scores = torch.sigmoid(pred[..., self.num_classes:self.num_classes + 1].contiguous())

        # pred_deltas = pred[..., self.num_classes + 1:].contiguous()
        # pred_boxes = deltas_to_boxes(pred_deltas.detach(), self.anchors.to(pred_deltas.device),
        #                              input_size=self.infer_size)

        # return pred_class_probs, pred_log_class_probs, pred_scores, pred_deltas, pred_boxes

        pred_scores = torch.sigmoid(pred[..., 0:1].contiguous())

        pred_deltas = pred[..., 1:].contiguous()
        pred_boxes = deltas_to_boxes(pred_deltas.detach(), self.anchors.to(pred_deltas.device),
                                     input_size=self.infer_size)

        return pred_scores, pred_deltas, pred_boxes

class PredictionResolverSingleClass(nn.Module):
    def __init__(self, cfg, log_softmax=False):
        super(PredictionResolverSingleClass, self).__init__()
        self.log_softmax = log_softmax
        self.input_size = cfg.input_size
        self.num_classes = cfg.num_classes
        self.anchors = torch.from_numpy(cfg.anchors).unsqueeze(0).float()
        self.anchors_per_grid = cfg.anchors_per_grid

    def forward(self, pred):
        pred_class_probs = F.softmax(pred[..., :self.num_classes].contiguous(), dim=-1)

        pred_scores = torch.sigmoid(pred[..., self.num_classes:self.num_classes + 1].contiguous())

        pred_deltas = pred[..., self.num_classes + 1:].contiguous()
        pred_boxes = deltas_to_boxes_tflite(pred_deltas, self.anchors.to(pred_deltas.device),
                                     input_size=self.input_size)

        return pred_class_probs, pred_scores, pred_boxes

class RPNLoss(nn.Module):
    def __init__(self, cfg):
        super(RPNLoss, self).__init__()
        # self.resolver = PredictionResolver(cfg, log_softmax=True)
        self.num_anchors = cfg.num_anchors
        self.class_loss_weight = cfg.class_loss_weight
        self.positive_score_loss_weight = cfg.positive_score_loss_weight
        self.negative_score_loss_weight = cfg.negative_score_loss_weight
        self.bbox_loss_weight = cfg.bbox_loss_weight

    # def forward(self, gt, pred_boxes, pred_deltas, pred_scores, pred_log_class_probs):
    def forward(self, gt, pred_boxes, pred_deltas, pred_scores):
        # slice gt tensor
        anchor_masks = gt[..., :1]
        gt_boxes = gt[..., 1:5]  # xyxy format
        gt_deltas = gt[..., 5:9]
        # gt_class_logits = gt[..., 9:]

        num_objects = torch.sum(anchor_masks, dim=[1, 2])
        num_objects = num_objects + 1
        overlaps = compute_overlaps(gt_boxes, pred_boxes) * anchor_masks

        # rpn_class_loss = torch.sum(
        #     self.class_loss_weight * anchor_masks * gt_class_logits * (-pred_log_class_probs),
        #     dim=[1, 2],
        # ) / num_objects

        positive_score_loss = torch.sum(
            self.positive_score_loss_weight * anchor_masks * (overlaps - pred_scores) ** 2,
            dim=[1, 2]
        ) / num_objects

        negative_score_loss = torch.sum(
            self.negative_score_loss_weight * (1 - anchor_masks) * (overlaps - pred_scores) ** 2,
            dim=[1, 2]
        ) / (self.num_anchors - num_objects)

        bbox_loss = torch.sum(
            self.bbox_loss_weight * anchor_masks * (pred_deltas - gt_deltas) ** 2,
            dim=[1, 2],
        ) / num_objects

        loss = bbox_loss + positive_score_loss + negative_score_loss
        return loss


class ROILoss(nn.Module): 
    def __init__(self, cfg):
        super(ROILoss, self).__init__()
        self.cfg = cfg

    def forward(self, labels, cls_scores):
        classification_loss = torch.empty((cls_scores.shape[0]), device = cls_scores.device)
        for i in range(cls_scores.shape[0]):
            per_image_loss = F.cross_entropy(cls_scores[i], labels[i], ignore_index=-1, reduction='mean')
            classification_loss[i] = per_image_loss
        return classification_loss


class SqueezeDetWithLoss(nn.Module):
    """ Model for training """
    def __init__(self, cfg):
        super(SqueezeDetWithLoss, self).__init__()
        self.base = SqueezeDetBase(cfg)
        self.resolver = PredictionResolver(cfg, log_softmax=False)
        self.rpnloss = RPNLoss(cfg)
        self.roialign = RoIAlign(input_size=cfg.input_size, infer_size = cfg.resized_image_size, output_size = cfg.crop_size, spatial_scale=1.0)
        self.classifier = MobileNetV2(num_classes=cfg.num_classes, width_mult=1.0, inverted_residual_setting=None, round_nearest=8, qat=cfg.qat)
        self.selectrrainingsamples = SelectTrainingSamples(cfg)
        self.roiloss = ROILoss(cfg)
        self.detect = False
        self.arch = cfg.arch
        self.cfg = cfg
        self.torch_interpolation_modes = ['nearest', 'bilinear', 'bicubic' , 'area']
    
    def forward(self, batch):
        if not self.detect:
            mode = random.choice(self.torch_interpolation_modes)
            if mode in ['bilinear', 'bicubic']:
                align_corners = random.choice([True, False])
            else:
                align_corners = None
        else:
            mode = 'nearest'
            align_corners = None
        resized_images = F.interpolate(
                    batch['image'],
                    size=self.cfg.resized_image_size,
                    mode=mode,
                    align_corners=align_corners)

        pred = self.base(resized_images)

        if not self.detect:
            filtered_boxes = []
            # resolver predictions
            pred_scores, pred_deltas, pred_boxes = self.resolver(pred)
            rpn_loss = self.rpnloss(batch['gt'], pred_boxes, pred_deltas, pred_scores)
            pred_scores  = pred_scores.detach().squeeze(dim=2)
            dets = {
                'scores': pred_scores,
                'boxes': pred_boxes}
            batch_size = dets['scores'].shape[0]
            for b in range(batch_size):
                det = {k: v[b] for k, v in dets.items()}
                det = self.filter(det, self.cfg.nms_thresh_train, post_nms_top_k = self.cfg.post_nms_top_k_train)
                filtered_boxes.append(det['boxes'])
            filtered_boxes = torch.stack(filtered_boxes)
            proposals, labels = self.selectrrainingsamples(batch['gt'], filtered_boxes)
            proposals = torch.stack(proposals)
            roi_boxes = self.roialign(batch['image'], proposals)
            cls_scores = self.classifier(roi_boxes)
            cls_scores = cls_scores.view(batch_size, -1, self.cfg.num_classes)
            cls_scores = torch.log_softmax(cls_scores, dim=-1)

            roi_class_loss = self.roiloss(labels, cls_scores)
            loss = rpn_loss + roi_class_loss
            loss_stat = {
                'loss':loss,
                'rpn_loss':rpn_loss,
                'class_loss': roi_class_loss,
            }
            return loss, loss_stat

        else:
            filtered_scores = []
            filtered_boxes = []
            pred_scores, pred_deltas, pred_boxes = self.resolver(pred)
            pred_scores  = pred_scores.detach().squeeze(dim=2)
            dets = {
                'scores': pred_scores,
                'boxes': pred_boxes}
            batch_size = dets['scores'].shape[0]
            for b in range(batch_size):
                det = {k: v[b] for k, v in dets.items()}
                det = self.filter(det, nms_thresh = self.cfg.nms_thresh_train, post_nms_top_k = self.cfg.post_nms_top_k_test)
                filtered_scores.append(det['scores'])
                filtered_boxes.append(det['boxes'])
            filtered_scores = torch.stack(filtered_scores)
            filtered_boxes = torch.stack(filtered_boxes)
            roi_boxes = self.roialign(batch['image'], filtered_boxes.clone())
            cls_scores = self.classifier(roi_boxes)
            cls_scores = cls_scores.view(batch_size, -1, self.cfg.num_classes)
            cls_scores = safe_softmax(cls_scores, dim=-1)
            
            pred_roi_class_scores , pred_roi_class_ids = torch.max(cls_scores, dim=2)
            det = { 'class_ids': pred_roi_class_ids.detach(),
                    'class_scores': pred_roi_class_scores.detach(),
                    'scores': filtered_scores,
                    'boxes': filtered_boxes}
            return det

    def fuse_model(self):
        if self.arch=='squeezedet':
            torch.quantization.fuse_modules(self.base, ['conv1', 'relu1'], inplace=True)
            for m in self.base.features:    
                if type(m) == Fire:
                    torch.quantization.fuse_modules(m, [['squeeze', 'activation_1'], ['expand1x1', 'activation_2'], ['expand3x3', 'activation_3']] , inplace=True)
        elif self.arch=='mobilenet_v2':
            # Fuse Conv+BN and Conv+BN+Relu modules prior to quantization
            # This operation does not change the numerics
            for m in self.base.modules():
                if type(m) == ConvBNReLU:
                    torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
                if type(m) == InvertedResidual:
                    for idx in range(len(m.conv)):
                        if type(m.conv[idx]) == nn.Conv2d:
                            torch.quantization.fuse_modules(m.conv, [str(idx), str(idx + 1)], inplace=True)

    def filter(self, det, nms_thresh, post_nms_top_k):

            ## Pre NMS TopK
            orders = torch.argsort(det['scores'], descending=True)[:self.cfg.keep_top_k]
            scores = det['scores'][orders]
            boxes = det['boxes'][orders, :]
            rm_small_keep = box_ops.remove_small_boxes(boxes, self.cfg.object_size_thresh[0])
            scores_filt1 = scores[rm_small_keep]
            boxes_filt1 = boxes[rm_small_keep]
            keeps = nms(boxes_filt1, scores_filt1, nms_thresh)
            diff = post_nms_top_k - keeps.shape[0]
            if diff >0:
                indices = torch.randint(0, keeps.shape[0], (diff,))            
                keeps = torch.cat((keeps, keeps[indices]), dim=0)
            scores_filt2 = scores_filt1[keeps]
            boxes_filt2 = boxes_filt1[keeps,:]
            det = {
                'scores': scores_filt2[:post_nms_top_k],
                'boxes': boxes_filt2[:post_nms_top_k, :]}

            return det


class SqueezeDet(nn.Module):
    """ Model for inference """
    def __init__(self, cfg):
        super(SqueezeDet, self).__init__()
        self.base = SqueezeDetBase(cfg)
        self.arch = cfg.arch
        self.qat = cfg.qat

    def forward(self, image):
        pred = self.base(image)
        return pred
    
    def fuse_model(self):
        if self.arch=='squeezedet':
            torch.quantization.fuse_modules(self.base, ['conv1', 'relu1'], inplace=True)
            for m in self.base.features:    
                if type(m) == Fire:
                    torch.quantization.fuse_modules(m, [['squeeze', 'activation_1'], ['expand1x1', 'activation_2'], ['expand3x3', 'activation_3']] , inplace=True)
        elif self.arch=='mobilenet_v2':
            # Fuse Conv+BN and Conv+BN+Relu modules prior to quantization
            # This operation does not change the numerics
            for m in self.base.modules():
                if type(m) == ConvBNReLU:
                    torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
                if type(m) == InvertedResidual:
                    for idx in range(len(m.conv)):
                        if type(m.conv[idx]) == nn.Conv2d:
                            torch.quantization.fuse_modules(m.conv, [str(idx), str(idx + 1)], inplace=True)


class SqueezeDetWithResolver(nn.Module):
    """ Model for inference """
    def __init__(self, cfg):
        super(SqueezeDetWithResolver, self).__init__()
        self.base = SqueezeDetBase(cfg)
        self.resolver = PredictionResolverSingleClass(cfg, log_softmax=False)
        self.arch = cfg.arch
        self.qat = cfg.qat
        self.grid_size = cfg.grid_size
        self.anchors_per_grid = cfg.anchors_per_grid
        self.num_classes = cfg.num_classes


    def forward(self, image):
        pred = self.base(image)
        pred_class_probs, pred_scores, pred_boxes = self.resolver(pred)
        # pred_boxes = xyxy_to_xywh(pred_boxes)
        return pred_boxes, pred_scores, pred_class_probs

    def fuse_model(self):
        if self.arch=='squeezedet':
            torch.quantization.fuse_modules(self.base, ['conv1', 'relu1'], inplace=True)
            for m in self.base.features:    
                if type(m) == Fire:
                    torch.quantization.fuse_modules(m, [['squeeze', 'activation_1'], ['expand1x1', 'activation_2'], ['expand3x3', 'activation_3']] , inplace=True)
        elif self.arch=='mobilenet_v2':
            # Fuse Conv+BN and Conv+BN+Relu modules prior to quantization
            # This operation does not change the numerics
            for m in self.base.modules():
                if type(m) == ConvBNReLU:
                    torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
                if type(m) == InvertedResidual:
                    for idx in range(len(m.conv)):
                        if type(m.conv[idx]) == nn.Conv2d:
                            torch.quantization.fuse_modules(m.conv, [str(idx), str(idx + 1)], inplace=True)


    def format_output(self, pred_boxes, class_scores, obj):
        batch_size = pred_boxes.shape[0]
        output = torch.cat(
            (
                pred_boxes.view(batch_size, -1, 4),
                obj.view(batch_size, -1, 1),
                class_scores.view(batch_size, -1, self.num_classes)
            ),
            -1
        )
        return output


def xyxy_to_xywh(boxes_xyxy):
    return torch.cat([
        (boxes_xyxy[..., [0]] + boxes_xyxy[..., [2]]) / 2.,
        (boxes_xyxy[..., [1]] + boxes_xyxy[..., [3]]) / 2.,
        boxes_xyxy[..., [2]] - boxes_xyxy[..., [0]],
        boxes_xyxy[..., [3]] - boxes_xyxy[..., [1]]
    ], dim=-1)