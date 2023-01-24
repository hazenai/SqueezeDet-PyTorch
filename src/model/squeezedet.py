import torch
import torch.nn as nn

from model.modules import deltas_to_boxes, deltas_to_boxes_tflite, compute_overlaps, safe_softmax
import torch.nn.functional as F

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


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes, momentum=0.1),
            # Replace with ReLU
            nn.ReLU(inplace=False)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, qat):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.qat = qat
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup, momentum=0.1),
        ])
        self.conv = nn.Sequential(*layers)
        # Replace torch.add with floatfunctional
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_res_connect:
            if self.qat:
                return self.skip_add.add(x, self.conv(x))
            else:
                return torch.add(x, self.conv(x))
        else:
            return self.conv(x)


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
        # elif cfg.arch == 'squeezedetplus':
        #     self.features = nn.Sequential(
        #         nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=3),
        #         nn.ReLU(inplace=True),
        #         nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        #         Fire(96, 96, 64, 64),
        #         Fire(128, 96, 64, 64),
        #         Fire(128, 192, 128, 128),
        #         nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        #         Fire(256, 192, 128, 128),
        #         Fire(256, 288, 192, 192),
        #         Fire(384, 288, 192, 192),
        #         Fire(384, 384, 256, 256),
        #         nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        #         Fire(512, 384, 256, 256),
        #         Fire(512, 384, 256, 256),
        #         Fire(512, 384, 256, 256),
        #     )
        #     out_channels = 512
        elif self.arch == 'mobilenet_v2':
            width_mult=1.0
            round_nearest=8
            block = InvertedResidual
            input_channel = 32
            last_channel = 256
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1, self.qat],
                [6, 24, 2, 2, self.qat],
                [6, 32, 3, 2, self.qat],
                [6, 64, 2, 1, self.qat],
                # [6, 96, 3, 1, self.qat],
                # [6, 160, 3, 1, self.qat],
                # [6, 320, 1, 1, self.qat],
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

        else:
            raise ValueError('Invalid architecture.')

        self.dropout = nn.Dropout(cfg.dropout_prob, inplace=True) \
            if cfg.dropout_prob > 0 else None
        self.convdet = nn.Conv2d(out_channels,
                                 cfg.anchors_per_grid * (cfg.num_classes + 5),
                                 kernel_size=3, padding=1)

        self.init_weights()

    def forward(self, x):
        if self.qat:
            x = self.quant(x)
        if self.arch=='squeezedet':
            x = self.conv1(x)
            x = self.relu1(x)
        x = self.features(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.convdet(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(-1, self.num_anchors, self.num_classes + 5)
        if self.qat:
            x = self.dequant(x)
        return x
    def init_weights(self):
        if self.arch=='squeezedet':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if m is self.convdet:
                        #nn.init.normal_(m.weight, mean=0.0, std=0.002)   
                        #nn.init.normal_(m.weight, mean=0.0, std=0.041)   #good_result_1                   
                        nn.init.normal_(m.weight, mean=0.0, std=0.023)                               
                    else:
                        #nn.init.normal_(m.weight, mean=0.0, std=0.005)     
                        #nn.init.normal_(m.weight, mean=0.0, std=0.083)       #good_result_1                     
                        nn.init.normal_(m.weight, mean=0.0, std=0.045)                           
                    if m.bias is not None:   
                        #nn.init.constant_(m.bias, 0) 
                        #nn.init.uniform(m.bias,a=-0.01,b=0.01)  
                        #nn.init.normal_(m.weight, mean=0.0, std=0.1)         #good_result_1                       
                        nn.init.normal_(m.weight, mean=0.0, std=0.053)                           
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
                    

class PredictionResolver(nn.Module):
    def __init__(self, cfg, log_softmax=False):
        super(PredictionResolver, self).__init__()
        self.log_softmax = log_softmax
        self.input_size = cfg.input_size
        self.num_classes = cfg.num_classes
        self.anchors = torch.from_numpy(cfg.anchors).unsqueeze(0).float()
        self.anchors_per_grid = cfg.anchors_per_grid

    def forward(self, pred):
        pred_class_probs = safe_softmax(pred[..., :self.num_classes].contiguous(), dim=-1)
        pred_log_class_probs = None if not self.log_softmax else \
            torch.log_softmax(pred[..., :self.num_classes].contiguous(), dim=-1)

        pred_scores = torch.sigmoid(pred[..., self.num_classes:self.num_classes + 1].contiguous())

        pred_deltas = pred[..., self.num_classes + 1:].contiguous()
        pred_boxes = deltas_to_boxes(pred_deltas, self.anchors.to(pred_deltas.device),
                                     input_size=self.input_size)

        return pred_class_probs, pred_log_class_probs, pred_scores, pred_deltas, pred_boxes


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

class Loss(nn.Module):
    def __init__(self, cfg):
        super(Loss, self).__init__()
        self.resolver = PredictionResolver(cfg, log_softmax=True)
        self.num_anchors = cfg.num_anchors
        self.class_loss_weight = cfg.class_loss_weight
        self.positive_score_loss_weight = cfg.positive_score_loss_weight
        self.negative_score_loss_weight = cfg.negative_score_loss_weight
        self.bbox_loss_weight = cfg.bbox_loss_weight

    def forward(self, pred, gt):
        # slice gt tensor
        anchor_masks = gt[..., :1]
        gt_boxes = gt[..., 1:5]  # xyxy format
        gt_deltas = gt[..., 5:9]
        gt_class_logits = gt[..., 9:]

        # resolver predictions
        pred_class_probs, pred_log_class_probs, pred_scores, pred_deltas, pred_boxes = self.resolver(pred)

        num_objects = torch.sum(anchor_masks, dim=[1, 2])
        num_objects = num_objects + 1
        overlaps = compute_overlaps(gt_boxes, pred_boxes) * anchor_masks

        class_loss = torch.sum(
            self.class_loss_weight * anchor_masks * gt_class_logits * (-pred_log_class_probs),
            dim=[1, 2],
        ) / num_objects

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


        # positive_score_loss = torch.sum((1/len([x for x in anchor_masks[0] if x==1]))*(anchor_masks - (anchor_masks * pred_scores)) ,dim=[1, 2])    # Added line            
        # negative_score_loss = torch.sum((1/len([x for x in anchor_masks[0] if x==0])*2) * ((1-anchor_masks) * pred_scores),dim=[1, 2])    # Added line               

        loss = class_loss + positive_score_loss + negative_score_loss + bbox_loss    
        loss_stat = {
            'loss': loss,
            'class_loss': class_loss,
            'score_loss': positive_score_loss + negative_score_loss, 
            'bbox_loss': bbox_loss
        }

        return loss, loss_stat


class SqueezeDetWithLoss(nn.Module):
    """ Model for training """
    def __init__(self, cfg):
        super(SqueezeDetWithLoss, self).__init__()
        self.base = SqueezeDetBase(cfg)
        self.resolver = PredictionResolver(cfg, log_softmax=False)
        self.loss = Loss(cfg)
        self.detect = False
        self.arch = cfg.arch

    def forward(self, batch):
        pred = self.base(batch['image'])
        if not self.detect:
            loss, loss_stats = self.loss(pred, batch['gt'])
            return loss, loss_stats
        
        else:
            pred_class_probs, _, pred_scores, _, pred_boxes = self.resolver(pred)
            pred_class_probs *= pred_scores
            pred_class_ids = torch.argmax(pred_class_probs, dim=2)
            pred_scores = torch.max(pred_class_probs, dim=2)[0]
            det = {'class_ids': pred_class_ids,
                'scores': pred_scores,
                'boxes': pred_boxes}
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