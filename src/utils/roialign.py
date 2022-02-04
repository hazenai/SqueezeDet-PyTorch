"""
    Description: Implementation of the "CropAndResize" layer
    in the YOLT network. ROIAlign is used to perform this operation
    since its ONNX and OpenVino conversion is supported

    Author: raza@hazen.ai
"""

import torch
from torch import nn, Tensor
from torch.nn.modules.utils import _pair
from torch.jit.annotations import List, BroadcastingList2
from torchvision.transforms import Normalize as normalize


def _cat(tensors: List[Tensor], dim: int = 0) -> Tensor:
    """
    Efficient version of torch.cat that avoids a copy if there is
    only a single element in a list
    """
    # TODO add back the assert
    # assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def convert_boxes_to_roi_format(boxes: List[Tensor]) -> Tensor:
    concat_boxes = _cat([b for b in boxes], dim=0)
    temp = []
    for i, b in enumerate(boxes):
        temp.append(torch.full_like(b[:, :1], i))
    ids = _cat(temp, dim=0)
    rois = torch.cat([ids, concat_boxes], dim=1)
    return rois


def check_roi_boxes_shape(boxes: Tensor):
    if isinstance(boxes, (list, tuple)):
        for _tensor in boxes:
            assert _tensor.size(1) == 4, \
                'The shape of the tensor in the boxes list is \
                 not correct as List[Tensor[L, 4]]'
    elif isinstance(boxes, torch.Tensor):
        assert boxes.size(1) == 5, 'The boxes tensor shape is not correct as Tensor[K, 5]'
    else:
        assert False, 'boxes is expected to be a Tensor[L, 5] or a List[Tensor[K, 4]]'
    return


def roi_align(
    input: Tensor,
    boxes: Tensor,
    input_size: BroadcastingList2[int],
    output_size: BroadcastingList2[int],
    spatial_scale: float = 1.0,
) -> Tensor:
    """
    Performs Region of Interest (RoI) Align operator described in Mask R-CNN
    Arguments:
        input (Tensor[N, C, H, W]): input tensor
        boxes (Tensor[K, 5] or List[Tensor[L, 4]]): the box coordinates in (x1, y1, x2, y2)
            format where the regions will be taken from. If a single Tensor is passed,
            then the first column should contain the batch index. If a list of Tensors
            is passed, then each Tensor will correspond to the boxes for an element i
            in a batch
        output_size (int or Tuple[int, int]): the size of the output after the cropping
            is performed, as (height, width)
        spatial_scale (float): a scaling factor that maps the input coordinates to
            the box coordinates. Default: 1.0
        sampling_ratio (int): number of sampling points in the interpolation grid
            used to compute the output value of each pooled output bin. If > 0,
            then exactly sampling_ratio x sampling_ratio grid points are used. If
            <= 0, then an adaptive number of grid points are used (computed as
            ceil(roi_width / pooled_w), and likewise for height). Default: -1
        aligned (bool): If False, use the legacy implementation.
            If True, pixel shift it by -0.5 for align more perfectly about two neighboring pixel indices.
            This version in Detectron2
    Returns:
        output (Tensor[K, C, output_size[0], output_size[1]])
    """
    _boxes = [boxes[i, :, :4] for i in range(len(boxes))]
    check_roi_boxes_shape(_boxes)
    rois = _boxes
    output_size = _pair(output_size)
    if not isinstance(rois, torch.Tensor):
        rois = convert_boxes_to_roi_format(rois)
    input_shape = input.shape
    sw = torch.tensor(input_shape[3], dtype=torch.float)/input_size[1]
    sh = torch.tensor(input_shape[2], dtype=torch.float)/input_size[0]
    rois[:, 1] = rois[:, 1]*sw
    rois[:, 3] = rois[:, 3]*sw
    rois[:, 2] = rois[:, 2]*sh
    rois[:, 4] = rois[:, 4]*sh
    output = torch.ops.torchvision.roi_align(input,
                                             rois, spatial_scale,
                                             output_size[0], output_size[1],
                                             sampling_ratio=-1, aligned=True
                                             )
    return output


class RoIAlign(nn.Module):
    def __init__(self, input_size, infer_size, output_size, spatial_scale):
        super(RoIAlign, self).__init__()
        self.input_size = input_size
        self.infer_size = infer_size
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, image_tensor: Tensor, boxes: Tensor) -> Tensor:
        
        inf_h, inf_w = self.infer_size
        inp_h, inp_w = self.input_size
        boxes[..., [0]] = (boxes[..., [0]]/inf_w) * inp_w
        boxes[..., [1]] = (boxes[..., [1]]/inf_h) * inp_h
        boxes[..., [2]] = (boxes[..., [2]]/inf_w) * inp_w
        boxes[..., [3]] = (boxes[..., [3]]/inf_h) * inp_h

        return roi_align(
            image_tensor,
            boxes,
            self.input_size,
            self.output_size,
            self.spatial_scale
        )
