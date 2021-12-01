"""
Images implements based on Numpy
"""
import torch
import torchvision.transforms as transforms
import glob
import math
import numpy as np
import cv2
import random
import secrets
from PIL import Image

from PIL import Image, ImageDraw
from PIL import ImageFilter, ImageOps
from imgaug.augmentables.polys import Polygon, PolygonsOnImage
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imgaug as ia
import imgaug.augmenters as iaa

def whiten(image, image_meta, mean=0., std=1.):
    """
    :param image:
    :param image_meta:
    :param mean: float or np.ndarray(1, 1, 3)
    :param std: float or np.ndarray(1, 1, 3)
    :return:
    """
    image = (image - mean) / std
    image_meta.update({'rgb_mean': mean, 'rgb_std': std})
    return image, image_meta


def drift(image, image_meta, prob=0., boxes=None):
    drifts = np.array([0, 0], dtype=np.int32)
    drifted_size = np.array(image.shape, dtype=np.int32)
    if np.random.uniform() < prob:
        max_drift_y = image_meta['orig_size'][0] // 4
        max_drift_x = image_meta['orig_size'][1] // 8
        max_boxes_y = min(boxes[:, 1]) if boxes is not None else max_drift_y
        max_boxes_x = min(boxes[:, 0]) if boxes is not None else max_drift_x
        dy = np.random.randint(-max_drift_y, min(max_drift_y, max_boxes_y))
        dx = np.random.randint(-max_drift_x, min(max_drift_x, max_boxes_x))
        drifts = np.array([dy, dx], dtype=np.int32)

        image_height = image_meta['orig_size'][0] - dy
        image_width = image_meta['orig_size'][1] - dx
        orig_x, orig_y = max(dx, 0), max(dy, 0)
        drift_x, drift_y = max(-dx, 0), max(-dy, 0)

        drifted_image = np.zeros((image_height, image_width, 3)).astype(np.float32)
        drifted_image[drift_y:, drift_x:, :] = image[orig_y:, orig_x:, :]
        image = drifted_image
        drifted_size = np.array(image.shape, dtype=np.int32)

        if boxes is not None:
            boxes[:, [0, 2]] -= dx
            boxes[:, [1, 3]] -= dy

    image_meta.update({'drifts': drifts, 'drifted_size': drifted_size})

    return image, image_meta, boxes


def flip(image, image_meta, prob=0., boxes=None):
    """
    :param image:
    :param image_meta:
    :param prob:
    :param boxes: xyxy format
    :return:
    """
    flipped = False
    if np.random.uniform() < prob:
        flipped = True
        image = image[:, ::-1, :].copy()

    if flipped and boxes is not None:
        image_width = image.shape[1]
        boxes_widths = boxes[:, 2] - boxes[:, 0]
        boxes[:, 0] = image_width - 1 - boxes[:, 2]
        boxes[:, 2] = boxes[:, 0] + boxes_widths

    image_meta.update({'flipped': flipped})

    return image, image_meta, boxes


def resize(image, image_meta, target_size, boxes=None):
    width, height = image.size[:2]
    scales = np.array([target_size[0] / height, target_size[1] / width], dtype=np.float32)
    # image = cv2.resize(image, (target_size[1], target_size[0]))
    image = transforms.Resize(target_size) (image)
    if boxes is not None:
        boxes[:, [0, 2]] *= scales[1]
        boxes[:, [1, 3]] *= scales[0]

    image_meta.update({'scales': scales})
    return image, image_meta, boxes


def crop_or_pad(image, image_meta, target_size, boxes=None):
    """
    :param image:
    :param image_meta:
    :param target_size: (height, width)
    :param boxes: xyxy format
    :return:
    """
    padding, crops = np.zeros(4, dtype=np.int16), np.zeros(4, dtype=np.int16)  # (top, bottom, left, right) format

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

    image, boxes = pad(image, padding, boxes=boxes)
    image, boxes = crop(image, crops, boxes=boxes)

    image_meta.update({'padding': padding, 'crops': crops})

    return image, image_meta, boxes


def pad(image, padding, boxes=None):
    """
    :param image:
    :param padding: (top_pad, bottom_pad, left_pad, right_pad)
    :param boxes: xyxy format
    :return:
    """
    if not np.all(padding == 0):
        padding = (padding[:2], padding[2:], [0, 0])
        image = np.pad(image, padding, mode='constant')
        if boxes is not None:
            boxes[:, [0, 2]] += padding[2]
            boxes[:, [1, 3]] += padding[0]

    return image, boxes


def crop(image, crops, boxes=None):
    """
    :param image:
    :param crops: (top_crop, bottom_crop, left_crop, right_crop)
    :param boxes: xyxy format
    :return:
    """
    if not np.all(crops == 0):
        image = image[crops[0]:-crops[1], :, :] if crops[1] > 0 else image[crops[0]:, :, :]
        image = image[:, crops[2]:-crops[3], :] if crops[3] > 0 else image[:, crops[2]:, :]
        if boxes is not None:
            boxes[:, [0, 2]] -= crops[2]
            boxes[:, [1, 3]] -= crops[0]
            boxes = np.maximum(boxes, 0.)

    return image, boxes


def image_postprocess(image, image_meta):
    if 'scales' in image_meta:
        image = cv2.resize(image, tuple(image_meta['orig_size']))

    if 'padding' in image_meta:
        image = crop(image, image_meta['padding'])

    if 'crops' in image_meta:
        image = pad(image, image_meta['crops'])

    if 'flipped' in image_meta and image_meta['flipped']:
        image = image[:, ::-1, :]

    if 'drifts' in image_meta:
        padding = [image_meta['drifts'][0], 0, image_meta['drifts'][1], 0]
        image = pad(image, padding)[0]

    if 'rgb_mean' in image_meta and 'rgb_std' in image_meta:
        image = image * image_meta['rgb_std'] + image_meta['rgb_mean']

    return image

def pil_to_cv2(img):
    """This function converts PIL image to opencv.

    Args:
        img (PIL Image): Image in PIL format.

    Returns:
        Image (opencv): Image object in opencv format.
    """
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv2_to_pil(img):
    """This function converts opencv image to PIL.

    Args:
        img (opencv Image): Image in opencv format.

    Returns:
        Image (PIL): Image object in PIL format.
    """
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def blackout_image(img, bboxes):
    crop_pad = 5
    nonlabel = -1
    area_thresh = 1.0
    obj_path = '/home/aamir/Documents/LPR/alpr_train_data/synthetic_plates'
    im_ext=''
    if not obj_path:
        assert 'obj_path not configured '
    obj_paths = glob.glob(f"{obj_path}/*{im_ext}")

    color_imgaug = [
        iaa.Sometimes(0.8, iaa.OneOf([
            iaa.GaussianBlur(sigma=(2, 4)),
            iaa.AverageBlur(k=(2, 5)),
            iaa.MotionBlur(k=(3, 10), angle=(-25, 25))
        ])),
        iaa.Sometimes(0.7, iaa.OneOf([
            iaa.Cutout(nb_iterations=(5, 20), size=(0.01, 0.08), squared=False, fill_mode="gaussian", fill_per_channel=True),
            # iaa.SaltAndPepper(p=(0.01, 0.2)),
            # iaa.Dropout(p=(0.01, 0.05), per_channel=0.5),
            iaa.ReplaceElementwise((0.01, 0.3), [0, 255]),
        ])),
    ]

    shape_imgaug = [
        iaa.Sometimes(0.5, iaa.Affine(rotate=(-10, 10), fit_output=True)),
        iaa.Sometimes(0.5, iaa.PerspectiveTransform(scale=0.05, keep_size=True, fit_output=True)),
        iaa.Fliplr(0.5),
    ]

    # aug = Compose([
    #         HSVPro(0.9, 0.1, 0.7, 0.7),
    #     ])
    mode = img.mode
    img_w, img_h = img.size
    out_boxes = bboxes.tolist()
    if bboxes.nelement():
        for bbox_i in bboxes:
            # bbox = bbox_i[1:]
            bbox = bbox_i

            # Coordinates x1,y1,x2,y2
            left = int(bbox[0])
            right = int(bbox[2])
            top = int(bbox[1])
            bottom = int(bbox[3])

            cx = int((left + right) / 2)
            cy = int((top + bottom) / 2)
            w = int(right - left)
            h = int(bottom - top)

            #### Replace License Plate with some other area from the image ####
            # area_i = (right-left)*(bottom-top)
            # crop_pad_i = int(math.sqrt(area_i)*(5/100))

            # new_crop = img.crop(((cx+w), (cy), (cx+w+int(w*1.5)), (cy+int(h*1.5))))

            # mask = Image.new('L', (int(w*1.5), int(h*1.5)), 255)
            # blck = Image.new('L', ((int(w*1.5))-crop_pad_i, (int(h*1.5))-crop_pad_i), 0)
            # mask.paste(blck, (int(crop_pad_i/2), int(crop_pad_i/2)))
            # blurmask = ImageOps.invert(
            #     mask.filter(ImageFilter.GaussianBlur(crop_pad_i/2))
            # )
            # img.paste(new_crop, ((left-int(w/3)), top-int(h/3)), blurmask)
            #### Replace License Plate with some other area from the image ####
            obj_index = random.randint(0, len(obj_paths)-1)
            obj_image = Image.open(obj_paths[obj_index])
            obj_image = obj_image.convert(mode)

            # Pil Augmentations
            # obj_image = aug(obj_image)[0]

            # ImageAug Color Augmentations
            cv2_img = pil_to_cv2(obj_image)
            for aug in color_imgaug:
                cv2_img = aug(image=cv2_img)
            obj_image = cv2_to_pil(cv2_img)

            obj_w, obj_h = obj_image.size

            obj_aspect_ratio = obj_w / obj_h
            obj_h = max(h//2,16)
            obj_w = int(obj_h*obj_aspect_ratio)
            obj_image = obj_image.resize((obj_w, obj_h))

            # ImageAug shape Augmentations
            cv2_img = pil_to_cv2(obj_image)
            obj_bbox = BoundingBox(0, 0, obj_w, obj_h)
            obj_poly = Polygon([
                (0, 0), (obj_w, 0), (obj_w, obj_h), (0, obj_h), (0, 0)])

            aug_obj_img, aug_obj_bbox, aug_obj_poly = \
                cv2_img,\
                BoundingBoxesOnImage([obj_bbox], shape=cv2_img.shape),\
                PolygonsOnImage([obj_poly], shape=cv2_img.shape)

            # for aug in shape_imgaug:
            #     aug_obj_img, aug_obj_bbox, aug_obj_poly = \
            #         aug(
            #             image=aug_obj_img,
            #             bounding_boxes=aug_obj_bbox,
            #             polygons=aug_obj_poly
            #         )

            # Co-ordinates cx,cy,w,h
            left = aug_obj_bbox[0].x1
            right = aug_obj_bbox[0].x2
            top = aug_obj_bbox[0].y1
            bottom = aug_obj_bbox[0].y2

            # cx = int((left + right) / 2)
            # cy = int((top + bottom) / 2)
            # w = int(right - left)
            # h = int(bottom - top)

            area_i = (right-left)*(bottom-top)
            crop_pad_i = int(math.sqrt(area_i)*(crop_pad/100))
            crop_pad_i = crop_pad_i+1 if crop_pad_i % 2 == 0 else crop_pad_i

            ref_obj_aug_poly = []
            for x, y in aug_obj_poly[0]:
                ref_obj_aug_poly.append((x, y))

            aug_obj_img = cv2_to_pil(aug_obj_img)
            obj_w, obj_h = aug_obj_img.size

            # displacement_y = \
            #     random.randint(max(int(0.1 * h), 1), int(2 * h))
            displacement_y = \
                    random.randint(max(int(0.5 * h), 1), int(2 * h))
            flag = secrets.randbelow(1000)/1000 < 0.5
            obj_cy = cy - displacement_y if flag else cy + displacement_y
            obj_top = obj_cy - int(obj_h/2)
            obj_bottom = obj_cy + int(obj_h/2)

            displacement_x = random.randint(int(w * 0.5), 1 * w)
            # displacement_x = random.randint(int(w * 0.1), 2 * w)
            flag = secrets.randbelow(1000)/1000 < 0.5
            obj_cx = cx - random.randint(int(w/2)+obj_w+w, int(w/2)+obj_w+(2*w)) if flag else cx + random.randint(int(w * 1.5), 2 * w)
            obj_left = obj_cx - int(obj_w/2)
            obj_right = obj_cx + int(obj_w/2)

            if obj_cx < 0 or obj_cx > img_w or obj_cy < 0 or obj_cy > img_h:
                continue

            mask_img = Image.new('L', (obj_w, obj_h), 0)
            ImageDraw.Draw(mask_img).polygon(ref_obj_aug_poly, outline=255, fill=255)

            alpha_channel = Image.new('L', img.size, 0)
            alpha_channel.paste(mask_img, (obj_left, obj_top))
            kernel = np.ones((crop_pad_i, crop_pad_i), np.uint8)
            alpha_channel = cv2.erode(np.array(alpha_channel), kernel, iterations = 1)
            alpha_channel = cv2.GaussianBlur(alpha_channel, (crop_pad_i, crop_pad_i), crop_pad_i, 0)
            alpha_channel = Image.fromarray(alpha_channel)

            out_obj_image = Image.new(img.mode, img.size, 0)
            out_obj_image.paste(aug_obj_img, (obj_left, obj_top))

            img.paste(out_obj_image, (0, 0), alpha_channel)
            out_boxes.append([
                # bbox_i,
                max(min(obj_left, img_w), 0),
                max(min(obj_top, img_h), 0),
                max(min(obj_right, img_w), 0),
                max(min(obj_bottom, img_h), 0)
            ])
    tmp = img.resize((img.size[0]*2, img.size[1]*2))
    img = tmp.resize((img.size[0], img.size[1]))
    return img, torch.tensor(out_boxes)

def synthetic_plates(image, image_meta, prob, label=torch.empty(0, 5)):
    if secrets.randbelow(1000)/1000 < prob:
        augmented_im, augmented_label = blackout_image(image, label)
    else:
        augmented_im, augmented_label = image, label
    return augmented_im, augmented_label