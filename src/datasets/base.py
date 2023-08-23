import os

import numpy as np
import torch.utils.data

from utils.image import whiten, drift, flip, resize, crop_or_pad
from utils.boxes import compute_deltas, visualize_boxes
import imgaug as ia
import imgaug.augmenters as iaa
import random
import torchvision.transforms as transforms
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from PIL import Image,ImageDraw
import cv2     

import random

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, phase, cfg):
        super(BaseDataset, self).__init__()
        self.phase = phase
        self.cfg = cfg
        if cfg.dataset=='lpr':
            self.seq = iaa.Sequential([
                iaa.SomeOf((0, 2),[
                    iaa.Flipud(0.5),
                    iaa.Fliplr(0.1),
                ]),
                # Perspective/Affine
                iaa.Sometimes(
                    p=0.3,
                    then_list=iaa.OneOf([
                            iaa.Affine(
                                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                                    rotate=(-20, 20),
                                    shear=(-15, 15),
                                    order=[0, 1],
                                    cval=(0, 255),
                                    mode=ia.ALL
                                ),
                            iaa.CropAndPad(percent=(-0.3, 0.3), pad_mode=ia.ALL),
                            iaa.PerspectiveTransform(scale=(0.02, 0.15)),
                            # iaa.PiecewiseAffine(scale=(0.01, 0.1)),
                        ])
                ),
                iaa.Sometimes(
                    0.1,
                    iaa.OneOf([
                        iaa.Rain(speed=(0.1, 0.2)),
                        iaa.Snowflakes(flake_size=(0.2, 0.7), speed=(0.007, 0.03))
                    ])
                ),
                iaa.Sometimes(
                    p=0.05,
                    then_list=iaa.OneOf([
                    # iaa.Cartoon(blur_ksize=3, segmentation_size=1.0, saturation=2.0, edge_prevalence=1.0),
                    iaa.DirectedEdgeDetect(alpha=(0.0, 0.3), direction=(0.0, 1.0)),
                    iaa.Solarize(p=1),
                    iaa.pillike.Posterize(nb_bits=(1, 8)),
                    # iaa.RandAugment(m=(1, 5)),
                    ])
                ),
                iaa.Sometimes(
                    p=0.20,
                    then_list=iaa.OneOf([
                        ## Smoothing
                        iaa.OneOf([
                            iaa.pillike.FilterSmooth(),
                            iaa.pillike.FilterSmoothMore()
                        ]),
                        ## Blurring
                        iaa.OneOf([
                            iaa.imgcorruptlike.DefocusBlur(severity=1),
                            iaa.imgcorruptlike.ZoomBlur(severity=1),
                            iaa.MotionBlur(k=(3, 15), angle=(0, 360),direction=(-1.0, 1.0)),
                            iaa.imgcorruptlike.MotionBlur(severity=1),
                            iaa.BilateralBlur(d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250)),
                        ]),
                        ## Edge Enhancement
                        iaa.OneOf([
                            iaa.pillike.FilterEdgeEnhance(),
                            iaa.pillike.FilterEdgeEnhanceMore(),
                            iaa.pillike.FilterContour(),
                            iaa.pillike.FilterDetail(),            
                        ])
                    ])
                ),
            ],random_order=True)
        elif cfg.dataset=='yolo':
            self.seq = iaa.Sequential([
            
            # epoch time increases x2
            # iaa.Sometimes(
            #     0.1,
            #     iaa.OneOf([
            #         iaa.AdditiveLaplaceNoise(scale=0.2*255),
            #         iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
            #     ])
            # ),
            iaa.Sometimes(
                0.3,
                iaa.OneOf([
                    iaa.Multiply((0.95,1.05), per_channel=0.25),
                    iaa.LinearContrast((0.95,1.05)),
                ])
            ),
            
            
            iaa.Sometimes(
                0.2,
                iaa.OneOf([
                    iaa.MultiplyElementwise((0.5, 1.5), per_channel=0.5),
                ])
                
            ),
            iaa.Sometimes(
                0.1,
                iaa.OneOf(
                    [
                        iaa.Dropout(p=(0.0, 0.1), per_channel=0.5),
                        iaa.CoarseDropout(
                            p=(0.02, 0.1), size_percent=(0.02, 0.15), per_channel=0.5
                        ),
                        # iaa.Dropout2d(p=1),
                    ]
                ),
            ),
            iaa.Sometimes(
                0.1,
                iaa.OneOf([
                    iaa.Fliplr(),
                ])
            ),
            iaa.Sometimes(
                0.1,
                iaa.OneOf([
                    iaa.imgcorruptlike.JpegCompression(severity=2),
                    iaa.imgcorruptlike.JpegCompression(severity=1),
                    iaa.imgcorruptlike.Pixelate(severity=2),
                    iaa.CropAndPad(percent=(-0.3, 0.3), pad_mode=ia.ALL, keep_size=False),
                    iaa.PerspectiveTransform(scale=(0.02, 0.125), keep_size=False),
                ])
            ),

            iaa.Sometimes(
                0.4,
                iaa.OneOf([
                    iaa.GaussianBlur((0.0, 2.0)),
                    iaa.AverageBlur((2,5)),
                    iaa.MedianBlur((3,5))
                ])
            ),
            iaa.Sometimes(
                0.2,
                iaa.OneOf([
                    iaa.MotionBlur(k=(3, 4), angle=(0, 360), direction=(-1.0, 1.0)),
                    iaa.BilateralBlur(
                        d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250)
                    ),
                    iaa.ChangeColorspace(from_colorspace="BGR", to_colorspace="HSV"),
                ])
            ),
            iaa.Sometimes(
                p=0.1,
                then_list=iaa.OneOf([
                    # # Smoothing
                    iaa.pillike.FilterSmooth(),
                    iaa.pillike.FilterSmoothMore()
                ])
            ),
            iaa.Sometimes(
                0.1,
                iaa.OneOf([
                    iaa.ChangeColorTemperature((3500, 15000)),
                ])
            ),
            ],random_order=True)

    def __getitem__(self, index):
        image, image_id = self.load_image(index)
        # gt_class_ids, gt_boxes = self.load_annotations(index)
        gt_class_ids, gt_boxes = self.load_annotations_comma_and_Space_format(index)

        image_meta = {'index': index,
                      'image_id': image_id,
                      'orig_size': np.array(image.shape, dtype=np.int32)}
        
        image, image_visualize, image_meta, gt_boxes, gt_class_ids = self.preprocess(image, image_meta, gt_boxes, gt_class_ids)
        gt = self.prepare_annotations(gt_class_ids, gt_boxes)

        inp = {'image': image,
               'image_meta': image_meta,
               'gt': gt}

        if self.cfg.debug == 1:
            # commented below lines for custom augmentation visualization           
            # if self.cfg.dataset=='yolo':
            #     image_visualize = image_visualize * image_meta['rgb_std'] + image_meta['rgb_mean']

            save_path = os.path.join(self.cfg.debug_dir, image_meta['image_id'] + '.png')         #'_'+ str(random.randint(0,1000)) + '.png')             
            visualize_boxes(image_visualize, gt_class_ids, gt_boxes,
                            class_names=self.class_names,
                            save_path=save_path)

        return inp

    def __len__(self):
        return len(self.sample_ids)

    def preprocess(self, image, image_meta, boxes=None, class_ids=None):
        if self.cfg.dataset=='lpr':
            if self.cfg.forbid_resize:
                image, image_meta, boxes = crop_or_pad(image, image_meta, self.input_size, boxes=boxes)
            else:
                image, image_meta, boxes = resize(image, image_meta, self.input_size, boxes=boxes)    

        if self.phase == "train":
            # # Added below line(s) for custom augmentation. Start point                                                                                                                   
            prob = random.random()
            if boxes is not None:
                if prob < 0.5:                                                                  
                    image = Image.fromarray(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB))                                             
                    image_width = image.width - 1                                        
                    image_height = image.height - 1                                                                                       
                    box = boxes[random.randint(0, len(boxes)-1)]                                                              
                    box_width = box[2]-box[0]                                                          
                    box_height = box[3]-box[1]    
                    a = random.choice([0,1,2])                                                                             
                    if a==0:                                                        
                        tblr = [int(9*box_height), int(4.5*box_height), int(6.5*box_width), int(6.5*box_width)]                                                        
                    elif a==1:                                                        
                        tblr = [int(4.5*box_height), int(2*box_height), int(3*box_width), int(3*box_width)]                                                        
                    elif a==2:                                                        
                        tblr = [int(0.12*box_height), int(0.12*box_height), int(0.2*box_width), int(0.2*box_width)]                                                        
                    x1 = np.max((0, int(box[0]-tblr[2])))                                           
                    y1 = np.max((0, int(box[1]-tblr[0])))                                                 
                    x2 = np.min((int(image_width), int(box[2]+tblr[3])))                                                        
                    y2 = np.min((int(image_height), int(box[3]+tblr[1])))                                                                        
                    crop_img_dim = (y1, image_width-x2, image_height-y2, x1)                                                          
                    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)                                                         
                    boxes_aug = []                                                         
                    for box, label in zip(boxes, class_ids):                                                
                        boxes_aug.append(BoundingBox(box[0],box[1],box[2],box[3], label=label))                                               
                    boxes_augmented = BoundingBoxesOnImage(boxes_aug,shape=image.shape)                                                               

                    augmentation = iaa.Sequential([
                        iaa.Crop(px=crop_img_dim, keep_size=False),
                        ])                                                                                                

                    image_aug, bbs_aug = augmentation(image = image, bounding_boxes=boxes_augmented)
                    bbs_aug = BoundingBoxesOnImage(bbs_aug,shape=image_aug.shape)                                                 
                    bbs_aug = bbs_aug.remove_out_of_image(fully=True).clip_out_of_image()
                    boxes = np.zeros((len(bbs_aug.bounding_boxes),4))
                    class_ids = []                                                       
                    for i in range(len(bbs_aug.bounding_boxes)):
                        boxes[i]= [bbs_aug.bounding_boxes[i].x1,bbs_aug.bounding_boxes[i].y1,
                                    bbs_aug.bounding_boxes[i].x2,bbs_aug.bounding_boxes[i].y2]
                        class_ids.append(bbs_aug.bounding_boxes[i].label)
                    class_ids = np.array(class_ids, dtype=np.int16)
                    image = image_aug.astype(np.float32)
                    if not len(boxes):
                        boxes = None                   
                    # End of added line(s) for custom augmentation                                                                

            # prob = random.random()
            # if boxes is not None:
            #     if prob < 0.6:              # change prob from 0.7 to 0.6                  
            #         image, image_meta, boxes = resize(image, image_meta, (np.max((32,image.shape[0])), np.max((32,image.shape[1]))), boxes=boxes)   #Added line for LPD  
            #         image = Image.fromarray(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB))                                             

            #         # # Start of added line(s) for custom augmentation                                                                        
            #         # image_width = image.width - 1                                        
            #         # image_height = image.height - 1                                                                                       
            #         # box = boxes[random.randint(0, len(boxes)-1)]                                                              
            #         # box_width = box[2]-box[0]                                                          
            #         # box_height = box[3]-box[1]    
            #         # a = random.choice([0,1,2])                                                                             
            #         # if a==0:                                                        
            #         #     tblr = [int(9*box_height), int(4.5*box_height), int(6.5*box_width), int(6.5*box_width)]                                                        
            #         # elif a==1:                                                        
            #         #     tblr = [int(4.5*box_height), int(2*box_height), int(3*box_width), int(3*box_width)]                                                        
            #         # elif a==2:                                                        
            #         #     tblr = [int(0.12*box_height), int(0.12*box_height), int(0.2*box_width), int(0.2*box_width)]                                                        
            #         # x1 = np.max((0, int(box[0]-tblr[2])))                                           
            #         # y1 = np.max((0, int(box[1]-tblr[0])))                                                 
            #         # x2 = np.min((int(image_width), int(box[2]+tblr[3])))                                                        
            #         # y2 = np.min((int(image_height), int(box[3]+tblr[1])))                                                                        
            #         # crop_img_dim = (y1, image_width-x2, image_height-y2, x1)                                                          
            #         # augmentation = iaa.Sequential([
            #         #     iaa.Crop(px=crop_img_dim, keep_size=False),
            #         #     ])                                                                                                
            #         # # End of added line(s) for custom augmentation                                                                     

            #         image = transforms.ColorJitter(brightness=(0.8,1.2), contrast=(0.8, 1.2),
            #                                 saturation=(0.8, 1.2), hue=(-0.2, 0.2)) (image)            
            #         image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            #         boxes_aug = []
            #         for box, label in zip(boxes, class_ids):
            #             boxes_aug.append(BoundingBox(box[0],box[1],box[2],box[3], label=label))
            #         boxes_augmented = BoundingBoxesOnImage(boxes_aug,shape=image.shape)
            #         image_aug, bbs_aug = self.seq(image = image, bounding_boxes=boxes_augmented)
            #         bbs_aug = bbs_aug.remove_out_of_image(fully=True).clip_out_of_image()
            #         # if(random.random()<0.67):                                                             # Added line for LPD                                                           
            #         #     image_aug, bbs_aug = augmentation(image = image_aug, bounding_boxes=bbs_aug)      # Added line for LPD                   
            #         #     bbs_aug = bbs_aug.remove_out_of_image(fully=True).clip_out_of_image()             # Added line for LPD                                      
            #         boxes = np.zeros((len(bbs_aug.bounding_boxes),4))
            #         class_ids = []
            #         for i in range(len(bbs_aug.bounding_boxes)):
            #             boxes[i]= [bbs_aug.bounding_boxes[i].x1,bbs_aug.bounding_boxes[i].y1,
            #                         bbs_aug.bounding_boxes[i].x2,bbs_aug.bounding_boxes[i].y2]
            #             class_ids.append(bbs_aug.bounding_boxes[i].label)
            #         class_ids = np.array(class_ids, dtype=np.int16)
            #         image = image_aug.astype(np.float32)
            #         if not len(boxes):
            #             boxes = None                   
                    

        if self.cfg.dataset=='yolo':
            # # Trajectory Specific
            # drift_prob = self.cfg.drift_prob if self.phase == 'train' else 0.
            # flip_prob = self.cfg.flip_prob if self.phase == 'train' else 0.
            # image, image_meta = whiten(image, image_meta, mean=self.rgb_mean, std=self.rgb_std)
            # image, image_meta, boxes = drift(image, image_meta, prob=drift_prob, boxes=boxes)
            # image, image_meta, boxes = flip(image, image_meta, prob=flip_prob, boxes=boxes)
            # if self.cfg.forbid_resize:
            #     image, image_meta, boxes = crop_or_pad(image, image_meta, self.input_size, boxes=boxes)
            # else:
            #     image, image_meta, boxes = resize(image, image_meta, self.input_size, boxes=boxes) 
       
            
            image, image_meta, boxes = resize(image, image_meta, self.input_size, boxes=boxes)   #Added line for kitti  

            image_visualize = image
            # image = image.transpose(2, 0, 1)


            image, image_meta = whiten(image, image_meta, self.rgb_mean, self.rgb_std)
            # resize the image
            image, image_meta, boxes = resize(image, image_meta, self.input_size, boxes=None)
            image = (image * 2) - 1
            image_visualize = image

        elif self.cfg.dataset=='lpr':
            # LPR Specific
            image = Image.fromarray(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB))
            image_visualize = transforms.Grayscale(num_output_channels=3) (image)
            image = transforms.ToTensor()(image_visualize)
            image_visualize = cv2.cvtColor(np.array(image_visualize), cv2.COLOR_RGB2BGR)

        if boxes is not None:
            # Added below 2 line(s) for LPD                       
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0., self.input_size[1] - 1.)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0., self.input_size[0] - 1.)            
            # boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0., image_meta['orig_size'][1] - 1.)
            # boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0., image_meta['orig_size'][0] - 1.)
            if self.cfg.dataset=='lpr':
                inds = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) >= 16
                boxes = boxes[inds]
                class_ids = class_ids[inds]
                if not len(boxes):
                    boxes = None

        return image, image_visualize, image_meta, boxes, class_ids

    def prepare_annotations(self, class_ids, boxes):
        """
        :param class_ids:
        :param boxes: xyxy format
        :return: np.ndarray(#anchors, #classes + 9)
        """
        gt = np.zeros((self.num_anchors, self.num_classes + 9), dtype=np.float32)
        if boxes is not None:
            deltas, anchor_indices = compute_deltas(boxes, self.anchors)
            gt[anchor_indices, 0] = 1.  # mask
            gt[anchor_indices, 1:5] = boxes
            gt[anchor_indices, 5:9] = deltas
            gt[anchor_indices, 9 + class_ids] = 1.  # class logits

        return gt

    def get_sample_ids(self):
        raise NotImplementedError

    def load_image(self, index):
        raise NotImplementedError

    def load_annotations(self, index):
        raise NotImplementedError

    def save_results(self, results):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError
    
    # def preprocess(self):
    #     raise NotImplementedError
