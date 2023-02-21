import torch                                                                                                                                                                                                   
from collections import Counter                                                                                                    
import os                       
                                               
# Change below paths to compute mAP for different boxes according to your requirements                                                                             
base_path = '/home/hazen/Documents/PROJECTS/2.ObjDetSD/SqueezeDet-PyTorch'                                          
gt_boxes_idx_path = os.path.join(base_path, 'data/kitti/image_sets/val.txt')                                              
gt_boxes_path = os.path.join(base_path, 'data/kitti/training/label_2')                                                                        
pred_boxes_path = os.path.join(base_path, 'exp/default/results/data')                                                                                                                     

def iou_calc(pred_bbox, gt_bbox):                                                                                                                                          
    pred_x1, pred_y1, pred_x2, pred_y2 = pred_bbox                                                     
    gt_x1, gt_y1, gt_x2, gt_y2 = gt_bbox                                                          
                                                                                                                                                              
    x1 = max(pred_x1, gt_x1)                                                                                                                                 
    y1 = max(pred_y1, gt_y1)                                                                            
    x2 = min(pred_x2, gt_x2)                                                                             
    y2 = min(pred_y2, gt_y2)                                                              
                                                                                                                        
    intersection = max(0, x2-x1+1) * max(0, y2-y1+1)                    
                                                                                                                       
    pred_area = (pred_x2-pred_x1+1) * (pred_y2-pred_y1+1)                                                       
    gt_area = (gt_x2-gt_x1+1) * (gt_y2-gt_y1+1)                                     
    union = pred_area + gt_area - intersection                                        

    if union==0:                                                                                                               
        return 0                                                                              
    else:                                                                       
        return intersection/union                                                       
                                                                                                                                      
num_classes = ('licenseplate','car')                                                                                                       
iou_threshold = 0.8                                                                                                                                 
average_precisions = []                                                                                                     
epsilon = 1e-6                                                                           
with open(gt_boxes_idx_path, 'r') as fp:
    names = fp.readlines()

names = [name.strip() for name in names]                                                                                                                                                                                                              
                                                     
pred_boxes, true_boxes = [], []                                                                                                                           
                                                                                                   
for name in names:                                                                                                                 
    file_path = os.path.join(gt_boxes_path, name+'.txt')                                                                                  
    with open(file_path, 'r') as fp:
        annotations = fp.readlines()
    annotations = [ann.strip().split(' ') for ann in annotations]                                                     
    for ann in annotations:                                                
        box = [float(x) for x in ann[4:8]]                                                                  
        true_boxes.append([name, ann[0].lower(), box[0], box[1], box[2], box[3]])                                  
                                                                                        
for name in os.listdir(pred_boxes_path):                                                                                                                                                    
    if name.endswith('.txt'):                                                                                         
        with open(os.path.join(pred_boxes_path,name), 'r') as fp:                                                                
            annotations = fp.readlines()                                                                              
    annotations = [ann.strip().split(' ') for ann in annotations]                                                     
    for ann in annotations:                                                
        box = [float(x) for x in ann[4:8]]                                                                  
        pred_boxes.append([name.split('.')[0], ann[0].lower(), float(ann[-1]), box[0], box[1], box[2], box[3]])                                                                                             


for c in num_classes:                                                                                                                                           
    detections, ground_truths = [], []                                                                              
                                                                                                                                         
    for detection in pred_boxes:
        if detection[1] == c:                                                                                                              
            detections.append(detection)                                                                                                      

    for true_box in true_boxes:                                                                                                
        if true_box[1] == c:                                                                                                                        
            ground_truths.append(true_box)                                                                                                                                     
    
    amount_bboxes = Counter([gt[0] for gt in ground_truths])                                                                             
    for key, val in amount_bboxes.items():                                                                                                                       
        amount_bboxes[key] = torch.zeros(val)                                                                                              
                                                                                                 
    detections.sort(key=lambda x:x[2], reverse=True)                                                                                                                               
    TP = torch.zeros((len(detections)))                                                                                        
    FP = torch.zeros((len(detections)))                                                                                        
    total_true_bboxes = len(ground_truths)                                                                                                      
                                                                                                                                     
    for detection_idx, detection in enumerate(detections):                                                                              
        ground_truth_img = [ bbox for bbox in ground_truths if bbox[0]==detection[0] ]                                                                                                                                 
                                                                                               
        num_gts = len(ground_truth_img)                                                                                       
        best_iou = 0                                                                               
                                                                                                                             
        for idx, gt in enumerate(ground_truth_img):                                                                       
                                                                                                            
            iou = iou_calc(torch.tensor(detection[3:]), torch.tensor(gt[2:]))                                                                                      
                                                                                                                           
            if iou>best_iou:                                                               
                best_iou=iou                                                                        
                best_gt_idx = idx                                                                     

        if best_iou >= iou_threshold:                                                                         
            if amount_bboxes[detection[0]][best_gt_idx] == 0:                                                                           
                TP[detection_idx] = 1                                                        
                amount_bboxes[detection[0]][best_gt_idx] = 1                                                                         
            else:                                                                                               
                FP[detection_idx] = 1                                                                  
        else:                                                                    
            FP[detection_idx] = 1                                                                               

    TP_cumsum = torch.cumsum(TP, dim=0)                                                                           
    FP_cumsum = torch.cumsum(FP, dim=0)                                                                           

    recalls = TP_cumsum/(total_true_bboxes+epsilon)                                                          
    precisions = torch.divide(TP_cumsum, (TP_cumsum+FP_cumsum+epsilon))                                                                             
    precisions = torch.cat(( torch.tensor([1]), precisions ))                                                                           
    recalls = torch.cat(( torch.tensor([0]), recalls ))                                                                           
    average_precisions.append(torch.trapz(precisions, recalls))                                                     
    break      # Add break statement as we only have one class in this case i-e licenseplate                                                                                                               
                                                                                                                        
mAP = sum(average_precisions)/len(average_precisions)                                                             
print(mAP*100)                                                                                                  