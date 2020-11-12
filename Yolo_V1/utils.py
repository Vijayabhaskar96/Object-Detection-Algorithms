import torch
import numpy as np
from collections import Counter
from torchvision.ops.boxes import batched_nms,box_iou

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)
def convert_cellboxes(predictions, S, device):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """

    # predictions = predictions.cpu()
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)

    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat((predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0)
    
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1).to(device)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]

    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(-1)

    converted_preds = torch.cat((predicted_class.float(), best_confidence, converted_bboxes), dim=-1)
    return converted_preds


def calculate_ap(precisions, recalls):
    precisions = torch.cat((torch.tensor([1]), precisions))
    recalls = torch.cat((torch.tensor([0]), recalls))
    return torch.trapz(precisions, recalls)
def mAP(pred_boxes, true_boxes, iou_threshold=0.5, num_classes=20):
    if len(pred_boxes)==0 or len(true_boxes) == 0:
        return 0
    # list storing all AP for respective classes
    average_precisions = []

    pred_boxes = torch.tensor(pred_boxes)
    true_boxes = torch.tensor(true_boxes)
    # used for numerical stability later on
    epsilon = 1e-10
    for c in range(num_classes):
        detections = pred_boxes[pred_boxes[:,1]==c]
        ground_truths = true_boxes[true_boxes[:,1]==c]
        amount_bboxes = Counter(ground_truths[:,0].int().tolist())
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)
        detections = sorted(detections,key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue
        for detection_idx, detection in enumerate(detections):
            ground_truth_img = ground_truths[ground_truths[:,0] == detection[0]]
            num_gts = len(ground_truth_img)
            if num_gts==0:
                best_iou = 0
            else:
                ious = box_iou(yolo_to_normal(detection[-4:].unsqueeze(0)), yolo_to_normal(ground_truth_img[:,-4:]))
                best_iou, best_gt_idx = ious.max(1)
                best_iou, best_gt_idx = best_iou[0], best_gt_idx[0]
            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0].int().item()][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0].int().item()][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.div(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        average_precisions.append(calculate_ap(precisions, recalls))
    return sum(average_precisions) / len(average_precisions)

def yolo_to_normal(boxes):
    """
    Converts [Xmid,Ymid,W,H] format to [Xmin,Ymin,Xmax,Ymax]
    Arguments:
     boxes: N x [Xmid,Ymid,W,H]
    returns:
     boxes: N x [Xmin,Ymin,Xmax,Ymax]
    """
    Xmid, Ymid, W, H = [boxes[:,i] for i in range(4)]
    #Didn't add 1 
    Xmin = Xmid - (W/2)
    Ymin = Ymid - (H/2)
    Xmax = Xmid + (W/2)
    Ymax = Ymid + (H/2)
    return torch.stack([Xmin,Ymin,Xmax,Ymax],dim=-1)

def get_bboxesmine(x,y,predictions,
    iou_threshold,
    threshold,
    S,device):
    all_pred_boxes = []
    all_true_boxes = []

    true_bboxes = convert_cellboxes(y,S,device).reshape(y.shape[0], S * S, -1)
    bboxes = convert_cellboxes(predictions,S,device).reshape(predictions.shape[0], S * S, -1)
    train_idx = 0
    for idx in range(len(x)):
        chosen_bboxes = bboxes[idx,bboxes[idx,:,1]>threshold]
        if len(chosen_bboxes)==0:
            continue
        bboxes_idx, bboxes_conf, bboxes_alone = chosen_bboxes[:,0], chosen_bboxes[:,1], chosen_bboxes[:,2:]
        nms_boxes_idxs = batched_nms(boxes=yolo_to_normal(bboxes_alone), scores=bboxes_conf,idxs=bboxes_idx,iou_threshold=iou_threshold)
        nms_boxes = chosen_bboxes[nms_boxes_idxs]
        idx_arr = torch.repeat_interleave(torch.tensor(idx),torch.tensor(len(nms_boxes))).unsqueeze(1).to(device)
        nms_boxes = torch.cat([idx_arr, nms_boxes],dim=1)
        true_bboxes2 = true_bboxes[idx,true_bboxes[idx,:,1]>threshold]
        idx_arr = torch.repeat_interleave(torch.tensor(idx),torch.tensor(len(true_bboxes2))).unsqueeze(1).to(device)
        true_bboxes2 = torch.cat([idx_arr, true_bboxes2],dim=1)
        all_pred_boxes.append(nms_boxes)
        all_true_boxes.append(true_bboxes2)
        train_idx += 1
    if all_pred_boxes:
        x = torch.cat(all_pred_boxes).tolist()
    else:
        x = []
    if all_true_boxes:
        y = torch.cat(all_true_boxes).tolist() 
    else:
        y = []
    return x,y