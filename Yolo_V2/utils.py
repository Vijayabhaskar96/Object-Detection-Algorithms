import torch
import numpy as np
from collections import Counter
from torchvision.ops.boxes import batched_nms, box_iou
from collections import namedtuple


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


def ElevenPointInterpolatedAP(rec, pre):
    """
    Calcualtes 11-point interpolated Average Precision

    Parameters:
        rec: Recall
            type: tensor
            shape: 1D tensor
        prec: Precision
            type: tensor
            shape: 1D tensor
    Result:
        ap: Average Precision
        type: float
    """
    recallValues = reversed(torch.linspace(0, 1, 11, dtype=torch.float64))
    rhoInterp = []
    # For each recallValues (0, 0.1, 0.2, ... , 1)
    for r in recallValues:
        # Obtain all indexs of recall values higher or equal than r
        GreaterRecallsIndices = torch.nonzero((rec >= r), as_tuple=False).flatten()
        pmax = 0
        # If there are recalls above r
        if GreaterRecallsIndices.nelement() != 0:
            # Choose the max precision value, from position min(GreaterRecallsIndices) to the end
            pmax = max(pre[GreaterRecallsIndices.min() :])
        #         print(r,pmax,GreaterRecallsIndices)
        rhoInterp.append(pmax)
    # By definition AP = sum(max(precision whose recall is above r))/11
    ap = sum(rhoInterp) / 11
    return ap


def AllPointInterpolatedAP(rec, prec):
    """
    Calcualtes All-point interpolated Average Precision

    Parameters:
        rec: Recall
            type: tensor
            shape: 1D tensor
        prec: Precision
            type: tensor
            shape: 1D tensor
    Result:
        ap: Average Precision
        type: float
    """

    mrec = torch.cat([torch.tensor([0]), rec, torch.tensor([1])])
    mpre = torch.cat([torch.tensor([0]), prec, torch.tensor([0])])
    # Find maximum of precision at each position and it's next position
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    # Find indices where recall value changes
    for i in range(len(mrec) - 1):
        if mrec[i] != mrec[i + 1]:
            ii.append(i + 1)
    ap = 0
    # Calculate area at each point recall changes
    for i in ii:
        width = (
            mrec[i] - mrec[i - 1]
        )  # Since x axis is recall this can be thought as width
        height = mpre[i]  # Since y axis is precision this can be thought as height
        area = width * height
        ap += area
    return ap


def calculate_ap(precisions, recalls):
    # precisions = torch.cat((torch.tensor([1]), precisions))
    # recalls = torch.cat((torch.tensor([0]), recalls))
    # return torch.trapz(precisions, recalls)
    return ElevenPointInterpolatedAP(recalls, precisions)


def mAP(pred_boxes, true_boxes, iou_threshold=0.5, num_classes=20):
    if len(pred_boxes) == 0 or len(true_boxes) == 0:
        return 0
    # list storing all AP for respective classes
    average_precisions = []
    pred_boxes = torch.tensor(pred_boxes)
    true_boxes = torch.tensor(true_boxes)
    # used for numerical stability later on
    epsilon = 1e-10
    for c in range(num_classes):
        detections = pred_boxes[pred_boxes[:, 1] == c]
        ground_truths = true_boxes[true_boxes[:, 1] == c]
        amount_bboxes = Counter(ground_truths[:, 0].int().tolist())
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)
        detections = sorted(detections, key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue
        for detection_idx, detection in enumerate(detections):
            ground_truth_img = ground_truths[ground_truths[:, 0] == detection[0]]
            num_gts = len(ground_truth_img)
            if num_gts == 0:
                best_iou = 0
            else:
                ious = box_iou(
                    yolo_to_normal(detection[-4:].unsqueeze(0)),
                    yolo_to_normal(ground_truth_img[:, -4:]),
                )
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
    Xmid, Ymid, W, H = [boxes[:, i] for i in range(4)]
    # Didn't add 1
    Xmin = Xmid - (W / 2)
    Ymin = Ymid - (H / 2)
    Xmax = Xmid + (W / 2)
    Ymax = Ymid + (H / 2)
    return torch.stack([Xmin, Ymin, Xmax, Ymax], dim=-1)


def cells_to_boxes(cells, S, B):
    boxes = cells.reshape(S * S * B, -1)
    box_confidence = boxes[:, 4].unsqueeze(-1)
    box_class = boxes[:, 5:].argmax(-1).unsqueeze(-1)
    box_coords = boxes[:, :4]
    converted_preds = torch.cat((box_class.float(), box_confidence, box_coords), dim=-1)
    return converted_preds


def get_bboxesmine(x, y, predictions, iou_threshold, threshold, S, B, device):
    all_pred_boxes = []
    all_true_boxes = []

    for idx in range(len(x)):
        true_bboxes = cells_to_boxes(y[idx], S, B)
        bboxes = cells_to_boxes(predictions[idx], S, B)
        # ious = intersection_over_union(predictions[idx][...,:4],y[idx][...,:4].float()).flatten()
        bboxes[:, 1] = torch.sigmoid(bboxes[:, 1])
        bboxes = bboxes[bboxes[:, 1] > threshold]
        bboxes_idx, bboxes_conf, bboxes_alone = (
            bboxes[:, 0],
            bboxes[:, 1],
            bboxes[:, 2:],
        )
        if len(bboxes) == 0:
            continue
        nms_boxes_idxs = batched_nms(
            boxes=yolo_to_normal(bboxes_alone),
            scores=bboxes_conf,
            idxs=bboxes_idx,
            iou_threshold=iou_threshold,
        )
        nms_boxes = bboxes[nms_boxes_idxs]
        idx_arr = torch.full(
            size=(len(nms_boxes), 1), fill_value=float(idx), device=device
        )
        nms_boxes = torch.cat([idx_arr, nms_boxes], dim=1)

        true_bboxes2 = true_bboxes[true_bboxes[:, 0] > threshold]
        # idx_arr = torch.repeat_interleave(torch.tensor(idx),torch.tensor(len(true_bboxes2))).unsqueeze(1).to(device)
        idx_arr = torch.full(
            size=(len(true_bboxes2), 1), fill_value=float(idx), device=device
        )
        true_bboxes2 = torch.cat([idx_arr, true_bboxes2], dim=1)
        all_pred_boxes.append(nms_boxes)
        all_true_boxes.append(true_bboxes2)
    if all_pred_boxes:
        x = torch.cat(all_pred_boxes).tolist()
    else:
        x = []
    if all_true_boxes:
        y = torch.cat(all_true_boxes).tolist()
    else:
        y = []
    return x, y


class WeightReader:
    def __init__(self, weight_file):
        self.offset = 5
        self.all_weights = np.fromfile(weight_file, dtype=np.float32)
        print(f"Weights length:{len(self.all_weights)} offset:{self.offset}")

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset - size : self.offset]

    def reset(self):
        self.offset = 5


def load_conv_block(block, wr, with_bn=True):
    if with_bn:
        num_bn_biases = block.batchnorm.bias.numel()

        # Load the weights
        bn_biases = torch.from_numpy(wr.read_bytes(num_bn_biases))
        bn_weights = torch.from_numpy(wr.read_bytes(num_bn_biases))
        bn_running_mean = torch.from_numpy(wr.read_bytes(num_bn_biases))
        bn_running_var = torch.from_numpy(wr.read_bytes(num_bn_biases))

        # Cast the loaded weights into dims of model weights.
        bn_biases = bn_biases.view_as(block.batchnorm.bias.data)
        bn_weights = bn_weights.view_as(block.batchnorm.weight.data)
        bn_running_mean = bn_running_mean.view_as(block.batchnorm.running_mean)
        bn_running_var = bn_running_var.view_as(block.batchnorm.running_var)

        # Copy the data to model
        block.batchnorm.bias.data.copy_(bn_biases)
        block.batchnorm.weight.data.copy_(bn_weights)
        block.batchnorm.running_mean.copy_(bn_running_mean)
        block.batchnorm.running_var.copy_(bn_running_var)
        # print("Applied Weights for CNNBlock.",wr.offset)
    else:
        # DummyBlock to add a fake .conv attribute
        DummyBlock = namedtuple("DummyBlock", ["conv"])
        block = DummyBlock(block)

        # Number of biases
        num_biases = block.conv.bias.numel()

        # Load the weights
        conv_biases = torch.from_numpy(wr.read_bytes(num_biases))

        # reshape the loaded weights according to the dims of the model weights
        conv_biases = conv_biases.view_as(block.conv.bias.data)

        # Finally copy the data
        block.conv.bias.data.copy_(conv_biases)
        # print("Applied Weights for CNN Layer.",wr.offset)

    # Let us load the weights for the Convolutional layers
    num_weights = block.conv.weight.numel()

    # Do the same as above for weights
    conv_weights = torch.from_numpy(wr.read_bytes(num_weights))

    conv_weights = conv_weights.view_as(block.conv.weight.data)
    block.conv.weight.data.copy_(conv_weights)
