import YoloV3_model
from collections import namedtuple
import numpy as np
import torch
from copy import deepcopy
from PIL import Image, ImageDraw, ImageFont
import PIL
from torchvision.ops.boxes import batched_nms
from torchvision.transforms import ToTensor
import random
import configs
from pathlib import Path


class InferenceModel:
    def __init__(self, weights_path):
        self.output_labels = []
        with open("./coco.names", "r") as f:
            for _, line in enumerate(f.readlines()):
                self.output_labels.append(line.strip())
        self.anchor_boxes = torch.tensor(
            [
                [0, 0, 10, 13],
                [0, 0, 16, 30],
                [0, 0, 33, 23],
                [0, 0, 30, 61],
                [0, 0, 62, 45],
                [0, 0, 59, 119],
                [0, 0, 116, 90],
                [0, 0, 156, 198],
                [0, 0, 373, 326],
            ]
        )
        self.model = YoloV3_model.YoloV3Model(num_attrib=5 + configs.NUM_CLASSES)
        self.load_weigts(weights_path)

    def load_weigts(self, weights_path):
        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()
        print("Weights Loaded!")

    def yolo_to_normal(self, boxes):
        Xmid, Ymid, W, H = [boxes[:, i] for i in range(4)]
        # Didn't add 1, because we dont have image height and width here.
        # doesn't matter much
        Xmin = Xmid - (W / 2)
        Ymin = Ymid - (H / 2)
        Xmax = Xmid + (W / 2)
        Ymax = Ymid + (H / 2)
        return torch.stack([Xmin, Ymin, Xmax, Ymax], dim=-1)

    def predict(self, filepath):
        final_bboxes = []
        orig_img = Image.open(filepath)
        result_img = deepcopy(orig_img)
        img_w, img_h = orig_img.size
        resized_tensor = ToTensor()(
            orig_img.resize(size=(416, 416), resample=PIL.Image.BILINEAR)
        ).float()
        inp = resized_tensor[[2, 1, 0], :].float()
        inp = resized_tensor.float()
        if len(inp.shape) == 3:
            inp = resized_tensor.unsqueeze(0)
        pred = self.model(inp)
        pred = self.post_process(pred)
        result_bboxes = self.get_bboxes(
            predictions=pred, S=[13, 26, 52], B=3, device=inp.device
        )
        for bbox in result_bboxes:
            c, conf, bb = self.output_labels[int(bbox[0])], bbox[1], bbox[-4:]
            xcenter, ycenter, bb_w, bb_h = bb
            xmin = int(np.round((xcenter - bb_w / 2) * img_w)) + 1
            ymin = int(np.round((ycenter - bb_h / 2) * img_h)) + 1
            xmax = int(np.round((xcenter + bb_w / 2) * img_w)) + 1
            ymax = int(np.round((ycenter + bb_h / 2) * img_h)) + 1
            self.draw_bb(result_img, c + " " + str(conf)[:5], (xmin, ymin, xmax, ymax))
            final_bboxes.append([c, conf, (xmin, ymin, xmax, ymax)])

        return result_img, final_bboxes

    def _preprocess(self, pred, anchor_boxes, S):
        anchor_boxes = anchor_boxes.to(pred.device)
        exist_mask = torch.round(torch.sigmoid(pred[..., 4:5]))
        cell_idx = torch.arange(S, device=pred.device)
        bx = exist_mask * torch.sigmoid(pred[..., 0:1]) + exist_mask * cell_idx.view(
            [1, 1, -1, 1, 1]
        )
        by = exist_mask * torch.sigmoid(pred[..., 1:2]) + exist_mask * cell_idx.view(
            [1, -1, 1, 1, 1]
        )
        bw = (
            exist_mask
            * anchor_boxes[:, 2].view([1, 1, 1, -1, 1])
            * exist_mask
            * torch.exp(pred[..., 2:3])
        )
        bh = (
            exist_mask
            * anchor_boxes[:, 3].view([1, 1, 1, -1, 1])
            * exist_mask
            * torch.exp(pred[..., 3:4])
        )
        pred[..., :4] = torch.cat([bx, by, bw, bh], dim=-1)
        return pred

    def post_process(self, pred):
        small_preprocessed_pred = self._preprocess(
            pred[0], self.anchor_boxes[6:9] / (416 / 13), S=13
        )
        medium_preprocessed_pred = self._preprocess(
            pred[1], self.anchor_boxes[3:6] / (416 / 26), S=26
        )
        large_preprocessed_pred = self._preprocess(
            pred[2], self.anchor_boxes[:3] / (416 / 52), S=52
        )
        return (
            small_preprocessed_pred,
            medium_preprocessed_pred,
            large_preprocessed_pred,
        )

    def cells_to_boxes(self, cells, S, B):
        boxes = cells.reshape(S * S * B, -1)
        box_confidence = boxes[:, 4].unsqueeze(-1)
        box_class = boxes[:, 5:].argmax(-1).unsqueeze(-1)
        box_coords = boxes[:, :4]
        converted_preds = torch.cat(
            (box_class.float(), box_confidence, box_coords), dim=-1
        )
        return converted_preds

    def get_bboxes(
        self,
        predictions,
        iou_threshold=0.5,
        threshold=0.5,
        S=None,
        B=None,
        device="cpu",
    ):
        all_pred_boxes = []

        for idx in range(len(predictions[0])):
            all_bboxes = []
            for i, grid_size in enumerate(S):
                bboxes = self.cells_to_boxes(predictions[i][idx], grid_size, B)
                if len(bboxes) == 0:
                    continue
                bboxes[:, 1] = torch.sigmoid(bboxes[:, 1])
                bboxes = bboxes[bboxes[:, 1] > threshold]
                bboxes[:, 2:] /= grid_size
                all_bboxes.append(bboxes)
            if len(all_bboxes) == 0:
                continue
            all_bboxes = torch.cat(all_bboxes)
            bboxes_idx, bboxes_conf, bboxes_alone = (
                all_bboxes[:, 0],
                all_bboxes[:, 1],
                all_bboxes[:, 2:],
            )
            nms_boxes_idxs = batched_nms(
                boxes=self.yolo_to_normal(bboxes_alone),
                scores=bboxes_conf,
                idxs=bboxes_idx,
                iou_threshold=iou_threshold,
            )
            nms_boxes = all_bboxes[nms_boxes_idxs]
            all_pred_boxes.append(nms_boxes)
        if all_pred_boxes:
            x = torch.cat(all_pred_boxes).tolist()
        else:
            x = []
        return x

    def random_color(self):
        return (random.randint(0, 200), random.randint(0, 200), random.randint(0, 200))

    def draw_bb(self, source_img, c, bb):
        draw = ImageDraw.Draw(source_img)
        xmin, ymin, xmax, ymax = bb
        color = self.random_color()
        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=color, width=2)
        draw.text(
            (xmin + 3, ymin + 1),
            c,
            fill=color,
            stroke_fill=(255, 255, 255),
            stroke_width=2,
            font=ImageFont.truetype("../arial.ttf"),
        )


filepath = "pretrained_yolov3-coco-model_weights.pth"
inference_pipeline = InferenceModel(weights_path=filepath)
result_pil_image, bboxes = inference_pipeline.predict(
    Path(configs.BASE_DIR) / "VOCdevkit/VOC2007/JPEGImages/000017.jpg"
)
result_pil_image.save("result.png")
print(bboxes)
