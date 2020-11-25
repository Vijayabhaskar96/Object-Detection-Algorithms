import YoloV2_model
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
    def __init__(self, weights_path, S=13, B=5, C=20):
        conv_config = namedtuple(
            "ConvConfig", ["kernel_size", "filters", "stride", "pad"]
        )
        maxpool_config = namedtuple("MaxPoolConfig", ["kernel_size", "stride"])
        architechture_config1 = [
            conv_config(3, 32, 1, 1),
            maxpool_config(2, 2),
            conv_config(3, 64, 1, 1),
            maxpool_config(2, 2),
            conv_config(3, 128, 1, 1),
            conv_config(1, 64, 1, 0),
            conv_config(3, 128, 1, 1),
            maxpool_config(2, 2),
            conv_config(3, 256, 1, 1),
            conv_config(1, 128, 1, 0),
            conv_config(3, 256, 1, 1),
            maxpool_config(2, 2),
            conv_config(3, 512, 1, 1),
            conv_config(1, 256, 1, 0),
            conv_config(3, 512, 1, 1),
            conv_config(1, 256, 1, 0),
            conv_config(3, 512, 1, 1),
        ]
        architechture_config2 = [
            maxpool_config(2, 2),
            conv_config(3, 1024, 1, 1),
            conv_config(1, 512, 1, 0),
            conv_config(3, 1024, 1, 1),
            conv_config(1, 512, 1, 0),
            conv_config(3, 1024, 1, 1),
            conv_config(3, 1024, 1, 1),
            conv_config(3, 1024, 1, 1),
        ]
        self.architechture = [architechture_config1, architechture_config2]
        self.output_labels = [
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]
        self.model = YoloV2_model.YoloV2Model(
            architechture=self.architechture, split_size=S, num_boxes=B, num_classes=C
        )
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
        #         r, g, b = orig_img.split()
        #         orig_img = Image.merge("RGB", (b, g, r))
        img_w, img_h = orig_img.size
        resized_tensor = ToTensor()(
            orig_img.resize(size=(416, 416), resample=PIL.Image.BILINEAR)
        ).float()
        resized_tensor = resized_tensor[[2, 1, 0], :].float()
        if len(resized_tensor.shape) == 3:
            inp = resized_tensor.unsqueeze(0)
        pred = self.model(inp)
        pred = self.post_process(pred)
        result_bboxes = self.get_bboxes(inp, pred)
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

    def post_process(self, pred):
        exist_mask = torch.round(torch.sigmoid(pred[..., 4:5]))
        anchor_boxes = torch.tensor(
            [
                [0, 0, 1.3221, 1.73145],
                [0, 0, 3.19275, 4.00944],
                [0, 0, 5.05587, 8.09892],
                [0, 0, 9.47112, 4.84053],
                [0, 0, 11.2364, 10.0071],
            ],
            device=pred.device,
        )
        cell_idx = torch.arange(13, device=pred.device)
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

    def cells_to_boxes(self, cells, S, B):
        boxes = cells.reshape(S * S * B, -1)
        box_confidence = boxes[:, 4].unsqueeze(-1)
        box_class = boxes[:, 5:].argmax(-1).unsqueeze(-1)
        box_coords = boxes[:, :4]
        converted_preds = torch.cat(
            (box_class.float(), box_confidence, box_coords), dim=-1
        )
        return converted_preds

    def get_bboxes(self, x, predictions, iou_threshold=0.5, threshold=0.4, S=13, B=5):
        all_pred_boxes = []

        for idx in range(len(x)):
            bboxes = self.cells_to_boxes(predictions[idx], S, B)
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
                boxes=self.yolo_to_normal(bboxes_alone),
                scores=bboxes_conf,
                idxs=bboxes_idx,
                iou_threshold=iou_threshold,
            )
            nms_boxes = bboxes[nms_boxes_idxs]
            nms_boxes[:, -4:] = nms_boxes[:, -4:] / S
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


filepath = "pretrained_yolov2-voc-model_weights.pth"
inference_pipeline = InferenceModel(weights_path=filepath)
result_pil_image, bboxes = inference_pipeline.predict(
    Path(configs.BASE_DIR) / "VOCdevkit/VOC2007/JPEGImages/000017.jpg"
)
result_pil_image.save("result.png")
