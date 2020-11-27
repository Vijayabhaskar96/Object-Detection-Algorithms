# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## Yolo V1 implementation in Pytorch
#
#
# References:
# * Redmon J, Divvala S, Girshick R, Farhadi A (2016) You only look once: Unified, real-time object detection. Proc IEEE Comput Soc Conf Comput Vis Pattern Recognit 2016-Decem:779–788 . https://doi.org/10.1109/CVPR.2016.91
# * Aladdin Persson's Playlist https://www.youtube.com/playlist?list=PLhhyoLH6Ijfw0TpCTVTNk42NN08H6UvNq
# * R. Padilla, S. L. Netto and E. A. B. da Silva, “A Survey on Performance Metrics for Object-Detection Algorithms,” 2020 International Conference on Systems, Signals and Image Processing (IWSSIP), Niterói, Brazil, 2020, pp. 237–242, doi: 10.1109/IWSSIP48289.2020.9145130.
# PDF available in the next reference repo.
# * https://github.com/rafaelpadilla/Object-Detection-Metrics/

# +
# import os
# kaggle_data = {
#     "username": "ENTER_YOUR_KAGGLE_USERNAME_HERE",
#     "key": "ENTER_YOUR_KAGGLE_KEY_HERE",
# }
# os.environ["KAGGLE_USERNAME"] = kaggle_data["username"]
# os.environ["KAGGLE_KEY"] = kaggle_data["key"]
# # !pip install pytorch-lightning
# # !pip install kaggle
# # !pip install --upgrade albumentations
# # !wget https://raw.githubusercontent.com/pjreddie/darknet/master/scripts/voc_label.py
# # !kaggle datasets download -d vijayabhaskar96/pascal-voc-2007-and-2012
# # !unzip pascal-voc-2007-and-2012.zip
# # %run voc_label.py
# -
from collections import namedtuple
import torch
from torch import nn
import albumentations as A
from albumentations.pytorch.transforms import ToTensor
from dataset import YoloV1DataModule
from utils import get_bboxes, intersection_over_union, mAP
import configs
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

seed_everything(42)


conv_config = namedtuple("ConvConfig", ["kernel_size", "filters", "stride", "pad"])
maxpool_config = namedtuple("MaxPoolConfig", ["kernel_size", "stride"])
repeat_block = namedtuple("Repeat", ["blocks", "n"])
architecture_config = [
    conv_config(7, 64, 2, 3),
    maxpool_config(2, 2),
    conv_config(3, 192, 1, 1),
    maxpool_config(2, 2),
    conv_config(1, 128, 1, 0),
    conv_config(3, 256, 1, 1),
    conv_config(1, 256, 1, 0),
    conv_config(3, 512, 1, 1),
    maxpool_config(2, 2),
    repeat_block([conv_config(1, 256, 1, 0), conv_config(3, 512, 1, 1)], 4),
    conv_config(1, 512, 1, 0),
    conv_config(3, 1024, 1, 1),
    maxpool_config(2, 2),
    repeat_block([conv_config(1, 512, 1, 0), conv_config(3, 1024, 1, 1)], 2),
    conv_config(3, 1024, 1, 1),
    conv_config(3, 1024, 2, 1),
    conv_config(3, 1024, 1, 1),
    conv_config(3, 1024, 1, 1),
]


class YoloLoss(nn.Module):
    """
    Calculate the loss for yolo (v1) model
    """

    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        """
        S is split size of image (in paper 7),
        B is number of boxes (in paper 2),
        C is number of classes (in paper and VOC dataset is 20),
        """
        self.S = S
        self.B = B
        self.C = C

        # These are from Yolo paper, signifying how much we should
        # pay loss for no object (noobj) and the box coordinates (coord)
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # predictions are shaped (BATCH_SIZE, S*S(C+B*5) when inputted
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # Calculate IoU for the two predicted bounding boxes with target bbox
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        # Take the box with highest IoU out of the two prediction
        # Note that bestbox will be indices of 0, 1 for which bbox was best
        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., 20].unsqueeze(3)  # in paper this is Iobj_i

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        # Set boxes with no object in them to 0. We only take out one of the two
        # predictions, which is the one with highest Iou calculated previously.
        box_predictions = exists_box * (
            bestbox * predictions[..., 26:30] + (1 - bestbox) * predictions[..., 21:25]
        )
        box_targets = exists_box * target[..., 21:25]

        # Take sqrt of width, height of boxes to ensure that
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # pred_box is the confidence score for the bbox with highest IoU
        pred_box = (
            bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]
        )
        pred_box = iou_maxes * pred_box
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21]),
        )

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.mse(
            torch.flatten(
                exists_box * predictions[..., :20],
                end_dim=-2,
            ),
            torch.flatten(
                exists_box * target[..., :20],
                end_dim=-2,
            ),
        )

        loss = (
            self.lambda_coord * box_loss  # first two rows in paper
            + object_loss  # third row in paper
            + self.lambda_noobj * no_object_loss  # forth row
            + class_loss  # fifth row
        )

        return loss


class CNNBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, stride, pad):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            bias=False,
        )
        self.batchnorm = nn.BatchNorm2d(filters)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class YoloV1Model(pl.LightningModule):
    def __init__(
        self,
        in_channels=3,
        architecture=None,
        split_size=7,
        num_boxes=2,
        num_classes=20,
    ):
        super(YoloV1Model, self).__init__()
        self.in_channels = in_channels
        self.darknet = self._create_conv(architecture)
        self.fcs = self._create_fcs(split_size, num_boxes, num_classes)
        self.train_transform = A.Compose(
            [
                A.Resize(width=448, height=448),
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=0),
                ToTensor(),
            ],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
        )
        self.test_transform = A.Compose(
            [A.Resize(width=448, height=448), ToTensor()],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
        )

    def forward(self, x):
        x = self.darknet(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fcs(x)
        return x

    def _create_conv(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if "Conv" in str(type(x)):
                layers += [
                    CNNBlock(
                        in_channels=in_channels,
                        filters=x.filters,
                        kernel_size=x.kernel_size,
                        stride=x.stride,
                        pad=x.pad,
                    )
                ]
                in_channels = x.filters

            elif "MaxPool" in str(type(x)):
                layers += [
                    nn.MaxPool2d(
                        kernel_size=(x.kernel_size, x.kernel_size),
                        stride=(x.stride, x.stride),
                    )
                ]

            elif "Repeat" in str(type(x)):
                convs = x.blocks
                num_repeats = x.n

                for _ in range(num_repeats):
                    for conv in convs:
                        layers += [
                            CNNBlock(
                                in_channels,
                                conv.filters,
                                kernel_size=conv.kernel_size,
                                stride=conv.stride,
                                pad=conv.pad,
                            )
                        ]
                        in_channels = conv.filters
        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        self.S, self.B, self.C = split_size, num_boxes, num_classes

        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * self.S * self.S, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, self.S * self.S * (self.C + self.B * 5)),
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=configs.LEARNING_RATE,
            weight_decay=configs.WEIGHT_DECAY,
        )
        return {"optimizer": optimizer}

    def _calc_map(self, x, y, pred):
        pred_boxes, target_boxes = get_bboxes(
            x=x,
            y=y,
            predictions=pred,
            iou_threshold=0.5,
            threshold=0.4,
            S=self.S,
            device=self.device,
        )
        mean_avg_prec = mAP(pred_boxes, target_boxes, iou_threshold=0.5)
        return mean_avg_prec

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred_y = self(x)
        loss = YoloLoss()(pred_y, y)
        self.log("train_loss", loss, prog_bar=True)
        with torch.no_grad():
            mAP = self._calc_map(x, y, pred_y)
            self.log("train_mAP", mAP, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred_y = self(x)
        loss = YoloLoss()(pred_y, y)
        self.log("valid_loss", loss, prog_bar=True)
        mAP = self._calc_map(x, y, pred_y)
        self.log("valid_mAP", mAP, prog_bar=True)
        return loss


if __name__ == "__main__":
    model = YoloV1Model(
        architecture=architecture_config, split_size=7, num_boxes=2, num_classes=20
    )
    data = YoloV1DataModule()
    trainer = pl.Trainer(gpus=1, max_epochs=1000)
    trainer.fit(model, datamodule=data)
