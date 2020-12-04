from collections import namedtuple
import torch
from torch import nn
from torch.nn import functional as F
from dataset import YoloV3DataModule
from utils import get_bboxes, intersection_over_union, mAP
import configs
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

conv_config = namedtuple("ConvConfig", ["kernel_size", "filters", "stride", "pad"])
conv_without_bn_config = namedtuple(
    "ConvWithoutBnConfig", ["kernel_size", "filters", "stride", "pad"]
)
repeat_res_block = namedtuple("RepeatWithResidual", ["blocks", "n"])
repeat_block = namedtuple("Repeat", ["blocks", "n"])
darknet_base_config = [
    [
        conv_config(3, 32, 1, 1),
        conv_config(3, 64, 2, 1),
        repeat_res_block([conv_config(1, 32, 1, 0), conv_config(3, 64, 1, 1)], 1),
        conv_config(3, 128, 2, 1),
        repeat_res_block([conv_config(1, 64, 1, 0), conv_config(3, 128, 1, 1)], 2),
    ],
    [
        conv_config(3, 256, 2, 1),
        repeat_res_block([conv_config(1, 128, 1, 0), conv_config(3, 256, 1, 1)], 8),
    ],
    [
        conv_config(3, 512, 2, 1),
        repeat_res_block([conv_config(1, 256, 1, 0), conv_config(3, 512, 1, 1)], 8),
    ],
    [
        conv_config(3, 1024, 2, 1),
        repeat_res_block([conv_config(1, 512, 1, 0), conv_config(3, 1024, 1, 1)], 4),
    ],
]
small_scale_config = [
    [repeat_block([conv_config(1, 512, 1, 0), conv_config(3, 1024, 1, 1)], 2)],
    [conv_config(1, 512, 1, 0)],
    [
        conv_config(3, 1024, 1, 1),
        conv_without_bn_config(1, 3 * (configs.NUM_CLASSES + 5), 1, 0),
    ],
]
medium_scale_config = [
    [repeat_block([conv_config(1, 256, 1, 0), conv_config(3, 512, 1, 1)], 2)],
    [conv_config(1, 256, 1, 0)],
    [
        conv_config(3, 512, 1, 1),
        conv_without_bn_config(1, 3 * (configs.NUM_CLASSES + 5), 1, 0),
    ],
]
large_scale_config = [
    repeat_block([conv_config(1, 128, 1, 0), conv_config(3, 256, 1, 1)], 3),
    conv_without_bn_config(1, 3 * (configs.NUM_CLASSES + 5), 1, 0),
]


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


class DoubleConvWithResidual(nn.Module):
    def __init__(self, repeat_block, in_channels):
        super(DoubleConvWithResidual, self).__init__()
        conv = repeat_block.blocks[0]
        self.conv1 = CNNBlock(
            in_channels,
            conv.filters,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            pad=conv.pad,
        )
        in_channels = conv.filters
        conv = repeat_block.blocks[1]
        self.conv2 = CNNBlock(
            in_channels,
            conv.filters,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            pad=conv.pad,
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return torch.add(x, out)


def conv_upsample_block(in_channels, filters, kernel_size=1, stride=1, pad=0):
    conv = CNNBlock(
        in_channels=in_channels,
        filters=filters,
        kernel_size=kernel_size,
        stride=stride,
        pad=pad,
    )
    upsample = nn.Upsample(scale_factor=2, mode="nearest")
    return nn.Sequential(conv, upsample)


def _create_conv(architecture, in_channels):
    layers = []
    for x in architecture:
        if "ConvConfig" in str(type(x)):
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
        elif "ConvWithoutBnConfig" in str(type(x)):
            layers += [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=x.filters,
                    kernel_size=x.kernel_size,
                    stride=x.stride,
                    padding=x.pad,
                )
            ]
        elif "RepeatWithResidual" in str(type(x)):
            for _ in range(x.n):
                layers += [DoubleConvWithResidual(x, in_channels=in_channels)]
                in_channels = x.blocks[-1].filters
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


class DarkNet53(nn.Module):
    def __init__(self, architecture):
        super(DarkNet53, self).__init__()
        self.part1 = _create_conv(architecture[0], 3)
        self.part2 = _create_conv(architecture[1], 128)
        self.part3 = _create_conv(architecture[2], 256)
        self.part4 = _create_conv(architecture[3], 512)

    def forward(self, x):
        x = self.part1(x)
        large_out = self.part2(x)
        medium_out = self.part3(large_out)
        small_out = self.part4(medium_out)
        return small_out, medium_out, large_out


class YoloV3Tail(nn.Module):
    def __init__(
        self,
        small_part_config,
        medium_part_config,
        large_part_config,
        num_anchors,
        num_attrib,
    ):
        super(YoloV3Tail, self).__init__()
        self.num_anchors = num_anchors
        self.num_attrib = num_attrib
        self.small_part_first = _create_conv(small_part_config[0], 1024)
        self.small_part_branch = _create_conv(small_part_config[1], 1024)
        self.small_part_end = _create_conv(small_part_config[2], 512)
        self.small_part_branch_net = conv_upsample_block(
            in_channels=512, filters=256, kernel_size=1, stride=1
        )

        self.medium_part_first = _create_conv(medium_part_config[0], 768)
        self.medium_part_branch = _create_conv(medium_part_config[1], 512)
        self.medium_part_end = _create_conv(medium_part_config[2], 256)
        self.medium_part_branch_net = conv_upsample_block(
            in_channels=256, filters=128, kernel_size=1, stride=1
        )

        self.large_part = _create_conv(large_part_config, 384)

    def forward(self, small_out, medium_out, large_out):
        x = self.small_part_first(small_out)
        sbranch = self.small_part_branch(x)
        tail_small_out = self.small_part_end(sbranch)
        b, _, w, h = tail_small_out.shape
        tail_small_out = tail_small_out.permute(0, 2, 3, 1).view(
            b, w, h, self.num_anchors, self.num_attrib
        )

        x = self.small_part_branch_net(sbranch)
        x = torch.cat([x, medium_out], dim=1)
        x = self.medium_part_first(x)
        mbranch = self.medium_part_branch(x)
        tail_medium_out = self.medium_part_end(mbranch)
        b, _, w, h = tail_medium_out.shape
        tail_medium_out = tail_medium_out.permute(0, 2, 3, 1).view(
            b, w, h, self.num_anchors, self.num_attrib
        )

        x = self.medium_part_branch_net(mbranch)
        x = torch.cat([x, large_out], dim=1)
        tail_large_out = self.large_part(x)
        b, _, w, h = tail_large_out.shape
        tail_large_out = tail_large_out.permute(0, 2, 3, 1).view(
            b, w, h, self.num_anchors, self.num_attrib
        )

        return tail_small_out, tail_medium_out, tail_large_out


# +
class YoloV2Loss(nn.Module):
    """
    Calculate the loss for yolo (v2) model
    """

    def __init__(self, anchor_boxes, S):
        super(YoloV2Loss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.anchor_boxes = anchor_boxes
        self.S = S

    def forward(self, predictions, target, device):
        self.anchor_boxes = self.anchor_boxes.to(device)
        exist_mask = target[..., 4:5]
        existing_boxes = exist_mask * predictions
        cell_idx = torch.arange(self.S, device=device)
        bx = exist_mask * torch.sigmoid(
            predictions[..., 0:1]
        ) + exist_mask * cell_idx.view([1, 1, -1, 1, 1])
        by = exist_mask * torch.sigmoid(
            predictions[..., 1:2]
        ) + exist_mask * cell_idx.view([1, -1, 1, 1, 1])
        bw = (
            exist_mask
            * self.anchor_boxes[:, 2].view([1, 1, 1, -1, 1])
            * exist_mask
            * torch.exp(predictions[..., 2:3])
        )
        bh = (
            exist_mask
            * self.anchor_boxes[:, 3].view([1, 1, 1, -1, 1])
            * exist_mask
            * torch.exp(predictions[..., 3:4])
        )

        ious = intersection_over_union(
            torch.cat([bx, by, bw, bh], dim=-1), target[..., :4]
        )

        xy_loss = self.mse(torch.cat([bx, by], dim=-1), target[..., :2])
        bwbh = torch.cat([bw, bh], dim=-1)
        wh_loss = self.mse(
            torch.sqrt(torch.abs(bwbh) + 1e-32),
            torch.sqrt(torch.abs(target[..., 2:4]) + 1e-32),
        )
        obj_loss = self.mse(
            exist_mask, exist_mask * ious * torch.sigmoid(existing_boxes[..., 4:5])
        )
        no_obj_loss = self.mse(
            (1 - exist_mask),
            (
                ((1 - exist_mask) * (1 - torch.sigmoid(predictions[..., 4:5])))
                * ((ious.max(-1)[0] < 0.6).int().unsqueeze(-1))
            ),
        )
        class_loss = F.nll_loss(
            (exist_mask * F.log_softmax(predictions[..., 5:], dim=-1)).flatten(
                end_dim=-2
            ),
            target[..., 5:].flatten(end_dim=-2).argmax(-1),
        )
        return 5 * xy_loss + 5 * wh_loss + obj_loss + no_obj_loss + class_loss


class YoloV3Loss(nn.Module):
    """
    Calculate the loss for yolo (v3) model
    """

    def __init__(self):
        super(YoloV3Loss, self).__init__()
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
        self.small_loss = YoloV2Loss(self.anchor_boxes[6:9] / (416 / 13), S=13)
        self.medium_loss = YoloV2Loss(self.anchor_boxes[3:6] / (416 / 26), S=26)
        self.large_loss = YoloV2Loss(self.anchor_boxes[:3] / (416 / 52), S=52)

    def forward(self, predictions, target, device):
        s_loss = self.small_loss(predictions[0], target[0], device)
        m_loss = self.medium_loss(predictions[1], target[1], device)
        l_loss = self.large_loss(predictions[2], target[2], device)
        return s_loss + m_loss + l_loss


# -


class YoloV3Model(pl.LightningModule):
    def __init__(self, num_anchors=3, num_attrib=configs.NUM_CLASSES + 5):
        super(YoloV3Model, self).__init__()
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
        self.base_model = DarkNet53(darknet_base_config)
        self.tail = YoloV3Tail(
            small_scale_config,
            medium_scale_config,
            large_scale_config,
            num_anchors,
            num_attrib,
        )

    def forward(self, x):
        s, m, l = self.base_model(x)
        s, m, l = self.tail(s, m, l)
        return s, m, l

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=2e-5, weight_decay=configs.WEIGHT_DECAY
        )
        return {"optimizer": optimizer}

    def _preprocess(self, pred, anchor_boxes, S):
        anchor_boxes = anchor_boxes.to(self.device)
        exist_mask = torch.round(torch.sigmoid(pred[..., 4:5]))
        cell_idx = torch.arange(S, device=self.device)
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

    def _calc_map(self, y, pred):
        pred_boxes = []
        target_boxes = []
        small_preprocessed_pred = self._preprocess(
            pred[0], self.anchor_boxes[6:9] / (416 / 13), S=13
        )
        medium_preprocessed_pred = self._preprocess(
            pred[1], self.anchor_boxes[3:6] / (416 / 26), S=26
        )
        large_preprocessed_pred = self._preprocess(
            pred[2], self.anchor_boxes[:3] / (416 / 52), S=52
        )

        pred_boxes, target_boxes = get_bboxes(
            y=y,
            predictions=(
                small_preprocessed_pred,
                medium_preprocessed_pred,
                large_preprocessed_pred,
            ),
            iou_threshold=0.5,
            threshold=0.5,
            S=[13, 26, 52],
            B=3,
            device=self.device,
        )

        mean_avg_prec = mAP(pred_boxes, target_boxes, iou_threshold=0.5)
        return mean_avg_prec

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred_y = self(x)
        loss = YoloV3Loss()(pred_y, y, device=self.device)
        self.log("train_loss", loss, prog_bar=True)
        with torch.no_grad():
            mAP = self._calc_map(
                y, (pred_y[0].clone(), pred_y[1].clone(), pred_y[2].clone())
            )
            self.log("train_mAP", mAP, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred_y = self(x)
        loss = YoloV3Loss()(pred_y, y, device=self.device)
        self.log("valid_loss", loss, prog_bar=True)
        mAP = self._calc_map(
            y, (pred_y[0].clone(), pred_y[1].clone(), pred_y[2].clone())
        )
        self.log("valid_mAP", mAP, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred_y = self(x)
        loss = YoloV3Loss()(pred_y, y, device=self.device)
        self.log("test_loss", loss, prog_bar=True)
        mAP = self._calc_map(
            y, (pred_y[0].clone(), pred_y[1].clone(), pred_y[2].clone())
        )
        self.log("test_mAP", mAP, prog_bar=True)
        return loss


if __name__ == "__main__":
    model = YoloV3Model(num_attrib=5 + configs.NUM_CLASSES)
    data = YoloV3DataModule()
    trainer = pl.Trainer(
        gpus=1, overfit_batches=1, checkpoint_callback=False, max_epochs=10
    )
    trainer.fit(model, datamodule=data)
