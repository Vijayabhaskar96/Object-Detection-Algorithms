import torch
import YoloV1_model
from dataset import YoloV1DataModule
from collections import namedtuple
from utils import WeightReader, load_conv_block
import pytorch_lightning as pl
import numpy as np

conv_config = namedtuple("ConvConfig", ["kernel_size", "filters", "stride", "pad"])
maxpool_config = namedtuple("MaxPoolConfig", ["kernel_size", "stride"])
repeat_block = namedtuple("Repeat", ["blocks", "n"])
architechture_config = [
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
DummyBlock = namedtuple("DummyBlock", ["conv"])

model = YoloV1_model.YoloV1Model(
    architechture=architechture_config,
    split_size=7,
    num_boxes=2,
    num_classes=20,
)
weight_reader = WeightReader("./yolov1.weights", initial_offset=5)


def apply_weights_on_seq(seq_block):
    for block in seq_block.children():
        if "CNNBlock" in str(block.__class__):
            load_conv_block(block, weight_reader)


apply_weights_on_seq(model.darknet)
print("After Loading:", weight_reader.offset)

for p in model.darknet.parameters():
    p.requires_grad = False

data = YoloV1DataModule()
trainer = pl.Trainer(gpus=1, checkpoint_callback=False, max_epochs=10000)
trainer.fit(model, datamodule=data)

# # save_pretrained_model in pth format
# filepath = "pretrained_yolov1-voc-model_weights.pth"
# torch.save(model.state_dict(), filepath)
