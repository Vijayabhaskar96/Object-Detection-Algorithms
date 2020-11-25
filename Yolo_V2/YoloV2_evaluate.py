import torch
import YoloV2_model
from dataset import YoloV2DataModule
from collections import namedtuple
from utils import WeightReader, load_conv_block
import pytorch_lightning as pl

conv_config = namedtuple("ConvConfig", ["kernel_size", "filters", "stride", "pad"])
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
DummyBlock = namedtuple("DummyBlock", ["conv"])

model = YoloV2_model.YoloV2Model(
    architechture=[architechture_config1, architechture_config2],
    split_size=13,
    num_boxes=5,
    num_classes=20,
)

weight_reader = WeightReader("yolov2-voc.weights", initial_offset=5)


def apply_weights_on_seq(seq_block):
    for block in seq_block.children():
        if "CNNBlock" in str(block.__class__):
            load_conv_block(block, weight_reader)


apply_weights_on_seq(model.darknet_before_skip)
apply_weights_on_seq(model.darknet_after_skip)
load_conv_block(model.middle, weight_reader)
load_conv_block(model.conv_end1, weight_reader)
load_conv_block(model.final_conv, weight_reader, with_bn=False)
print("After Loading:", weight_reader.offset)

data = YoloV2DataModule()
for p in model.parameters():
    p.requires_grad = False

trainer = pl.Trainer(gpus=1, checkpoint_callback=False, max_epochs=10000)
trainer.test(model, datamodule=data)
# TODO find and fix why mAP is low with pretrained weights ~0.60s vs 76.8

# save_pretrained_model in pth format
filepath = "pretrained_yolov2-voc-model_weights.pth"
torch.save(model.state_dict(), filepath)
