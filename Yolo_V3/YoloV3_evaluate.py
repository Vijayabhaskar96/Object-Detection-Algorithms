import torch
import YoloV3_model
from dataset import YoloV3DataModule
from collections import namedtuple
from utils import WeightReader, load_conv_block
import pytorch_lightning as pl

conv_config = namedtuple("ConvConfig", ["kernel_size", "filters", "stride", "pad"])
maxpool_config = namedtuple("MaxPoolConfig", ["kernel_size", "stride"])
DummyBlock = namedtuple("DummyBlock", ["conv"])

model = YoloV3_model.YoloV3Model()
data = YoloV3DataModule()
# +
weight_reader = WeightReader("yolov3.weights", initial_offset=5)


def apply_weights_on_seq(seq_block):
    for block in seq_block.children():
        if "CNNBlock" in str(block.__class__):
            load_conv_block(block, weight_reader)
        elif "DoubleConvWithResidual" in str(block.__class__):
            load_conv_block(block.conv1, weight_reader)
            load_conv_block(block.conv2, weight_reader)
        elif "Conv2d" in str(block.__class__):
            load_conv_block(block, weight_reader, with_bn=False)


apply_weights_on_seq(model.base_model.part1)
apply_weights_on_seq(model.base_model.part2)
apply_weights_on_seq(model.base_model.part3)
apply_weights_on_seq(model.base_model.part4)

apply_weights_on_seq(model.tail.small_part_first)
apply_weights_on_seq(model.tail.small_part_branch)
apply_weights_on_seq(model.tail.small_part_end)
apply_weights_on_seq(model.tail.small_part_branch_net)

apply_weights_on_seq(model.tail.medium_part_first)
apply_weights_on_seq(model.tail.medium_part_branch)
apply_weights_on_seq(model.tail.medium_part_end)
apply_weights_on_seq(model.tail.medium_part_branch_net)

apply_weights_on_seq(model.tail.large_part)

print("After Loading:", weight_reader.offset)
# -
# TODO create Datamodule for COCO and evaluate on COCO dataset,
# for now I'm just loading official COCO weights to export pytorch friendly weights

trainer = pl.Trainer(gpus=1, checkpoint_callback=False, max_epochs=10000)
trainer.test(model, datamodule=data)

# save_pretrained_model in pth format
# filepath = "pretrained_yolov3-coco-model_weights.pth"
# torch.save(model.state_dict(), filepath)
