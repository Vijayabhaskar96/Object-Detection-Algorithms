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

# ## Yolo V2 implementation in Pytorch
#

import os
kaggle_data={"username":"ENTER_YOUR_KAGGLE_USERNAME_HERE","key":"ENTER_YOUR_KAGGLE_KEY_HERE"}
os.environ['KAGGLE_USERNAME']=kaggle_data["username"]
os.environ['KAGGLE_KEY']=kaggle_data["key"]
# !pip install pytorch-lightning
# !pip install kaggle
# !pip install --upgrade albumentations
# !wget https://raw.githubusercontent.com/pjreddie/darknet/master/scripts/voc_label.py
# !kaggle datasets download -d vijayabhaskar96/pascal-voc-2007-and-2012
# !unzip pascal-voc-2007-and-2012.zip
# %run voc_label.py

from collections import namedtuple
import torch
from torch import nn
from torch.nn import functional as F
from dataset import YoloV2DataModule
from utils import get_bboxesmine, intersection_over_union, mAP
import configs
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
# %matplotlib inline
seed_everything(42)

conv_config = namedtuple("ConvConfig",["kernel_size","filters","stride","pad"])
maxpool_config = namedtuple("MaxPoolConfig",["kernel_size","stride"])
architechture_config1 = [conv_config(3, 32, 1, 1),
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
architechture_config2 = [maxpool_config(2, 2),
                        conv_config(3, 1024, 1, 1),
                        conv_config(1, 512, 1, 0),
                        conv_config(3, 1024, 1, 1),
                        conv_config(1, 512, 1, 0),
                        conv_config(3, 1024, 1, 1),
                        conv_config(3, 1024, 1, 1),
                        conv_config(3, 1024, 1, 1)
                        ]


class YoloV2Loss(nn.Module):
    """
    Calculate the loss for yolo (v2) model
    """

    def __init__(self, S=13, B=5, C=20):
        super(YoloV2Loss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.anchor_boxes = torch.tensor([[0,0,1.3221, 1.73145],[0,0,3.19275, 4.00944],[0,0,5.05587, 8.09892],[0,0,9.47112, 4.84053],[0,0,11.2364, 10.0071]])

        """
        S is split size of image (in paper 7),
        B is number of boxes (in paper 2),
        C is number of classes (in paper and VOC dataset is 20),
        """
        self.S = S
        self.B = B
        self.C = C


    def forward(self, predictions, target, device,epoch=0):
        self.anchor_boxes = self.anchor_boxes.to(device)
        exist_mask = target[...,0:1]
        existing_boxes = exist_mask * predictions
        non_existing_boxes = (1.0 - exist_mask) * predictions
        cell_idx = torch.arange(13,device=device)
        bx = exist_mask*torch.sigmoid(predictions[...,-4:-3]) + exist_mask*cell_idx.view([1,1,-1,1,1])
        by = exist_mask*torch.sigmoid(predictions[...,-3:-2]) + exist_mask*cell_idx.view([1,-1,1,1,1])
        bw = exist_mask*self.anchor_boxes[:,2].view([1,1,1,-1,1]) * exist_mask*torch.exp(predictions[...,-2:-1])
        bh = exist_mask*self.anchor_boxes[:,3].view([1,1,1,-1,1]) * exist_mask*torch.exp(predictions[...,-1:])

        ious = intersection_over_union(torch.cat([bx,by,bw,bh], dim=-1),target[...,-4:])

        xy_loss = self.mse(torch.cat([bx,by], dim=-1), target[...,-4:-2])
        bwbh = torch.cat([bw,bh], dim=-1)
        wh_loss = self.mse(torch.sqrt(torch.abs(bwbh)+1e-32),torch.sqrt(torch.abs(target[...,-2:])+1e-32))
        obj_loss = self.mse(exist_mask*ious,existing_boxes[...,0:1])
        no_obj_loss = self.mse(torch.zeros_like(non_existing_boxes[...,0:1]),non_existing_boxes[...,0:1])
        class_loss = F.nll_loss((exist_mask*F.log_softmax(predictions[..., 1:-4],dim=-1)).flatten(end_dim=-2),target[..., 1:-4].flatten(end_dim=-2).argmax(-1))
        return 5*xy_loss + 5*wh_loss + obj_loss + no_obj_loss + class_loss


#Thanks to Zhenliang He for the code
#https://discuss.pytorch.org/t/is-there-any-layer-like-tensorflows-space-to-depth-function/3487/15
class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x


class CNNBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, stride, pad):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels = in_channels,
                              out_channels = filters,
                              kernel_size = kernel_size,
                              stride = stride,
                              padding = pad,
                              bias = False)
        self.batchnorm = nn.BatchNorm2d(filters)
        self.leakyrelu = nn.LeakyReLU(0.1)
    def forward(self, x):
        x = self.leakyrelu(self.batchnorm(self.conv(x)))
        return x


class YoloV2Model(pl.LightningModule):
    def __init__(self, architechture=None, split_size=13, num_boxes=5, num_classes=20):
        super(YoloV2Model, self).__init__()
        self.S=split_size
        self.B=num_boxes
        self.C=num_classes
        self.darknet_before_skip = self._create_conv(architechture[0],in_channels=3)
        self.middle =  CNNBlock(in_channels = 512,
                                filters = 64,
                                kernel_size = 1,
                                stride = 1,
                                pad = 0)
        self.space_to_depth = SpaceToDepth(block_size=2)
        self.darknet_after_skip = self._create_conv(architechture[1],in_channels=512)
        self.conv_end1 =  CNNBlock(in_channels = 1280,
                                filters = 1024,
                                kernel_size = 3,
                                stride = 1,
                                pad = 1)
        self.final_conv =  nn.Conv2d(in_channels = 1024,
                              out_channels = num_boxes*(4+1+num_classes),
                              kernel_size = 1,
                              stride = 1,
                              padding = 0,
                              bias = False)
        self.loss = YoloV2Loss()
        self.anchor_boxes = torch.tensor([[0,0,1.3221, 1.73145],[0,0,3.19275, 4.00944],[0,0,5.05587, 8.09892],[0,0,9.47112, 4.84053],[0,0,11.2364, 10.0071]],device=self.device)

    def forward(self, x):
        x = self.darknet_before_skip(x)
        middle = self.middle(x)
        middle = self.space_to_depth(middle)
        x = self.darknet_after_skip(x)
        x = torch.cat([middle,x],dim=1)
        x = self.conv_end1(x)
        x = self.final_conv(x)
        x = x.permute(0,2,3,1)
        x = x.view(-1,self.S, self.S, self.B,4+1+self.C)
        return x
        
    def _create_conv(self, architecture,in_channels):
        layers = []

        for x in architecture:
            if "Conv" in str(type(x)):
                layer = CNNBlock(in_channels = in_channels,
                                    filters = x.filters,
                                    kernel_size = x.kernel_size,
                                    stride = x.stride,
                                    pad = x.pad)
                layers += [layer]
                in_channels = x.filters

            elif "MaxPool" in str(type(x)):
                layers += [nn.MaxPool2d(kernel_size=(x.kernel_size, x.kernel_size), stride=(x.stride, x.stride))]
        return nn.Sequential(*layers)

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5, weight_decay=configs.WEIGHT_DECAY)
        return {'optimizer': optimizer}

    def _calc_map(self, x, y, pred):
        self.anchor_boxes = self.anchor_boxes.to(self.device)
        exist_mask = pred[...,0:1]
        cell_idx = torch.arange(13,device=self.device)
        bx = exist_mask*torch.sigmoid(pred[...,-4:-3]) + exist_mask*cell_idx.view([1,1,-1,1,1])
        by = exist_mask*torch.sigmoid(pred[...,-3:-2]) + exist_mask*cell_idx.view([1,-1,1,1,1])
        bw = exist_mask*self.anchor_boxes[:,2].view([1,1,1,-1,1]) * exist_mask*torch.exp(pred[...,-2:-1])
        bh = exist_mask*self.anchor_boxes[:,3].view([1,1,1,-1,1]) * exist_mask*torch.exp(pred[...,-1:])
        pred[...,-4:]=torch.cat([bx,by,bw,bh],dim=-1)
        pred_boxes, target_boxes = get_bboxesmine(x=x,y=y,predictions=pred,iou_threshold=0.5, threshold=0.4, S=self.S,B=self.B, device=self.device)
        mean_avg_prec = mAP(pred_boxes,target_boxes,iou_threshold=0.5)
        return mean_avg_prec

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred_y = self(x)
        loss = YoloV2Loss()(pred_y, y, device=self.device,epoch=self.current_epoch)
        self.log('train_loss', loss, prog_bar=True)
        with torch.no_grad():
            mAP = self._calc_map(x, y, pred_y.clone())
            self.log('train_mAP', mAP, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred_y = self(x)
        loss = YoloV2Loss()(pred_y, y, device=self.device,epoch=self.current_epoch)
        self.log('valid_loss', loss, prog_bar=True)
        mAP = self._calc_map(x.detach(), y.detach(), pred_y.detach())
        self.log('valid_mAP', mAP,prog_bar=True)
        return loss

model = YoloV2Model(architechture = [architechture_config1,architechture_config2], split_size=13, num_boxes=5, num_classes=20)
data = YoloV2DataModule()

trainer = pl.Trainer(gpus=1,overfit_batches=1,max_epochs=1000)
trainer.fit(model,datamodule=data)
