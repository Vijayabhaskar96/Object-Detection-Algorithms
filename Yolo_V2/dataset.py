
"""
Creates a Pytorch dataset to load the Pascal VOC dataset
"""

import torch
from torch._C import dtype
from torchvision.ops.boxes import box_iou
import cv2
from pathlib import Path
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch.transforms import ToTensor
from torch.utils.data import DataLoader
import configs
import copy
output_labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                 "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
                 "train", "tvmonitor"]
def pathify(path):
    if type(path) == str:
        return Path(path)
    else:
        return path

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, img_source_files, S=13, B=5, C=20, transform=None):
        self.source_files = img_source_files
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
        self.anchor_boxes = torch.tensor([[0,0,1.3221, 1.73145],[0,0,3.19275, 4.00944],[0,0,5.05587, 8.09892],[0,0,9.47112, 4.84053],[0,0,11.2364, 10.0071]])/self.S
        self.image_paths = []
        self.label_paths = []

        if type(self.source_files)==str:
            self.source_files = [pathify(self.source_files)]
        elif type(self.source_files)==list:
            self.source_files = [pathify(file) for file in self.source_files]
        for source_file in self.source_files:
            self.image_paths += Path(source_file).read_text().strip().split("\n")
        self.image_paths = [pathify(p) for p in self.image_paths]
        self.label_paths = [p.parent.with_name("labels")/f"{p.stem}.txt" for p in self.image_paths]  

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label_path = self.label_paths[index]
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]
                boxes.append([class_label, x, y, width, height])

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = torch.tensor(boxes)

        if self.transform:
            output_labels_list = boxes[:,0].int().tolist()
            if type(output_labels_list)==str:
                output_labels_list = [output_labels_list]
            transformed_items = self.transform(image = image, bboxes = boxes[:,1:], class_labels=output_labels_list)
            image = transformed_items["image"]
            boxes = transformed_items["bboxes"]
            class_labels = transformed_items["class_labels"]

        # Convert To Cells
        # 13 x 13 x Num of Anchor box x (class_confidence,...Class prediction...,[x,y,w,h])
        label_matrix = torch.zeros((self.S, self.S, self.B, 1 + self.C + 4),dtype=torch.float64)
        for box, class_label in zip(boxes, class_labels):
            anchor_boxes = copy.deepcopy(self.anchor_boxes)
            x, y, width, height = box
            class_label = int(class_label)
            
            # i,j represents the cell row and cell column
            i, j = int(self.S * x), int(self.S * y)
            # x_cell, y_cell = self.S * x - i, self.S * y - j
            x_cell, y_cell = self.S * x, self.S * y

            """
            Calculating the width and height of cell of bounding box,
            relative to the cell is done by the following, with
            width as the example:
            
            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)
            
            Then to find the width relative to the cell is simply:
            width_pixels/cell_pixels, simplification leads to the
            formulas below.
            """
            xmin = x-width/2
            ymin = y-height/2
            xmax = xmin + width
            ymax = ymin + height
            #We need iou of anchor box and bbox as if they have same xmin and ymin, as only the width and height matters while assigning the bbox to anchor box.
            anchor_boxes[:,0] = xmin 
            anchor_boxes[:,1] = ymin
            anchor_boxes[:,2] = xmin + anchor_boxes[:,2]/2
            anchor_boxes[:,3] = ymin + anchor_boxes[:,3]/2

            width_cell, height_cell = (width * self.S, height * self.S)

            #Assign bbox to max Anchor box overlap
            ious = box_iou(anchor_boxes,torch.tensor([xmin,ymin,xmax,ymax]).unsqueeze(0).float())
            _, max_idx = ious.max(0)

            # Box coordinates
            box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
            #j,i because in array row denotes height, to index height j is used first
            label_matrix[j, i, max_idx[0],-4:] = box_coordinates
            
            #Set confidence to 1
            label_matrix[j, i, max_idx[0], 0] = 1
            # Set one hot encoding for class_label
            label_matrix[j, i, max_idx[0], 1 + class_label] = 1

        return image, label_matrix


class YoloV2DataModule(pl.LightningDataModule):
    def setup(self, stage=None):
        BASE_PATH = pathify(configs.BASE_DIR)
        self.train_transform = A.Compose([A.Resize(width=416, height=416),
                                    A.ShiftScaleRotate(shift_limit=0.2,scale_limit=0.2,rotate_limit=0),A.Normalize(),ToTensor()],
                            bbox_params=A.BboxParams(format='yolo',label_fields=['class_labels']))
        self.test_transform = A.Compose([A.Resize(width=416, height=416),A.Normalize(),ToTensor()],
                                    bbox_params=A.BboxParams(format='yolo',label_fields=['class_labels']))
        if stage == 'fit' or stage is None:
            train_files = [BASE_PATH/"2007_train.txt",BASE_PATH/"2012_train.txt"]
            val_files = [BASE_PATH/"2007_val.txt",BASE_PATH/"2012_val.txt"]
            self.train_dataset = VOCDataset(train_files,transform=self.test_transform)
            self.val_dataset = VOCDataset(val_files,transform=self.test_transform)

        if stage == 'test':
            test_files = [BASE_PATH/"2007_test.txt"]
            self.test_dataset = VOCDataset(test_files, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=configs.BATCH_SIZE, num_workers=configs.NUM_WORKERS, pin_memory=configs.PIN_MEMORY, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=configs.BATCH_SIZE, pin_memory=configs.PIN_MEMORY)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=configs.BATCH_SIZE)
