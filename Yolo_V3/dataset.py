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

output_labels = [
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


def pathify(path):
    if type(path) == str:
        return Path(path)
    else:
        return path

BASE_PATH = pathify(configs.BASE_DIR)
class COCODataset(torch.utils.data.Dataset):
    def __init__(self, img_source_files, set_type, transform=None):
        self.source_files = img_source_files
        self.set_type = set_type
        self.transform = transform
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
        self.image_paths = []
        self.label_paths = []

        if type(self.source_files) == str:
            self.source_files = [pathify(self.source_files)]
        elif type(self.source_files) == list:
            self.source_files = [pathify(file) for file in self.source_files]
        for source_file in self.source_files:
            self.image_paths += Path(source_file).read_text().strip().split("\n")
        self.label_paths = [BASE_PATH/"labels"/p for p in self.image_paths]
        self.image_paths = [(BASE_PATH/set_type/p).with_suffix(".jpg") for p in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def _generate_label_matrix(self, S, boxes, class_labels, anchor_boxes):
        label_matrix = torch.zeros(
            (S, S, len(anchor_boxes), 5 + configs.NUM_CLASSES), dtype=torch.float64
        )
        for box, class_label in zip(boxes, class_labels):

            x, y, width, height = box
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            i, j = int(S * x), int(S * y)
            x_cell, y_cell = S * x, S * y

            xmin = x - width / 2
            ymin = y - height / 2
            xmax = xmin + width
            ymax = ymin + height
            # We need iou of anchor box and bbox as if they have same xmin and ymin, as only the width and height matters while assigning the bbox to anchor box.
            anchor_boxes[:, 0] = xmin
            anchor_boxes[:, 1] = ymin
            anchor_boxes[:, 2] = xmin + anchor_boxes[:, 2] / 2
            anchor_boxes[:, 3] = ymin + anchor_boxes[:, 3] / 2

            width_cell, height_cell = (width * S, height * S)

            # Assign bbox to max Anchor box overlap
            ious = box_iou(
                anchor_boxes,
                torch.tensor([xmin, ymin, xmax, ymax]).unsqueeze(0).float(),
            )
            _, max_idx = ious.max(0)

            # Box coordinates
            box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
            # j,i because in array row denotes height, to index height j is used first
            label_matrix[j, i, max_idx[0], :4] = box_coordinates

            # Set confidence to 1
            label_matrix[j, i, max_idx[0], 4] = 1
            # Set one hot encoding for class_label
            label_matrix[j, i, max_idx[0], 5 + class_label] = 1
        return label_matrix

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label_path = self.label_paths[index]
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = label.strip().split(",")
                class_label, x, y, width, height = int(class_label), float(x), float(y), float(width), float(height)
                boxes.append([class_label, x, y, width, height])

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = torch.tensor(boxes)

        if self.transform:
            output_labels_list = boxes[:, 0].int().tolist()
            if type(output_labels_list) == str:
                output_labels_list = [output_labels_list]
            transformed_items = self.transform(
                image=image, bboxes=boxes[:, 1:], class_labels=output_labels_list
            )
            image = transformed_items["image"]
            boxes = transformed_items["bboxes"]
            class_labels = transformed_items["class_labels"]

        # Convert To Cells
        # S x S x Num of Anchor box x ([x,y,w,h],class_confidence,...Class prediction...)
        small_label_matrix = self._generate_label_matrix(
            13, boxes, class_labels, copy.deepcopy(self.anchor_boxes)[6:9] / (416 / 13)
        )
        medium_label_matrix = self._generate_label_matrix(
            26, boxes, class_labels, copy.deepcopy(self.anchor_boxes)[3:6] / (416 / 26)
        )
        large_label_matrix = self._generate_label_matrix(
            52, boxes, class_labels, copy.deepcopy(self.anchor_boxes)[:3] / (416 / 52)
        )

        return image, (small_label_matrix, medium_label_matrix, large_label_matrix)


class YoloV3DataModule(pl.LightningDataModule):
    def setup(self, stage=None):
        self.train_transform = A.Compose(
            [
                A.Resize(width=416, height=416),
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=0),
                ToTensor(),
            ],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
        )
        self.test_transform = A.Compose(
            [A.Resize(width=416, height=416), ToTensor()],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
        )
        if stage == "fit" or stage is None:
            train_files = [BASE_PATH / "train2017.txt"]
            val_files = [BASE_PATH / "val2017.txt"]
            # self.train_dataset = COCODataset(train_files, "train2017", transform=self.test_transform)
            self.train_dataset = COCODataset(val_files, "val2017", transform=self.test_transform)
            self.val_dataset = COCODataset(val_files, "val2017", transform=self.test_transform)

        if stage == "test":
            test_files = [BASE_PATH / "val2017.txt"]
            self.test_dataset = COCODataset(test_files, "val2017", transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=configs.BATCH_SIZE,
            num_workers=configs.NUM_WORKERS,
            pin_memory=configs.PIN_MEMORY,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=configs.BATCH_SIZE,
            pin_memory=configs.PIN_MEMORY,
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=configs.BATCH_SIZE)
