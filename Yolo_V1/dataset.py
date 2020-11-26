"""
Creates a Pytorch dataset to load the Pascal VOC dataset
"""

import torch
import cv2
from pathlib import Path
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch.transforms import ToTensor
from torch.utils.data import DataLoader
import configs

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


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, img_source_files, S=7, B=2, C=20, transform=None):
        self.source_files = img_source_files
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
        self.image_paths = []
        self.label_paths = []
        if type(self.source_files) == str:
            self.source_files = [pathify(self.source_files)]
        elif type(self.source_files) == list:
            self.source_files = [pathify(file) for file in self.source_files]
        for source_file in self.source_files:
            self.image_paths += Path(source_file).read_text().strip().split("\n")
        self.image_paths = [pathify(p) for p in self.image_paths]
        self.label_paths = [
            p.parent.with_name("labels") / f"{p.stem}.txt" for p in self.image_paths
        ]

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
            output_labels_list = boxes[:, 0].int().tolist()
            if type(output_labels_list) == str:
                output_labels_list = [output_labels_list]
            transformed_items = self.transform(
                image=image, bboxes=boxes[:, 1:], class_labels=output_labels_list
            )
            image = transformed_items["image"] / 255.0
            boxes = transformed_items["bboxes"]
            class_labels = transformed_items["class_labels"]

        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        # label_matrix = torch.zeros((self.S, self.S, self.C + 5))
        for box, class_label in zip(boxes, class_labels):
            x, y, width, height = box
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

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
            width_cell, height_cell = (width * self.S, height * self.S)

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            if label_matrix[i, j, 20] == 0:
                # Set that there exists an object
                label_matrix[i, j, 20] = 1

                # Box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                label_matrix[i, j, 21:25] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1

        return image, label_matrix


class YoloV1DataModule(pl.LightningDataModule):
    def setup(self, stage=None):
        BASE_PATH = pathify(configs.BASE_DIR)
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
        if stage == "fit" or stage is None:
            train_files = [BASE_PATH / "2007_train.txt", BASE_PATH / "2012_train.txt"]
            val_files = [BASE_PATH / "2007_val.txt", BASE_PATH / "2012_val.txt"]
            self.train_dataset = VOCDataset(train_files, transform=self.train_transform)
            self.val_dataset = VOCDataset(val_files, transform=self.test_transform)

        if stage == "test":
            test_files = [BASE_PATH / "2007_test.txt"]
            self.test_dataset = VOCDataset(test_files, transform=self.test_transform)

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
