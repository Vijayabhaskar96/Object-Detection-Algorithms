from pathlib import Path
import pandas as pd
import json
import csv
from tqdm.auto import tqdm
from collections import defaultdict
import configs
import numpy as np


def convert(size, box):
    dw = 1.0 / (size[0])
    dh = 1.0 / (size[1])
    x = box[0] + box[2] / 2.0 - 1
    y = box[1] + box[3] / 2.0 - 1
    w = box[2]
    h = box[3]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def create_labels(set_name):
    BASE_PATH = Path(configs.BASE_DIR)
    label_path = BASE_PATH / "labels"
    annotation_path = BASE_PATH / "annotations"
    data = json.load(open(annotation_path / f"instances_{set_name}.json", "r"))
    id_to_cat = {}

    for i, cat in enumerate(data["categories"]):
        id_to_cat[cat["id"]] = cat["name"]
    old_ids = []

    for k, v in id_to_cat.items():
        old_ids.append(k)

    df_images = pd.DataFrame(data["images"])
    df_annon = pd.DataFrame(data["annotations"])

    df_images = df_images.rename(columns={"id": "image_id"})
    final_df = df_annon.merge(df_images, on=["image_id"])

    filename_to_bbox = defaultdict(list)
    for row in final_df.iterrows():
        bbox = convert(size=(row[1]["width"], row[1]["height"]), box=row[1]["bbox"])
        if any(np.array(row[1]["bbox"]) < 1):
            continue
        cat = old_ids.index(row[1]["category_id"])
        filename_to_bbox[row[1]["file_name"]].append([cat] + list(bbox))

    with open(BASE_PATH / f"{set_name}.txt", "w", newline="", encoding="utf-8") as f:
        valid_writer = csv.writer(f)
        for filename, bbox in tqdm(filename_to_bbox.items()):
            label_filename = (label_path / filename).with_suffix(".txt")
            with open(label_filename, "w", newline="", encoding="utf-8") as fp:
                csv_writer = csv.writer(fp)
                csv_writer.writerows(bbox)
            valid_writer.writerow([label_filename.name])


create_labels("val2017")
# uncomment if you want to train dataset
# create_labels("train2017")
