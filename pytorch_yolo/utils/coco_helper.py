import os

from datetime import datetime

import pandas as pd
from PIL import Image

from tqdm import tqdm


def create_info(description="", url="", version="0.0", contributor=""):
    return {
        "description": description,
        "url": url,
        "version": version,
        "year": datetime.now().year,
        "contributor": contributor,
        "date_created": str(datetime.now()),
    }


def create_license(url="", id=1, name=""):
    return {"url": url, "id": id, "name": name}


def create_category(supercategory="", id=1, name=""):
    return {"supercategory": supercategory, "id": id, "name": name}


def create_image_info(file_name, height, width, license_id=1, coco_url="", date_captured="", flickr_url="", id=1):
    return {
        "license": license_id,
        "file_name": file_name,
        "coco_url": coco_url,
        "height": height,
        "width": width,
        "date_captured": date_captured or str(datetime.now()),
        "flickr_url": flickr_url,
        "id": id,
    }


def create_annotation_info(image_id: int, bbox: list, segmentation=(), area=0, iscrowd=0, category_id=1, id=1):
    return {
        "segmentation": list(segmentation),
        "area": area,
        "iscrowd": iscrowd,
        "image_id": image_id,
        "bbox": bbox,
        "category_id": category_id,
        "id": id,
    }


def dataset_from_dict(data: dict, categories=(("car", "vehicle"),), image_dir="", has_image_info=False):
    info = create_info()
    license = create_license()
    category_info = [create_category(id=i, name=cat, supercategory=sup) for i, (cat, sup) in enumerate(categories)]

    images_info = []
    annotation_info = []

    anno_id = 0
    for i, (path, item) in tqdm(enumerate(data.items()), total=len(data)):
        if not has_image_info:
            image: Image.Image = Image.open(os.path.join(image_dir, path))
            images_info.append(create_image_info(path, image.height, image.width, id=i))
        else:
            height, width, item = item
            images_info.append(create_image_info(path, height, width, id=i))

        for item_info in item:
            bbox = [
                item_info["left"],
                item_info["top"],
                item_info["right"] - item_info["left"] + 1,
                item_info["bottom"] - item_info["top"] + 1,
            ]

            annotation_info.append(
                create_annotation_info(id=anno_id, bbox=bbox, image_id=i, category_id=item_info["type"])
            )
            anno_id += 1

    return {
        "info": info,
        "licenses": license,
        "images": images_info,
        "annotations": annotation_info,
        "categories": category_info,
    }


def dataset_from_df(df: pd.DataFrame, categories=(("car", "vehicle"),), image_dir=""):
    images = {}
    for _, row in df.iterrows():
        path = row["image_path"]

        if path in images:
            images[path].append(row)
        else:
            images[path] = [row]

    return dataset_from_dict(images, categories, image_dir)


def create_result_info(image_id: int, category_id: int, bbox: list, score: float):
    return {"image_id": image_id, "category_id": category_id, "bbox": bbox, "score": score}


def results_from_dict(data: dict, annotations: dict):
    path_to_id = {i["file_name"]: i["id"] for i in annotations["images"]}

    results = []
    for file_name, item in tqdm(data.items(), total=len(data)):
        image_id = path_to_id[file_name]

        for pred in item:
            cat_id = pred["type"]
            score = pred["score"]
            bbox = [pred["left"], pred["top"], pred["right"] - pred["left"] + 1, pred["bottom"] - pred["top"] + 1]

            results.append(create_result_info(image_id, cat_id, bbox, score))

    if len(results) == 0:
        results.append(create_result_info(1, 0, [0, 0, 0, 0], 0))

    return results


def results_from_df(df: pd.DataFrame, annotations: dict):
    images = {}
    for _, row in df.iterrows():
        path = row["image_path"]

        if path in images:
            images[path].append(row)
        else:
            images[path] = [row]

    return results_from_dict(images, annotations)


if __name__ == "__main__":
    df = pd.read_csv("/home/druzhinin/HDD/Projects/Detection/1000_and_10_000.csv")
    df["image_path"] = df["image_path"].str.replace("/home/druzhinin/Datasets/BMP/", "")
    df["score"] = 1

    data = dataset_from_df(df, image_dir="/home/druzhinin/Datasets/BMP/")
    result = results_from_df(df, data)
    print(len(result))

    import json

    with open("test_coco.json", "w") as file:
        json.dump(data, file)
