import logging
import requests
import urllib
import os
import zipfile
import json

import numpy as np

from .target_dataset import TargetDataset

class ImageNet(TargetDataset):
    def __init__(self, mobile_app_path, force = False):
        super().__init__(mobile_app_path=mobile_app_path,
                         force=force)
        self.name = "imagenet"
        self.out_ann_path = os.path.join(self.mobile_app_path, "java", "org", "mlperf", "inference", "assets", "imagenet_val.txt")

        self.min_normalized_bbox_area = 0.3
        self.max_nbox = 1

        self.compute_percentile_grp()
        self.load_classes()

    def __str__(self):
        return "imagenet"

    def load_classes(self):
        imagenet_classes_url = "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"
        logging.info("Loading imagenet classes")
        self.classes =  {v:k for (k,v) in eval(requests.get(imagenet_classes_url).text).items()}
        logging.debug(f"nb Imagenet classes: {len(self.classes.keys())}")

    def format_img_name(self, name):
        return f"ILSVRC2012_val_{name:08}.JPEG"


    def write_annotation(self, transformation_annotations, ann_file, img_path, new_img_name):
        label = transformation_annotations[img_path]['objects'][0]['target_label']
        logging.debug(f"Img {img_path}, imagenet label {label}")
        ann_file.write(str(label) + "\n")
