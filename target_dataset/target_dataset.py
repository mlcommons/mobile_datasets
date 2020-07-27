import logging
import os
from enum import Enum

import utils

class SubsamplingPolicy(Enum):
    random = 1
    balanced = 2

class TargetDataset:
    def __init__(self, mobile_app_path, force = False):
        self.force = force
        self.name = ""
        self.img_size = None
        self.out_ann_path = ""
        self.classes_url = ""
        self.ann_url = ""
        self.dataset_classes = []

        self.mobile_app_path = mobile_app_path

        self.tmp_path = os.path.join(self.mobile_app_path, "tmp_dataset_script") # temporary folder
        self.out_img_path = os.path.join(self.tmp_path, "img")
        utils.check_remove_dir(self.tmp_path, force = force)

        logging.info(f"Creating {self.out_img_path} directory")
        os.makedirs(self.out_img_path)
        self.min_normalized_bbox_area = 0.2
        self.class_sep = ", "
        self.classification = False

    def load_classes(self):
        raise NotImplementedError

    def format_img_name(self, name):
        raise NotImplementedError

    def intersecting_classes(self):
        raise NotImplementedError

    def read_annotations(self):
        raise NotImplementedError

    def subsample(self, N, policy = SubsamplingPolicy.random):
        raise NotImplementedError

    def write_annotations(self):
        raise NotImplementedError
