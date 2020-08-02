import logging
import urllib
import os
import zipfile
import json
import requests
import numpy as np

from .target_dataset import TargetDataset

class ADE20K_target(TargetDataset):
    def __init__(self, mobile_app_path, force = False):
        super().__init__(mobile_app_path=mobile_app_path,
                         force=force)
        self.name = "ADE20K"
        self.out_ann_path = os.path.join(self.tmp_path, "ade.txt") #TODO: png folder, TBD when we get final implementation of mobile_app
        self.img_size = (512, 512)

        self.class_sep = ';' # separator for classes of ade20k

        self.load_classes()

    def __str__(self):
        return "ade20k"

    def load_classes(self):
        ade20k_150_classes_url = "https://raw.githubusercontent.com/CSAILVision/sceneparsing/master/objectInfo150.csv"
        list_ade20k_classes = list(map(lambda x: x.split(",")[-1], requests.get(ade20k_150_classes_url).text.split("\n")))[1:32]
        self.classes_reverse = dict(zip([i for i in range(1,32)],list_ade20k_classes))
        self.classes = dict(zip(list_ade20k_classes,[i for i in range(1,32)]))



    def format_img_name(self, name):
        return f"ADE_val_{name:08}.jpg"

    def write_annotation(self):
        logging.warning("Annotations for ADE20K not implemented yet.")
        pass
