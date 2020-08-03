import cv2
import sys
import argparse
import os
import random
import urllib
import zipfile
import requests
import subprocess
import logging
from collections import defaultdict
from math import ceil
import matplotlib.pyplot as plt
import numpy as np
import csv
import json
import utils

# CONSTANTS
COCO_CLASSES_URL = "https://raw.githubusercontent.com/amikelive/coco-labels/master/coco-labels-2014_2017.txt"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

## For development, can use the 1st link, otherwise it'll download the entire dataset (3GB)
ADE20K_URL =  "https://github.com/ctrnh/mlperf_misc/raw/master/ADE20K_subset.zip"
#ADE20K_URL = "https://groups.csail.mit.edu/vision/datasets/ADE20K/ADE20K_2016_07_26.zip"
GOOGLE_CLASSES_URL = "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv"
GOOGLE_IMG_URL = "https://datasets.appen.com/appen_datasets/open-images/zip_files_copy/validation.zip"
GOOGLE_ANN_URL = "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv"

from .source_dataset import InputDataset

class GoogleDataset(InputDataset):
    def __init__(self, input_data_path,  yes_all):
        super().__init__(input_data_path=input_data_path,
                          yes_all=yes_all)


        self.params = dict(zip(['IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsInside'],
                          ["01" for i in range(5)])) # TODO: img with which attributes should we keep?
        self.params["IsGroupOf"] = "0"

        self.load_classes()

    def load_classes(self):
        """
        Load ADE20K classes in addition to self.type classes.
        """
        self.classes = {}
        self.classes_reverse = {}
        classes_file_path = os.path.join(self.input_data_path, "class-descriptions-boxable.csv")
        with open(classes_file_path, 'r') as csvfile:
            spamreader = csv.reader(csvfile)#, delimiter=' ', quotechar='|')
            for i,row in enumerate(spamreader):
                self.classes[row[1].lower()] = row[0]
                self.classes_reverse[row[0]] = row[1].lower()

    def create_ann_dict(self,):
        """
        This function reads Google annotation csv file in order to keep the information which interest us.
        
        ann_dict: (dict)
            ann_dict[img_name] is a dict with object_id as key.
            Then ann_dict[img_name][object_id] is a dict which stores bbox, label and area of the object_id object inside the img_name image.
            An image is stored in ann_dict iff:
                - it has at least 1 bbox intersecting with Targetdataset which has an area > 0.2 (hyperparameter TBD)
                - it respects params (IsOccluded, IsTruncated etc) (Those are also hyperparameters TBD)
                - For imagenet: this bbox is the only bbox annotated (only 1 significant object in image)
        """
        ann_csv_path = os.path.join(self.input_data_path, GOOGLE_ANN_URL.split("/")[-1])

        logging.info(f"Reading annotations from {ann_csv_path}")
        logging.info(f"Keeping images which have following parameters: {self.params}")
        img_to_delete = set()
        self.ann_dict = {}
        with open(ann_csv_path, 'r') as csvfile:
            content = csv.reader(csvfile)
            for i, row in enumerate(content):
                if i == 0:
                    col_titles = dict(zip(row, [j for j in range(len(row))]))
                else:
                    img_name = row[col_titles["ImageID"]] + ".jpg"
                    google_label = row[col_titles["LabelName"]]
                    if img_name not in img_to_delete:
                        for attribute in self.params.keys():
                            if row[col_titles[attribute]] not in self.params[attribute]:
                                img_to_delete.add(img_name)
                        if img_name not in self.ann_dict:
                            self.ann_dict[img_name] = {}
                        obj_id = len(self.ann_dict[img_name].keys())
                        self.ann_dict[img_name][obj_id] = {}
                        self.ann_dict[img_name][obj_id]["source_label"] = google_label
                        self.ann_dict[img_name][obj_id]["normalized_bbox"] = {"top": float(row[6]),
                                                                       "bot": float(row[7]),
                                                                       "left": float(row[4]),
                                                                       "right": float(row[5])}
                        self.ann_dict[img_name][obj_id]["normalized_area"] = utils.bbox_area(*self.ann_dict[img_name][obj_id]['normalized_bbox'].values())


        all_img = list(self.ann_dict.keys())
        for img_name in all_img:
            if img_name in img_to_delete:
                del self.ann_dict[img_name]
