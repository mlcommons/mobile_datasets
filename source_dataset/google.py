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

from contextlib import closing

from .source_dataset import InputDataset

class GoogleDataset(InputDataset):
    def __init__(self, input_data_path, tmp_path, force):
        super().__init__(input_data_path=input_data_path,tmp_path=tmp_path,
                          force=force)


        self.params = dict(zip(['IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsInside'],
                          ["01" for i in range(5)])) # TODO: img with which attributes should we keep?
        self.params["IsGroupOf"] = "0"

        for param in self.params.keys():
            if self.params[param] == "0":
                logging.info(f"Images with attribute {param} will not be kept")

        self.google_classes_url = "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv"
        self.google_img_url = "https://datasets.appen.com/appen_datasets/open-images/zip_files_copy/validation.zip"
        self.google_ann_url = "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv"

        self.load_classes()

    def download_dataset(self):
        """
        Downloads only images (modify self.*_url if you want a dataset different than validation) to temporary folder. (Annotations are not downloaded)
        """
        self.google_img_url = "https://datasets.appen.com/appen_datasets/open-images/zip_files_copy/validation.zip"
        img_name = self.google_img_url.split("/")[-1]
        utils.download_required_files(url=self.google_img_url, folder_path=self.tmp_path, file_name=img_name, force=self.force)
        self.input_data_path = os.path.join(self.tmp_path, img_name)


    def load_classes(self):
        self.classes = {}
        self.classes_reverse = {}
        with requests.Session() as s:
            cr = csv.reader(s.get(self.google_classes_url).content.decode('utf-8').splitlines())
            list_csv = list(cr)
            for i, row in enumerate(list_csv):
                self.classes[row[1].lower()] = row[0]
                self.classes_reverse[row[0]] = row[1].lower()

    def create_ann_dict(self,):
        """
        This function reads Google annotation csv file in order to store the information which interest us in self.ann_dict.

        ann_dict: (dict)
            ann_dict[img_name] is a dict with object_id as key.
            ann_dict[img_name][object_id] is a dict which stores bbox, label and area of the object_id object inside the img_name image.
            An image is stored in ann_dict iff:
                - it has at least 1 bbox intersecting with Targetdataset which has an area > 0.2 (hyperparameter TBD)
                - it respects params (IsOccluded, IsTruncated etc) (Those are also hyperparameters TBD)
                - for imagenet: this bbox is the only bbox annotated (only 1 significant object in image)
        """

        logging.info(f"Reading annotations from {self.google_ann_url}")
        img_to_delete = set()
        self.ann_dict = {}
        with requests.Session() as s:
            cr = csv.reader(s.get(self.google_ann_url).content.decode('utf-8').splitlines())
            list_csv = list(cr)
            for i, row in enumerate(list_csv):
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
                        self.ann_dict[img_name][obj_id]["normalized_area"] = ((float(row[7]) - float(row[6]))*(float(row[5])-float(row[4])))


        all_img = list(self.ann_dict.keys())
        for img_name in all_img:
            if img_name in img_to_delete:
                del self.ann_dict[img_name]
