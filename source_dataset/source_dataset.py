import cv2
import sys
import argparse
import os
import random
import shutil
import urllib
import zipfile
import subprocess
import logging
from collections import defaultdict
from math import ceil
import matplotlib.pyplot as plt
import numpy as np

# CONSTANTS
COCO_CLASSES_URL = "https://raw.githubusercontent.com/amikelive/coco-labels/master/coco-labels-2014_2017.txt"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
## For development, can use the 1st link, otherwise it'll download the entire dataset (3GB)
ADE20K_URL =  "https://github.com/ctrnh/mlperf_misc/raw/master/ADE20K_subset.zip"
#ADE20K_URL = "https://groups.csail.mit.edu/vision/datasets/ADE20K/ADE20K_2016_07_26.zip"

class InputDataset:
    """
    Class which represents the input dataset (e.g. ADE20K) that one wants to subsample from and reformat
    into a new type of dataset (e.g. imagenet).

    Different input datasets come with different formats. For example, ADE20K contains jpg images which are
    saved in different folders depending on their class (for example, images/training/a/abbey/ADE_train_00000970.jpg).
    When dealing with a new input dataset, one has to write a new class which inherits from InputDataset,
    and implement the corresponding methods.

    Attributes:
        yes_all: bool
            if True, answers yes to all questions asked by the script (such as permission to remove folders)
        type: str ("imagenet" or "coco")
            type of the dataset one wants to mimic
        new_img_size: (int, int)
            if images need rescaling, new_img_size is the new shape of the image
        mobile_app_path: str
            path to the folder containing the mobile_app repo
        tmp_path: str
            path to a temporary folder which will be created and removed at the end of the process
        out_ann_path: str
            path to the folder which contains the annotations files (in mobile_app repo)
        out_img_path: str
            path to the temporary folder where the script will dump the new dataset images before pushing to phone
        input_data_path: str
            path to the input dataset
    """
    def __init__(self, input_data_path, yes_all):
        self.yes_all = yes_all

        self.input_data_path = input_data_path

        self.class_sep = ", "


        if self.input_data_path is None:
            self.download_dataset()

        self.ann_dict = {}

    def download_dataset(self):
        """
        Downloads dataset from a url to the temp folder self.tmp_path and updates self.input_data_path accordingly.
        """
        raise ValueError("input_data_path must not be None, or download_dtaset should be implemented")



    def load_classes(self):
        raise NotImplementedError

    def create_ann_dict(self):
        raise NotImplementedError
