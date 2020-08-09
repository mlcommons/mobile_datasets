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


class InputDataset:
    """
    Class which represents the input dataset (e.g. Google open images) that one wants to subsample from.

    Different input datasets come with different formats. For example, ADE20K contains jpg images which are
    saved in different folders depending on their class (for example, images/training/a/abbey/ADE_train_00000970.jpg).
    When dealing with a new input dataset, one has to write a new class which inherits from InputDataset,
    and implement the corresponding methods.

    Attributes:
        force: bool
            if True, answers yes to all questions asked by the script (such as permission to remove folders)
        class_sep: str
            in the classes of a dataset, sometimes there are several names in one classes, seperated by class_sep: (e.g. "ball, sports ball" encode the same class, with class_sep=", ")
        input_data_path: str
            path to the input dataset
        ann_dict: dict
            dict to be created with self.create_ann_dict
    """
    def __init__(self, input_data_path, force, tmp_path):
        self.force = force
        self.input_data_path = input_data_path
        self.class_sep = ", "
        self.tmp_path = tmp_path


        if self.input_data_path is None:
            self.download_dataset()
        else:
            logging.info(f"Make sure that {self.input_data_path} only contains images which belong to the source dataset.")

        self.ann_dict = {}

    def download_dataset(self):
        """
        Downloads dataset from a url to the temp folder self.tmp_path and updates self.input_data_path accordingly.
        """
        raise ValueError("input_data_path must not be None, or download_dataset should be implemented")



    def load_classes(self):
        """
        Load source classes.
        """
        raise NotImplementedError

    def create_ann_dict(self):
        """
        This function reads the source dataset annotations in order to keep the information which interest us. (It fills self.ann_dict)
        ann_dict: (dict)
            ann_dict[img_name] is a dict with object_id as key.
            ann_dict[img_name][object_id] is a dict which stores bbox, label and area of the object_id object inside the img_name image.
            An image is stored in ann_dict iff:
                - it has at least 1 bbox intersecting with Targetdataset which has an area > 0.2 (hyperparameter TBD)
                - it respects params (IsOccluded, IsTruncated etc) (Those are also hyperparameters TBD)
                - For imagenet: this bbox is the only bbox annotated (only 1 significant object in image)
        """
        raise NotImplementedError
