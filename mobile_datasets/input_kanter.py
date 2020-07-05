"""
This is an example of how a custom dataset can be written
"""
import cv2
import sys
import argparse
import os
import random
import shutil
import urllib
import zipfile
import requests
import subprocess
import logging
from collections import defaultdict
from math import ceil
import matplotlib.pyplot as plt
import numpy as np

from input_dataset import InputDataset

class KanterDataset(InputDataset):
    def __init__(self, input_data_path, mobile_app_path, type,yes_all):
        super().__init__(input_data_path=input_data_path,
                        mobile_app_path=mobile_app_path,
                        type=type,
                        yes_all=yes_all)
        self.in_img_path = os.path.join(self.input_data_path, "img")
        self.in_annotations = json.load(open(os.path.join(self.input_data_path, "annotations", "labels.json"), 'r'))


    def subsample(self, N, policy="random"):
        """Policy for selecting images.

        Args:
            N: number of images we want to select
            policy: subsampling policy (TODO)
        Returns:
            list of selected image paths
        """
        if policy == "random":
            all_img_names = [f for f in os.listdir(self.in_img_path) \
                            if f.lower().endswith(("jpg", "png", "jpeg"))]
            if N < len(all_img_names):
                selected_img = random.sample(all_img_names, N)
            else:
                selected_img = all_img_names
        return list(map(lambda x:os.path.join(self.in_img_path, x), selected_img))

    def find_label(self, img_path):
        if self.type == "imagenet":
            img_name = img_path.split("/")[-1]
            label = self.IMAGENET_CLASSES[self.in_annotations[img_name]]
        return label
