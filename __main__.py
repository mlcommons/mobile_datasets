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

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# CONSTANTS
IMAGENET_CLASSES_URL = "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"
COCO_CLASSES_URL = "https://raw.githubusercontent.com/amikelive/coco-labels/master/coco-labels-2014_2017.txt"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
## For development, can use the 1st link, otherwise it'll download the entire dataset (3GB)
ADE20K_URL =  "https://github.com/ctrnh/mlperf_misc/raw/master/ADE20K_subset.zip"
#ADE20K_URL = "https://groups.csail.mit.edu/vision/datasets/ADE20K/ADE20K_2016_07_26.zip"


from input_dataset.ade20k import ADE20KDataset
from input_dataset.google import GoogleDataset




def main():
    # TODO: put in config file ?
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_path", type=str, help="directory of input dataset (img + annotations). If None, download script", default=None)
    parser.add_argument("--mobile_app_path", type=str, help="path to root directory containing the mobile app repo")
    parser.add_argument("--N", type=int, help="number of samples wanted in the dataset")
    parser.add_argument("--type", type=str.lower, help="coco or imagenet", choices=["coco", "imagenet"])
    parser.add_argument("--dataset", type=str.lower, default="ade20k", help="Kanter or ADE20K or other to implement", choices=["kanter", "ade20k","google"])
    parser.add_argument("-y", action="store_true", help="automatically answer yes to all questions. If on, the script may remove folders without permission.")
    parser.add_argument("--subsampling_strategy", type=str.lower, help="random or balanced", choices=["random", "balanced"], default="random")
    args = parser.parse_args()

    adb_devices = subprocess.run(["adb", "devices"], stdout=subprocess.PIPE, universal_newlines=True, check=True)
    assert len(adb_devices.stdout.split('\n')) > 3, "No device attached. Please connect your phone."
    logger.info(adb_devices.stdout)
    # TODO : check writing permission + allow only 1 device

    input_data_path = args.input_data_path
    if args.dataset == "kanter":
        import json
        input_dataset = KanterDataset(input_data_path=input_data_path,
                                mobile_app_path=args.mobile_app_path,
                                type=args.type,
                                yes_all=args.y)
    elif args.dataset == "ade20k":
        input_dataset = ADE20KDataset(input_data_path=input_data_path,
                                mobile_app_path=args.mobile_app_path,
                                type=args.type,
                                yes_all=args.y)
    elif args.dataset == "google":
        input_dataset = GoogleDataset(input_data_path=input_data_path,
                                mobile_app_path=args.mobile_app_path,
                                type=args.type,
                                yes_all=args.y)
    selected_img_path = input_dataset.subsample(N=args.N, policy=args.subsampling_strategy)
    input_dataset.process_dataset(selected_img_path=selected_img_path)


if __name__ == '__main__':
    main()
