import argparse
import os
import urllib
import zipfile
import subprocess
import logging
from collections import defaultdict
from math import ceil
import matplotlib.pyplot as plt
import numpy as np

from input_ade20k import ADE20KDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_path", type=str, help="directory of input dataset (img + annotations). If None, download script", default=None)
    parser.add_argument("--mobile_app_path", type=str, help="path to root directory containing the mobile app repo")
    parser.add_argument("--N", type=int, help="number of samples wanted in the dataset")
    parser.add_argument("--type", type=str.lower, help="coco or imagenet", choices=["coco", "imagenet"])
    parser.add_argument("--dataset", type=str.lower, default="ade20k", help="Kanter or ADE20K or other to implement", choices=["kanter", "ade20k"])
    parser.add_argument("-y", action="store_true", help="automatically answer yes to all questions. If on, the script may remove folders without permission.")
    parser.add_argument("--subsampling_strategy", type=str.lower, help="random or balanced", choices=["random", "balanced"], default="random")
    args = parser.parse_args()

    adb_devices = subprocess.run(["adb", "devices"], stdout=subprocess.PIPE, universal_newlines=True, check=True)
    assert len(adb_devices.stdout.split('\n')) > 3, "No device attached. Please connect your phone."
    logger.info(adb_devices.stdout)

    input_data_path = args.input_data_path
    if args.dataset == "kanter":
        import json
        in_dataset = KanterDataset(input_data_path=input_data_path,
                                mobile_app_path=args.mobile_app_path,
                                type=args.type,
                                yes_all=args.y)
    elif args.dataset == "ade20k":
        in_dataset = ADE20KDataset(input_data_path=input_data_path,
                                mobile_app_path=args.mobile_app_path,
                                type=args.type,
                                yes_all=args.y)

    selected_img_path = in_dataset.subsample(N=args.N, policy=args.subsampling_strategy)
    in_dataset.process_dataset(selected_img_path=selected_img_path)



if __name__ == '__main__':
    main()
