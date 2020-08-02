import sys
import argparse
import logging
import subprocess
import os
import shutil
import cv2

from source_dataset.ade20k import ADE20KDataset
from source_dataset.google import GoogleDataset

from target_dataset.coco import Coco
from target_dataset.imagenet import ImageNet
from target_dataset.ade20k_target import ADE20K_target

from transformation.transformation import Transformation
import log


def main():
    log.setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_path", type=str, help="directory of input dataset (img + annotations). If None, download script", default=None)
    parser.add_argument("--mobile_app_path", type=str, help="path to root directory containing the mobile app repo")
    parser.add_argument("--N", type=int, help="number of samples wanted in the dataset")
    parser.add_argument("--type", type=str.lower, help="coco, imagenet, or ade20k", choices=["coco", "imagenet", "ade20k"])
    parser.add_argument("--dataset", type=str.lower, default="ade20k", help="Kanter or ADE20K or other to implement", choices=["kanter", "ade20k","google"])
    parser.add_argument("-y", action="store_true", help="automatically answer yes to all questions. If on, the script may remove folders without permission.")
    parser.add_argument("--subsampling_strategy", type=str.lower, help="random or balanced", choices=["random", "balanced"], default="random")
    args = parser.parse_args()

    adb_devices = subprocess.run(["adb", "devices"], stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout.strip().split('\n')

    if len(adb_devices) < 2:
        logging.error("No device attached. Please connect your phone.")
        sys.exit()
    elif len(adb_devices) > 2:
        logging.error("Multiple devices connected:")
        for dev in adb_devices[1:]:
            logging.error("\t" + dev)
        logging.error("Script expects a single device.")
        sys.exit()

    [device_name, device_status] = adb_devices[1].split()

    if device_status == "unauthorized":
        logging.error("Please enable USB debugging.")
        sys.exit()

    logging.info("Found device: " + device_name)

    if args.type == "coco":
        target_dataset = Coco(mobile_app_path=args.mobile_app_path, force = args.y)
    elif args.type == "imagenet":
        target_dataset = ImageNet(mobile_app_path=args.mobile_app_path, force = args.y)
    elif args.type == "ade20k":
        target_dataset = ADE20K_target(mobile_app_path=args.mobile_app_path, force = args.y)

    input_data_path = args.input_data_path
    if args.dataset == "kanter":
        source_dataset = KanterDataset(input_data_path=input_data_path,
                                mobile_app_path=args.mobile_app_path,
                                type=args.type,
                                yes_all=args.y)
    elif args.dataset == "ade20k":
        source_dataset = ADE20KDataset(input_data_path=input_data_path,
                                mobile_app_path=args.mobile_app_path,
                                type=args.type,
                                yes_all=args.y)
    elif args.dataset == "google":
        source_dataset = GoogleDataset(input_data_path=input_data_path,
                                mobile_app_path=args.mobile_app_path,
                                type=args.type,
                                yes_all=args.y)

    transformation = Transformation(source=source_dataset, target=target_dataset)
    transformation.transform(N=args.N, policy=args.subsampling_strategy)

if __name__ == '__main__':
    main()
