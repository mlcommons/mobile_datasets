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

import log

def process_dataset(source, target, selected_img_path):
    """
    Processes the input dataset to mimic the format of self.type.
    Tasks performed:
    - Process each single image by calling self.process_single_img.
    For imagenet, the new images should follow the ILSVRC2012 validation set format: images should be in JPEG,
    and named ILSVRC2012_val_{idx:08}.JPEG where idx starts at 1.
    - Pushes images to the mobile phone sdcard/mlperf_datasets/{self.type}/img folder.
    (Existing images of this folder are deleted.)
    - Replaces ./mobile_app/java/org/mlperf/inference/assets/{self.type annotation file}.txt with corresponding new annotations.
    For imagenet, the i-th line of the annotation .txt file contains the imagenet label for image ILSVRC2012_val_{i:08}.JPEG.
    - Remove temporary folder (target.tmp_path)


    Args:
        source: SourceDataset
        target: TargetDataset
        selected_img_path: list[str]
            list of paths to images that we want to put in the new dataset
    """
    #### Process each image + write annotations in mobile_app txt file ####
    with open(target.out_ann_path,'w') as ann_file:
        for i, img_path in enumerate(selected_img_path):
            new_img_name = target.format_img_name(i + 1)

            process_single_img(img_path=img_path, new_img_path=os.path.join(target.out_img_path, new_img_name), img_size = target.img_size)

            source.write_annotation(ann_file=ann_file, img_path=img_path, new_img_name=new_img_name)

    #### Removes existing source.type/img folder from the phone ####
    mobile_dataset_path = os.path.join(os.sep,'sdcard','mlperf_datasets', source.type)
    phone_dataset = subprocess.run(["adb", "shell", "ls", mobile_dataset_path], stderr=subprocess.DEVNULL)
    if phone_dataset.returncode == 0:
        print(f"{mobile_dataset_path} exists. Its elements will be deleted.")
        delete = "n"
        while delete != "y":
            if not target.force:
                delete = input("Do you want to continue? [y/n] \n")
            if target.force or delete == "y":
                try:
                    subprocess.run(["adb", "shell", "rm", "-r", mobile_dataset_path], check=True,
                                         stderr=subprocess.PIPE, universal_newlines=True)
                    logging.info(f"{mobile_dataset_path} folder has been removed from the phone.")
                except subprocess.CalledProcessError as e:
                    raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.stderr))
                break
            elif delete == "n":
                logging.error("Cannot pursue without removing those elements.")
                sys.exit()
            else:
                logging.error("Please enter a valid answer (y or n).")

    #### Push to mobile ####
    logging.info(f"Creating {os.path.join(mobile_dataset_path,'img')} directory on the phone.")
    try:
        subprocess.run(["adb", "shell", "mkdir", "-p", os.path.join(mobile_dataset_path,"img")], check=True,
                        stderr=subprocess.PIPE, universal_newlines=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.stderr))

    logging.info(f"Pushing {target.out_img_path} to the phone at {mobile_dataset_path}")
    try:
        subprocess.check_call(["adb", "push", target.out_img_path, mobile_dataset_path], stdout=sys.stdout, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

    #### Remove temporary folder ####
    if not target.force:
        remove_tmp_folder = input(f"Do you want to remove the temporary folder located at {target.tmp_path} which has been created by the script? [y/n] \n")
    if target.force or remove_tmp_folder == 'y':
        subprocess.run(["rm", "-r", target.tmp_path], check=True)
        logging.info(f"{target.tmp_path} folder has been removed.")


def process_single_img(img_path, new_img_path, img_size):
    """
    Processes a single image.
    If img_size is specified, rescales the image.
    Otherwise, just copies to the new path.
    Args:
        img_path: str
            path to the image to process
        new_img_path: str
            output path to the new image
    """
    if img_size is None:
        logging.debug(f"Copying {img_path} to \n {new_img_path}")
        shutil.copyfile(img_path,
                        new_img_path)
    else:
        #logging.debug(f"Rescaling {img_path} to shape {self.new_img_size} and save to \n {new_img_path}")
        img = cv2.imread(img_path)
        resized_img = cv2.resize(img, img_size, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(new_img_path, resized_img)

def main():
    log.setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_path", type=str, help="directory of input dataset (img + annotations). If None, download script", default=None)
    parser.add_argument("--mobile_app_path", type=str, help="path to root directory containing the mobile app repo")
    parser.add_argument("--N", type=int, help="number of samples wanted in the dataset")
    parser.add_argument("--type", type=str.lower, help="coco or imagenet", choices=["coco", "imagenet"])
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
    if args.type == "imagenet":
        target_dataset = ImageNet(mobile_app_path=args.mobile_app_path, force = args.y)

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

    selected_img_path = source_dataset.subsample(target_dataset, N=args.N, policy=args.subsampling_strategy)
    process_dataset(source_dataset, target_dataset, selected_img_path)

if __name__ == '__main__':
    main()
