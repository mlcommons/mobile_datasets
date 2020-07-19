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
    def __init__(self, input_data_path, mobile_app_path, type, yes_all):
        self.yes_all = yes_all

        self.type = type
        self.new_img_size = None
        if self.type == "coco":
            self.new_img_size = (300, 300)


        self.mobile_app_path = mobile_app_path
        self.tmp_path = os.path.join(self.mobile_app_path, "tmp_dataset_script") # temporary folder
        self.out_img_path = os.path.join(self.tmp_path, "img")
        self.check_create_tmp_path()

        self.out_ann_path = os.path.join(self.mobile_app_path, "java", "org", "mlperf", "inference", "assets", self.type+"_val")
        if self.type == "imagenet":
            self.out_ann_path = self.out_ann_path + ".txt"
        elif self.type == "coco":
            self.out_ann_path = self.out_ann_path + ".pbtxt"

        self.input_data_path = input_data_path
        if self.input_data_path is None:
            self.download_dataset()



        # Parameters to mimic number of bbox of coco
        self.percentile = 10
        self.max_nbox_coco = 20
        self.min_nbox_coco = 1
        self.load_classes()




    def check_create_tmp_path(self):
        """
        Checks if self.tmp_path exists.
        If so, asks the user to either remove it. If user does not remove it, quit.
        If not, create self.tmp_path and self.out_img_path folders.
        """
        while os.path.isdir(self.tmp_path):
            if not self.yes_all:
                delete = input(f"{self.tmp_path} could not be created, folder already exists. Do you want to remove this folder? (y/n) \n")
            if self.yes_all or delete == "y":
                try:
                    rm_tmp = subprocess.run(["rm", "-r", self.tmp_path], check=True,
                                        stderr=subprocess.PIPE,
                                        universal_newlines=True)
                    logger.info(f"{self.tmp_path} has been removed.")
                except subprocess.CalledProcessError as e:
                    raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.stderr))

            elif delete == "n":
                logger.error("Cannot pursue without deleting folder.")
                sys.exit()
            else:
                logger.error("Please enter a valid answer (y or n).")
        logger.info(f"Creating {self.out_img_path} directory")
        os.makedirs(self.out_img_path)

    def load_classes(self):
        """
        Loads classes labels and indices depending on self.type
        """
        if self.type == "imagenet":
            logger.info("Loading imagenet classes")
            self.IMAGENET_CLASSES =  {v:k for (k,v) in eval(requests.get(IMAGENET_CLASSES_URL).text).items()}
            logger.debug(self.IMAGENET_CLASSES)
        if self.type == "coco":
            # list_coco_classes = (requests.get(COCO_CLASSES_URL).text).split("\n")
            # self.COCO_CLASSES = dict(zip(list_coco_classes, [i+1 for i in range(len(list_coco_classes))]))
            # self.COCO_CLASSES_reverse = dict(zip([i+1 for i in range(len(list_coco_classes))],list_coco_classes))
            import json
            logger.info(f"Downloading coco annotation classes to {self.tmp_path}...")
            zip_path, hdrs = urllib.request.urlretrieve(COCO_ANN_URL, os.path.join(self.tmp_path,
                                                                                   "annotations_trainval2017.zip"))
            logger.info(f"Extracting {zip_path} to temporary folder {self.tmp_path}...")
            with zipfile.ZipFile(f"{zip_path}", 'r') as z:
                z.extractall(f"{self.tmp_path}")
            annot_json_path =  os.path.join(self.tmp_path,
                                            "annotations", "instances_val2017.json")
            annot_json = json.load(open(annot_json_path, 'r'))
            categories = annot_json['categories']
            ids = list(map(lambda d: d['id'], categories))
            labels = list(map(lambda d: d['name'], categories))
            self.COCO_CLASSES_reverse = dict(zip(ids, labels))
            self.COCO_CLASSES = dict(zip(labels, ids))
            logger.debug(self.COCO_CLASSES)

            self.compute_n_box_per_img_coco(annot_json) # useful to match distribution of nb of bbox per img


    def compute_n_box_per_img_coco(self,coco_ann_dict):
        """
        n_box_per_img_coco : list of number of bbox per img in coco dataset
        """
        n_box_in_img_coco = defaultdict(int)
        for annot in coco_ann_dict["annotations"]:
            n_box_in_img_coco[annot["image_id"]] += 1
        self.n_box_per_img_coco = np.array(list(n_box_in_img_coco.values()))
        percentiles = [self.percentile*i for i in range(1, int(100/self.percentile))] # TODO: can be modified??should it be chosen by user?
        nbox_percentile = [np.percentile(self.n_box_per_img_coco, p) for p in percentiles]
        self.coco_percentile_groups = [[self.min_nbox_coco, nbox_percentile[0]]] + [[nbox_percentile[i], \
                                    nbox_percentile[i+1]] for i in range(0,len(nbox_percentile)-1)] + [[nbox_percentile[-1], self.max_nbox_coco+1]]
        logger.debug(f"percentile {self.percentile}, coco per grp {self.coco_percentile_groups}")

    def download_dataset(self):
        """
        Downloads dataset from a url to the temp folder self.tmp_path and updates self.input_data_path accordingly.
        """
        raise ValueError("input_data_path must not be None, or download_dtaset should be implemented")


    def process_dataset(self, selected_img_path):
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
        - Remove temporary folder (self.tmp_path)


        Args:
            selected_img_path: list[str]
                list of paths to images that we want to put in the new dataset
        """
        #### Process each image + write annotations in mobile_app txt file ####
        with open(self.out_ann_path,'w') as ann_file:
            for i, img_path in enumerate(selected_img_path):
                if self.type == "imagenet":
                    new_img_name = f"ILSVRC2012_val_{i+1:08}.JPEG"
                elif self.type == "coco":
                    new_img_name = f"{i+1:012}.jpg"

                self.process_single_img(img_path=img_path,
                                        new_img_path=os.path.join(self.out_img_path, new_img_name))
                self.write_annotation(ann_file=ann_file, img_path=img_path, new_img_name=new_img_name)



        #### Removes existing self.type/img folder from the phone ####
        mobile_dataset_path = os.path.join(os.sep,'sdcard','mlperf_datasets', self.type)
        phone_dataset = subprocess.run(["adb", "shell", "ls", mobile_dataset_path], stderr=subprocess.DEVNULL)
        if phone_dataset.returncode == 0:
            print(f"{mobile_dataset_path} exists. Its elements will be deleted.")
            delete = "n"
            while delete != "y":
                if not self.yes_all:
                    delete = input("Do you want to continue? [y/n] \n")
                if self.yes_all or delete == "y":
                    try:
                        subprocess.run(["adb", "shell", "rm", "-r", mobile_dataset_path], check=True,
                                             stderr=subprocess.PIPE, universal_newlines=True)
                        logger.info(f"{mobile_dataset_path} folder has been removed from the phone.")
                    except subprocess.CalledProcessError as e:
                        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.stderr))
                    break
                elif delete == "n":
                    logger.error("Cannot pursue without removing those elements.")
                    sys.exit()
                else:
                    logger.error("Please enter a valid answer (y or n).")

        #### Push to mobile ####
        logger.info(f"Creating {os.path.join(mobile_dataset_path,'img')} directory on the phone.")
        try:
            subprocess.run(["adb", "shell", "mkdir", "-p", os.path.join(mobile_dataset_path,"img")], check=True,
                            stderr=subprocess.PIPE, universal_newlines=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.stderr))

        logger.info(f"Pushing {self.out_img_path} to the phone at {mobile_dataset_path}")
        try:
            subprocess.check_call(["adb", "push", self.out_img_path, mobile_dataset_path], stdout=sys.stdout, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

        #### Remove temporary folder ####
        if not self.yes_all:
            remove_tmp_folder = input(f"Do you want to remove the temporary folder located at {self.tmp_path} which has been created by the script? [y/n] \n")
        if self.yes_all or remove_tmp_folder == 'y':
            subprocess.run(["rm", "-r", self.tmp_path], check=True)
            logger.info(f"{self.tmp_path} folder has been removed.")


    def process_single_img(self, img_path, new_img_path):
        """
        Processes a single image.
        If self.new_img_size is specified, rescales the image.
        Otherwise, just copies to the new path.
        Args:
            img_path: str
                path to the image to process
            new_img_path: str
                output path to the new image
        """
        if self.new_img_size is None:
            logger.debug(f"Copying {img_path} to \n {new_img_path}")
            shutil.copyfile(img_path,
                            new_img_path)
        else:
            #logger.debug(f"Rescaling {img_path} to shape {self.new_img_size} and save to \n {new_img_path}")
            img = cv2.imread(img_path)
            resized_img = cv2.resize(img, self.new_img_size, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(new_img_path, resized_img)

    def write_annotation(self, ann_file, img_path, new_img_name):
        """
        Write annotation of a given image, into the ann_file.
        Args:
            ann_file: io.TextIOWrapper
                annotation file where the final annotations are written
            img_path: str
                path to the image
            new_img_name: str
                name of the new image
        """
        raise NotImplementedError

    def subsample(self,):
        """
        Policy for subsampling.
        Returns:
            selected_img_path: list of paths to images that we want to put in the output dataset
        """
        raise NotImplementedError
