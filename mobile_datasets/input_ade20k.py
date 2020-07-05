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

class ADE20KDataset(InputDataset):
    def __init__(self, input_data_path, mobile_app_path, type, yes_all):
        super().__init__(input_data_path=input_data_path,
                          mobile_app_path=mobile_app_path,
                          type=type,
                          yes_all=yes_all)
        self.train_path = os.path.join(self.input_data_path, "images", "training")
        self.val_path = os.path.join(self.input_data_path, "images", "validation")
        self.in_annotations = {}

    def load_classes(self):
        """
        Load ADE20K classes in addition to self.type classes.
        """
        super().load_classes()
        from scipy.io import loadmat
        mat_path = os.path.join(self.input_data_path, "index_ade20k.mat")
        object_names = loadmat(mat_path)['index']['objectnames'][0][0][0]
        self.ADE20K_CLASSES = {}
        self.ADE20K_CLASSES_reverse = {}
        for i in range(len(object_names)):
            self.ADE20K_CLASSES[object_names[i][0]] = i+1
            self.ADE20K_CLASSES_reverse[i+1] = object_names[i][0]
            if i == 3:
                logger.debug(f"ADE20K_CLASSES: {self.ADE20K_CLASSES}")
                logger.debug(f"ADE20K_CLASSES_reverse: {self.ADE20K_CLASSES_reverse}")

    def is_in_imagenet(self, ade_class):
        """
        Checks if ade_class belongs to imagenet.
        If so, returns the corresponding imagenet class index.
        Args:
            ade_class: str
                ADE20K class label (name of the parent folder to the image in ADE20K dataset)
        Returns:
            None if the class does not intersect with imagenet classes
            (int) imagenet label index otherwise
        """
        spaced_class = " ".join(ade_class.split("_"))
        for imagenet_class in self.IMAGENET_CLASSES:
           if spaced_class in imagenet_class.split(", "):
               logger.debug(f"ADE20K class: {spaced_class} is in imagenet: {imagenet_class.split(', ')}")
               return self.IMAGENET_CLASSES[imagenet_class]
        return None

    def is_in_coco(self, ade_class_idx):
        """
        Checks if class corresponding to ade_class_idx belongs to coco.
        If so, returns the corresponding coco_val class index.
        Args:
            ade_class: int
                ADE20K class label index
        Returns:
            None if the class does not intersect with coco classes
            (int) coco label index otherwise
        """
        if ade_class_idx == 0:
            return None
        for single_label in self.ADE20K_CLASSES_reverse[ade_class_idx].split(", "):
            if single_label in self.COCO_CLASSES.keys():
                return self.COCO_CLASSES[single_label]
        return None


    def download_dataset(self):
        """
        Downloads ADE20K to temporary folder.
        """
        dataset_name = ADE20K_URL.split("/")[-1].split(".")[0]
        req = urllib.request.Request(ADE20K_URL, method="HEAD")
        size_file = urllib.request.urlopen(req).headers["Content-Length"]
        download = "n"
        while download != "y":
            if not self.yes_all:
                download = input(f"You are about to download {dataset_name} ({size_file} bytes) to the temporary folder {self.tmp_path}. Do you want to continue? [y/n] \n")
            if self.yes_all or download == "y":
                logger.info(f"Downloading dataset {dataset_name} at {ADE20K_URL} to temporary folder {self.tmp_path}...")
                zip_path, hdrs = urllib.request.urlretrieve(ADE20K_URL, f"{self.tmp_path}/{dataset_name}.zip")
                logger.info(f"Extracting {zip_path} to temporary folder {self.tmp_path}...")
                with zipfile.ZipFile(f"{zip_path}", 'r') as z:
                    z.extractall(f"{self.tmp_path}")
                self.input_data_path = zip_path[:-4]
                break
            elif download == "n":
                logger.error(f"Cannot pursue without downloading the dataset.")
                sys.exit()
            else:
                logger.error("Please enter a valid answer (y or n).")


    def subsample(self, N, policy="random"):
        """
        Subsamples from ADE20K: it considers all images which class intersects with imagenet classes.
        Args:
            N: int
                number of wanted samples.
            policy: ["random", "balanced"]
                type of policy can be:
                "random": randomly subsamples N images from images which class intersects with self.type classes.
                "balanced": (imagenet only) subsamples N images from images which class intersects with imagenet classes,
                            while keeping the frequencies of each class.

        Returns:
            selected_img_path: list of path to images we want to keep in the new dataset
        """
        intersecting_img = []
        logger.info(f"Subsampling ADE20K with a {policy} policy...")
        img_in_class = defaultdict(list)

        #### Fetch all images which class intersect with self.type classes ####
        for set_path in [self.train_path, self.val_path]:
            for letter in os.listdir(set_path):
                logger.debug(f"Folder with letter {letter}")
                letter_path = os.path.join(set_path, letter)
                for cur_class in os.listdir(letter_path):
                    class_path = os.path.join(letter_path, cur_class)

                    if self.type == "imagenet":
                        imagenet_label = self.is_in_imagenet(ade_class=cur_class)
                        if imagenet_label is not None:
                            for file in [f for f in os.listdir(class_path) if f.endswith(".jpg")]:
                                img_path = os.path.join(class_path, file)
                                intersecting_img.append(img_path)
                                self.in_annotations[img_path] = imagenet_label
                                img_in_class[imagenet_label].append(img_path)

                    elif self.type == "coco":
                        for seg_file in [f for f in os.listdir(class_path) if f.endswith('_seg.png')]:
                            img_path = os.path.join(class_path, seg_file[:-8] + ".jpg")
                            seg_img = plt.imread(os.path.join(class_path, seg_file))

                            n_row, n_col = seg_img.shape[0], seg_img.shape[1]
                            instance_seg_img = (seg_img[:,:,2]*255).astype("int")
                            class_seg_img = (seg_img[:,:,0]*255/10*256 + seg_img[:,:,1]*255).astype("int")
                            label_correspondance = {}
                            for ade_idx in np.unique(class_seg_img):
                                coco_label = self.is_in_coco(ade_idx)
                                if coco_label is not None:
                                    label_correspondance[ade_idx] = coco_label
                            if label_correspondance:
                                logger.debug(f"{img_path}, label correspondance {label_correspondance}")
                                self.in_annotations[img_path] = {}
                                # scaling factors
                                sr = 1/n_row
                                sc = 1/n_col
                                for r in range(n_row):
                                    for c in range(n_col):
                                        if class_seg_img[r,c] in label_correspondance:
                                            object_id = instance_seg_img[r,c]
                                            if object_id not in self.in_annotations[img_path]:
                                                self.in_annotations[img_path][object_id] = {}
                                                self.in_annotations[img_path][object_id]["label"] = label_correspondance[class_seg_img[r,c]]
                                                self.in_annotations[img_path][object_id]["normalized_bbox"] = [r*sr, r*sr, c*sc, c*sc]
                                            else:
                                                self.in_annotations[img_path][object_id]["normalized_bbox"] = [min(self.in_annotations[img_path][object_id]["normalized_bbox"][0], r*sr),
                                                                                                               max(self.in_annotations[img_path][object_id]["normalized_bbox"][1], r*sr),
                                                                                                               min(self.in_annotations[img_path][object_id]["normalized_bbox"][2], c*sc),
                                                                                                               max(self.in_annotations[img_path][object_id]["normalized_bbox"][3], c*sc)]
                                intersecting_img.append(img_path)

        #### Subsampling from images which intersect ####
        logger.info(f"Number of intersecting images : {len(intersecting_img)}")
        if N >= len(intersecting_img):
            logger.info("Number of intersecting images < N(Number of images we want to keep): keeping all intersecting images.")
            return intersecting_img

        if policy == "random":
            selected_img_path = random.sample(intersecting_img, N)

        elif policy == "balanced":
            if self.type == "imagenet":
                nb_total_img = len(intersecting_img)
                selected_img_path = []
                print(f"Number of total images {nb_total_img}")
                for cur_class in img_in_class.keys():
                    nb_img_in_class = len(img_in_class[cur_class])
                    new_nb_img_in_class = min(nb_img_in_class, ceil((nb_img_in_class*N) / nb_total_img))
                    selected_img_path += random.sample(img_in_class[cur_class], new_nb_img_in_class)
                    print(f"Class {cur_class}, nb img in class {nb_img_in_class}, new nb img {new_nb_img_in_class}")
                    print(f"Frequency = {nb_img_in_class/nb_total_img}")

                selected_img_path = random.sample(selected_img_path, N)
            else:
                raise NotImplementedError
        return selected_img_path

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
        if self.type == "imagenet":
            label = self.in_annotations[img_path]
            logger.debug(f"Img {img_path}, imagenet label {label}")
            ann_file.write(str(label) + "\n")
        elif self.type == "coco":
            ann_file.write("detection_results {\n")
            for obj in self.in_annotations[img_path].keys():
                ann_file.write("  objects {\n")
                ann_file.write(f"    class_id: {self.in_annotations[img_path][obj]['label']}\n")
                ann_file.write("    bounding_box {\n")
                ann_file.write(f"      normalized_top: {self.in_annotations[img_path][obj]['normalized_bbox'][0]}\n")
                ann_file.write(f"      normalized_bottom: {self.in_annotations[img_path][obj]['normalized_bbox'][1]}\n")
                ann_file.write(f"      normalized_left: {self.in_annotations[img_path][obj]['normalized_bbox'][2]}\n")
                ann_file.write(f"      normalized_right: {self.in_annotations[img_path][obj]['normalized_bbox'][3]}\n")
                ann_file.write("    }\n")
                ann_file.write("  }\n")
            ann_file.write(f'  image_name: "{new_img_name}"\n')
            ann_file.write(f'  image_id: {int(new_img_name.split(".")[0])}\n')
            ann_file.write("}\n")
