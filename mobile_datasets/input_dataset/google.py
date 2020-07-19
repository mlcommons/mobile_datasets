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
import csv
import json
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# CONSTANTS
IMAGENET_CLASSES_URL = "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"
COCO_CLASSES_URL = "https://raw.githubusercontent.com/amikelive/coco-labels/master/coco-labels-2014_2017.txt"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


## For development, can use the 1st link, otherwise it'll download the entire dataset (3GB)
ADE20K_URL =  "https://github.com/ctrnh/mlperf_misc/raw/master/ADE20K_subset.zip"
#ADE20K_URL = "https://groups.csail.mit.edu/vision/datasets/ADE20K/ADE20K_2016_07_26.zip"
GOOGLE_CLASSES_URL = "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv"
GOOGLE_IMG_URL = "https://datasets.appen.com/appen_datasets/open-images/zip_files_copy/validation.zip"
GOOGLE_ANN_URL = "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv"

from .input_dataset import InputDataset

class GoogleDataset(InputDataset):
    def __init__(self, input_data_path, mobile_app_path, type, yes_all):
        super().__init__(input_data_path=input_data_path,
                          mobile_app_path=mobile_app_path,
                          type=type,
                          yes_all=yes_all)
        self.in_annotations = {}
        self.intersecting_g_class, self.intersecting_coco, self.intersecting_g_idx, self.mapping_g_coco = self.intersecting_classes(DATASET_CLASSES=self.GOOGLE_CLASSES)

    def download_dataset(self):
        # Download images
        # download annotations  os.path.join(self.input_data_path, GOOGLE_ANN_URL.split("/")[-1])
        # download descriptions-boxable.csv os.path.join(self.input_data_path, "class-descriptions-boxable.csv")

        raise NotImplementedError


    def load_classes(self):
        """
        Load ADE20K classes in addition to self.type classes.
        """
        super().load_classes()
        self.GOOGLE_CLASSES = {}
        self.GOOGLE_CLASSES_reverse = {}
        classes_file_path = os.path.join(self.input_data_path, "class-descriptions-boxable.csv")
        with open(classes_file_path, 'r') as csvfile:
            spamreader = csv.reader(csvfile)#, delimiter=' ', quotechar='|')
            for i,row in enumerate(spamreader):
                self.GOOGLE_CLASSES[row[1].lower()] = row[0]
                self.GOOGLE_CLASSES_reverse[row[0]] = row[1].lower()


    def intersecting_classes(self, DATASET_CLASSES):
        intersecting_data_class= set()
        intersecting_coco = set()
        intersecting_data_idx = set()
        mapping_data_coco = {}
        for data_class in DATASET_CLASSES.keys():
            for data_single_class in data_class.split(", "):
                for coco_class in self.COCO_CLASSES.keys():
                    for coco_single_class in coco_class.split(", "):
                        if data_single_class.lower() == coco_single_class:
                            intersecting_data_class.add(data_class)
                            intersecting_data_idx.add(DATASET_CLASSES[data_class])
                            intersecting_coco.add(coco_class)
                            mapping_data_coco[DATASET_CLASSES[data_class]] = coco_class
        return intersecting_data_class, intersecting_coco, intersecting_data_idx, mapping_data_coco

    def read_ann_csv(self):
        """
        keeps only img with at least 1 class in coco
        keeps images which respect params: TODO: code another way?
        """
        ann_csv_path = os.path.join(self.input_data_path, GOOGLE_ANN_URL.split("/")[-1])
        params = dict(zip(['IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsInside'],
                          ["01" for i in range(5)])) # TODO: img with which attributes should we keep?

        logger.info(f"Reading annotations from {ann_csv_path}")
        logger.info(f"Params {params}")
        img_to_delete = set()
        ann_dict = {}
        with open(ann_csv_path, 'r') as csvfile:
            content = csv.reader(csvfile)
            for i, row in enumerate(content):
                if i == 0:
                    col_titles = dict(zip(row, [j for j in range(len(row))]))
                else:
                    img_id = row[col_titles["ImageID"]] + ".jpg"
                    google_label = row[col_titles["LabelName"]]
                    if img_id not in img_to_delete and google_label in self.mapping_g_coco:
                        for attribute in params.keys():
                            if row[col_titles[attribute]] not in params[attribute]:
                                img_to_delete.add(img_id)
                        if img_id not in ann_dict:
                            ann_dict[img_id] = {}
                        obj_id = len(ann_dict[img_id].keys())
                        ann_dict[img_id][obj_id] = {}
                        ann_dict[img_id][obj_id]["label"] = self.mapping_g_coco[google_label]
                        ann_dict[img_id][obj_id]['normalized_bbox'] = [float(row[6]), float(row[7]), float(row[4]), float(row[5])]

        all_img = list(ann_dict.keys())
        for img_id in all_img:
            if img_id in img_to_delete:
                del ann_dict[img_id]
        return ann_dict

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
        logger.info(f"Subsampling google with a {policy} policy...")
        img_in_class = defaultdict(list)


        ann_dict = self.read_ann_csv()

        tmp_unselected = set()
        img_sort_percentiles = [[] for k in range(len(self.coco_percentile_groups))]

        #### Fetch all images which class intersect with self.type classes ####
        if self.type == "coco": #TODO: recoder propre
            for root, dirs, files in os.walk(self.input_data_path):
                for img_name in files:
                    if img_name in ann_dict.keys(): #if img_name.endswith(".jpg"):
                        img_path = os.path.join(root, img_name)
                        intersecting_img.append(img_path)
                        ann_img = ann_dict[img_name]
                        self.in_annotations[img_path] = { "objects": ann_dict[img_name] }
                        self.in_annotations[img_path]["number_bbox"] = len(ann_dict[img_name].keys())


                        for k in range(len(self.coco_percentile_groups)):
                            lower, upper = self.coco_percentile_groups[k]
                            keep = False
                            if lower <= self.in_annotations[img_path]["number_bbox"] < upper:
                                img_sort_percentiles[k].append(img_path)
                                keep = True
                        if not keep:
                            tmp_unselected.add(img_path)

                        # for obj in ann_img.keys():
                        #     if obj.startswith('obj_id'):
                        #         label_name = ann_img[obj]['label_name']
                        #         if label_name in self.mapping_g_coco:
                        #             self.in_annotations[img_path][obj] = {}
                        #             coco_idx = self.COCO_CLASSES[self.mapping_g_coco[label_name]]
                        #             self.in_annotations[img_path][obj]["label"] = coco_idx
                        #             #[top bo left right]
                        #             self.in_annotations[img_path][obj]["normalized_bbox"] = ann_img[obj]["nbbox"]


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

        if 1:#policy == "match_n_box_coco":
            n_img_per_percentile = ceil(N*self.percentile/100)
            logger.debug(f"n_img_per_percentile , {n_img_per_percentile}")
            debt = 0
            n_kept_img_grp = [0 for i in range(len(img_sort_percentiles))]

            for i in range(len(img_sort_percentiles)): # TODO: code better?
                n_img_grp = len(img_sort_percentiles[i])
                logger.debug(f"n_img_grp {i}-th grp:{n_img_grp}" )
                if n_img_grp < n_img_per_percentile:
                    n_kept_img_grp[i] += n_img_grp
                    debt += n_img_per_percentile - n_img_grp
                else:
                    n_kept_img_grp[i] += min(debt + n_img_per_percentile, n_img_grp)
                    debt -= (n_kept_img_grp[i] - n_img_per_percentile)

            if debt > 0:
                debt = 0
                for i in reversed(range(len(img_sort_percentiles))):
                    n_img_grp = len(img_sort_percentiles[i])
                    if n_img_grp < n_img_per_percentile:
                        #n_kept_img_grp[i] += n_img_grp
                        debt += n_img_per_percentile - n_img_grp
                    else:
                        n_img_give = min(n_img_grp - n_kept_img_grp[i], debt)
                        n_kept_img_grp[i] += n_img_give
                        debt -= n_img_give

            logger.debug(f"n_kept_img_grp {n_kept_img_grp}")

            for i in range(len(img_sort_percentiles)):
                selected_img_path += random.sample(img_sort_percentiles[i], n_kept_img_grp[i])

            logger.debug(f"Number selected_img_path {len(selected_img_path)}")

            if len(selected_img_path) <= N:
                selected_img_path += random.sample(tmp_unselected, N - len(selected_img_path))
                logger.debug(f"Added images out bound n_box_min/max : {N - len(selected_img_path)}")


            # Check the stats
            stats_g = []
            for img_path in selected_img_path:
                stats_g.append(self.in_annotations[img_path]["number_bbox"])
            stats_g = np.array(stats_g)
            for i in range(1, 20):
                p = 5*i
                logger.debug(f"percentile: {p}, g: {np.percentile(stats_g, p)}")
            logger.debug(f"mean:  g: {np.mean(stats_g)}")
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
        if self.type == "coco":
            ann_file.write("detection_results {\n")
            for obj in self.in_annotations[img_path]['objects'].keys():
                ann_file.write("  objects {\n")
                ann_file.write(f"    class_id: {self.in_annotations[img_path]['objects'][obj]['label']}\n")
                ann_file.write("    bounding_box {\n")
                ann_file.write(f"      normalized_top: {self.in_annotations[img_path]['objects'][obj]['normalized_bbox'][0]}\n")
                ann_file.write(f"      normalized_bottom: {self.in_annotations[img_path]['objects'][obj]['normalized_bbox'][1]}\n")
                ann_file.write(f"      normalized_left: {self.in_annotations[img_path]['objects'][obj]['normalized_bbox'][2]}\n")
                ann_file.write(f"      normalized_right: {self.in_annotations[img_path]['objects'][obj]['normalized_bbox'][3]}\n")
                ann_file.write("    }\n")
                ann_file.write("  }\n")
            ann_file.write(f'  image_name: "{new_img_name}"\n')
            ann_file.write(f'  image_id: {int(new_img_name.split(".")[0])}\n')
            ann_file.write("}\n")
