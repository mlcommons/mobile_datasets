import cv2
import sys
import argparse
import os
import random
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

# CONSTANTS
COCO_CLASSES_URL = "https://raw.githubusercontent.com/amikelive/coco-labels/master/coco-labels-2014_2017.txt"
COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

## For development, can use the 1st link, otherwise it'll download the entire dataset (3GB)
ADE20K_URL =  "https://github.com/ctrnh/mlperf_misc/raw/master/ADE20K_subset.zip"
#ADE20K_URL = "https://groups.csail.mit.edu/vision/datasets/ADE20K/ADE20K_2016_07_26.zip"
GOOGLE_CLASSES_URL = "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv"
GOOGLE_IMG_URL = "https://datasets.appen.com/appen_datasets/open-images/zip_files_copy/validation.zip"
GOOGLE_ANN_URL = "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv"

from .source_dataset import InputDataset

class GoogleDataset(InputDataset):
    def __init__(self, input_data_path, mobile_app_path, type, yes_all):
        super().__init__(input_data_path=input_data_path,
                          mobile_app_path=mobile_app_path,
                          type=type,
                          yes_all=yes_all)

        self.in_annotations = {}
        self.class_sep = ", "

        self.load_classes()

    def load_classes(self):
        """
        Load ADE20K classes in addition to self.type classes.
        """
        self.GOOGLE_CLASSES = {}
        self.GOOGLE_CLASSES_reverse = {}
        classes_file_path = os.path.join(self.input_data_path, "class-descriptions-boxable.csv")
        with open(classes_file_path, 'r') as csvfile:
            spamreader = csv.reader(csvfile)#, delimiter=' ', quotechar='|')
            for i,row in enumerate(spamreader):
                self.GOOGLE_CLASSES[row[1].lower()] = row[0]
                self.GOOGLE_CLASSES_reverse[row[0]] = row[1].lower()

    def intersecting_classes(self, target, DATASET_CLASSES):
        # TODO: modify coco/imagenet + move to input dataset level?
        if self.type == "coco":
            intersecting_data_class= set()
            intersecting_coco = set()
            intersecting_data_idx = set()
            mapping_data_coco = {}
            for data_class in DATASET_CLASSES.keys():
                for data_single_class in data_class.split(", "):
                    for coco_class in target.classes.keys():
                        for coco_single_class in coco_class.split(", "):
                            if data_single_class.lower() == coco_single_class:
                                intersecting_data_class.add(data_class)
                                intersecting_data_idx.add(DATASET_CLASSES[data_class])
                                intersecting_coco.add(coco_class)
                                mapping_data_coco[DATASET_CLASSES[data_class]] = coco_class
            return intersecting_data_class, intersecting_coco, intersecting_data_idx, mapping_data_coco
        elif self.type == "imagenet":
            intersecting_data_class= set()
            intersecting_imagenet = set()
            intersecting_data_idx = set()
            mapping_data_imagenet = {}
            for data_class in DATASET_CLASSES.keys():
                for data_single_class in data_class.split(", "):
                    for imagenet_class in target.IMAGENET_CLASSES.keys():
                        for imagenet_single_class in imagenet_class.split(", "):
                            if data_single_class.lower() == imagenet_single_class:
                                intersecting_data_class.add(data_class)
                                intersecting_data_idx.add(DATASET_CLASSES[data_class])
                                intersecting_imagenet.add(imagenet_class)
                                mapping_data_imagenet[DATASET_CLASSES[data_class]] = imagenet_class
            return intersecting_data_class, intersecting_imagenet, intersecting_data_idx, mapping_data_imagenet

        elif self.type == "ade20k":
            intersecting_data_class= set()
            intersecting_target = set()
            intersecting_data_idx = set()
            mapping_data_target = {}
            for data_class in DATASET_CLASSES.keys():
                for data_single_class in data_class.split(self.class_sep):
                    for target_class in target.classes.keys():
                        for target_single_class in target_class.split(target.class_sep):
                            if data_single_class.lower() == target_single_class:
                                intersecting_data_class.add(data_class)
                                intersecting_data_idx.add(DATASET_CLASSES[data_class])
                                intersecting_target.add(target_class)
                                mapping_data_target[DATASET_CLASSES[data_class]] = target_class
            return intersecting_data_class, intersecting_target, intersecting_data_idx, mapping_data_target


    def read_ann_csv(self, target):
        """
        This function reads Google annotation csv file in order to keep the information which interest us.
        Return:
            ann_dict: (dict)
                ann_dict[img_id] is a dict with object_id as key.
                Then ann_dict[img_id][object_id] is a dict which stores bbox, label and area of the object_id object inside the img_id image.
                An image is stored in ann_dict iff:
                    - it has at least 1 bbox intersecting with Targetdataset which has an area > 0.2 (hyperparameter TBD)
                    - it respects params (IsOccluded, IsTruncated etc) (Those are also hyperparameters TBD)
                    - For imagenet: this bbox is the only bbox annotated (only 1 significant object in image)
        """
        ann_csv_path = os.path.join(self.input_data_path, GOOGLE_ANN_URL.split("/")[-1])
        params = dict(zip(['IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsInside'],
                          ["01" for i in range(5)])) # TODO: img with which attributes should we keep?
        params["IsGroupOf"] = "0"
        logging.info(f"Reading annotations from {ann_csv_path}")
        logging.info(f"Params {params}")
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
                    if self.type == "imagenet":
                        if img_id not in img_to_delete and google_label in self.mapping_g_imagenet:
                            for attribute in params.keys():
                                if row[col_titles[attribute]] not in params[attribute]:
                                    img_to_delete.add(img_id)
                            if img_id not in ann_dict:
                                ann_dict[img_id] = {}
                                obj_id = 0
                                ann_dict[img_id][obj_id] = {}
                                ann_dict[img_id][obj_id]["label"] = target.IMAGENET_CLASSES[self.mapping_g_imagenet[google_label]]
                                ann_dict[img_id][obj_id]["normalized_bbox"] = {"top": float(row[6]),
                                                                               "bot": float(row[7]),
                                                                               "left": float(row[4]),
                                                                               "right": float(row[5])}
                                ann_dict[img_id][obj_id]["normalized_area"] = target.bbox_area(*ann_dict[img_id][obj_id]['normalized_bbox'].values())
                    elif self.type == "coco" or self.type == "ade20k":
                        if img_id not in img_to_delete and google_label in self.mapping_g_target:
                            #logging.debug(self.mapping_g_target[google_label])
                            for attribute in params.keys():
                                if row[col_titles[attribute]] not in params[attribute]:
                                    img_to_delete.add(img_id)
                            if img_id not in ann_dict:
                                ann_dict[img_id] = {}
                            obj_id = len(ann_dict[img_id].keys())
                            ann_dict[img_id][obj_id] = {}
                            ann_dict[img_id][obj_id]["label"] = target.classes[self.mapping_g_target[google_label]]
                            ann_dict[img_id][obj_id]["normalized_bbox"] = {"top": float(row[6]),
                                                                           "bot": float(row[7]),
                                                                           "left": float(row[4]),
                                                                           "right": float(row[5])}
                            ann_dict[img_id][obj_id]["normalized_area"] = target.bbox_area(*ann_dict[img_id][obj_id]['normalized_bbox'].values())

        all_img = list(ann_dict.keys())
        for img_id in all_img:
            if img_id in img_to_delete:
                del ann_dict[img_id]
            elif len(ann_dict[img_id].keys()) == 1 and ann_dict[img_id][0]["normalized_area"] < 0.2:
                ### Delete image if only 1 bbox which has small area
                #logging.debug(f"delete {img_id}, area bbox : {ann_dict[img_id][0]['normalized_area']}")
                del ann_dict[img_id]
        return ann_dict

    def subsample(self, target, N, policy="random"):
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
        logging.info(f"Subsampling google with a {policy} policy...")

        self.intersecting_g_class, self.intersecting_target, self.intersecting_g_idx, self.mapping_g_target = self.intersecting_classes(target, DATASET_CLASSES=self.GOOGLE_CLASSES)

        logging.debug(f"nb intersecting classes : {len(self.intersecting_g_class)} intersecting g_class {self.intersecting_g_class}")

        ann_dict = self.read_ann_csv(target)
        if self.type == "imagenet":
            img_in_class = defaultdict(list)
            for root, dirs, files in os.walk(self.input_data_path):
                for img_name in files:
                    if img_name in ann_dict.keys(): #if img_name.endswith(".jpg"):
                        img_path = os.path.join(root, img_name)
                        self.in_annotations[img_path] = ann_dict[img_name][0]['label']
                        intersecting_img.append(img_path)

        elif self.type == "ade20k":
            img_in_class = defaultdict(list)
            for root, dirs, files in os.walk(self.input_data_path):
                for img_name in files:
                    if img_name in ann_dict.keys(): #if img_name.endswith(".jpg"):
                        img_path = os.path.join(root, img_name)
                        intersecting_img.append(img_path)

        elif self.type == "coco": #TODO: recoder propre
            tmp_unselected = set()
            img_sort_percentiles = [[] for k in range(len(target.coco_percentile_grp))]
            for root, dirs, files in os.walk(self.input_data_path):
                for img_name in files:
                    if img_name in ann_dict.keys(): #if img_name.endswith(".jpg"):
                        img_path = os.path.join(root, img_name)
                        intersecting_img.append(img_path)
                        self.in_annotations[img_path] = { "objects": ann_dict[img_name],
                                                          "number_bbox": len(ann_dict[img_name].keys()) }

                        for k in range(len(target.coco_percentile_grp)):
                            lower, upper = target.coco_percentile_grp[k]
                            keep = False
                            if lower <= self.in_annotations[img_path]["number_bbox"] < upper:
                                areas = list(map(lambda obj_id: ann_dict[img_name][obj_id]["normalized_area"], ann_dict[img_name].keys()))
                                #logging.debug(f"area {areas}: list of norm area for each obj in img ")
                                diff_size = abs(target.coco_mean_area_percentile_grp[k] - np.mean(areas))
                                img_sort_percentiles[k].append([img_path, diff_size])
                                keep = True
                        if not keep:
                            tmp_unselected.add(img_path)

        #### Subsampling from images which intersect ####
        logging.info(f"Number of intersecting images : {len(intersecting_img)}")
        if N >= len(intersecting_img):
            logging.info("Number of intersecting images < N(Number of images we want to keep): keeping all intersecting images.")
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

        if self.type == "coco":#policy == "match_n_box_coco":
            n_img_per_percentile = ceil(N*target.percentile/100)
            logging.debug(f"n_img_per_percentile , {n_img_per_percentile}")
            debt = 0
            n_kept_img_grp = [0 for i in range(len(img_sort_percentiles))]

            for i in range(len(img_sort_percentiles)): # TODO: code better?
                n_img_grp = len(img_sort_percentiles[i])
                logging.debug(f"n_img_grp {i}-th grp:{n_img_grp}" )
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

            logging.debug(f"n_kept_img_grp {n_kept_img_grp}")

            for i in range(len(img_sort_percentiles)):
                img_sort_percentiles_grp = sorted(img_sort_percentiles[i],key= lambda pair:pair[1] )
                # selected_img_path += random.sample(img_sort_percentiles[i], n_kept_img_grp[i])
                selected_img_path += list(map(lambda p:p[0],img_sort_percentiles_grp[:n_kept_img_grp[i]]))

            logging.debug(f"Number selected_img_path {len(selected_img_path)}")

            if len(selected_img_path) <= N:
                selected_img_path += random.sample(tmp_unselected, N - len(selected_img_path))
                logging.debug(f"Added images out bound n_box_min/max : {N - len(selected_img_path)}")


            # Check the stats
            stats_g = []
            for img_path in selected_img_path:
                # print(img_path)
                # print(self.in_annotations[img_path])
                # print(self.in_annotations[img_path]["number_bbox"])
                stats_g.append(self.in_annotations[img_path]["number_bbox"])
            stats_g = np.array(stats_g)
            for i in range(1, 20):
                p = 5*i
                logging.debug(f"percentile: {p}, g: {np.percentile(stats_g, p)}")
            logging.debug(f"mean:  g: {np.mean(stats_g)}")
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
                ann_file.write(f"      normalized_top: {self.in_annotations[img_path]['objects'][obj]['normalized_bbox']['top']}\n")
                ann_file.write(f"      normalized_bottom: {self.in_annotations[img_path]['objects'][obj]['normalized_bbox']['bot']}\n")
                ann_file.write(f"      normalized_left: {self.in_annotations[img_path]['objects'][obj]['normalized_bbox']['left']}\n")
                ann_file.write(f"      normalized_right: {self.in_annotations[img_path]['objects'][obj]['normalized_bbox']['right']}\n")
                ann_file.write("    }\n")
                ann_file.write("  }\n")
            ann_file.write(f'  image_name: "{new_img_name}"\n')
            ann_file.write(f'  image_id: {int(new_img_name.split(".")[0])}\n')
            ann_file.write("}\n")
        elif self.type == "imagenet":
            label = self.in_annotations[img_path]
            logging.debug(f"Img {img_path}, imagenet label {label}")
            ann_file.write(str(label) + "\n")
        elif self.type == "ade20k":
            pass
