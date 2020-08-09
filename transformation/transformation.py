import logging
import urllib
import os
import zipfile
import json
import requests
import numpy as np
import random
import cv2

import sys
from enum import Enum
import subprocess
from math import ceil
import shutil

import utils

class SubsamplingPolicy(Enum):
    random = 1
    balanced = 2


class Transformation:
    """
    Class which represents the transformation between the source dataset and the target dataset.
    Attributes:
        out_img_path: str
            path to the temporary folder where the script will dump the new dataset images before pushing to phone
        all_annotations: dict
            to be created with self.create_all_annotations
    """
    def __init__(self, source, target):
        self.source = source
        self.target = target

        self.out_img_path = os.path.join(self.target.tmp_path, "img")
        logging.info(f"Creating {self.out_img_path} directory")
        os.makedirs(self.out_img_path)

        self.all_annotations = {}
        self.intersecting_classes()

    def intersecting_classes(self):
        """
        Finds intersecting classes between source and target classes.
        """
        self.intersecting_source_class = set()
        self.intersecting_target = set()
        self.intersecting_source_idx = set()
        self.mapping_source_target = {}
        for source_class in self.source.classes.keys():
            for source_single_class in source_class.split(self.source.class_sep):
                for target_class in self.target.classes.keys():
                    for target_single_class in target_class.split(self.target.class_sep):
                        if source_single_class.lower() == target_single_class:
                            self.intersecting_source_class.add(source_class)
                            self.intersecting_source_idx.add(self.source.classes[source_class])
                            self.intersecting_target.add(target_class)
                            self.mapping_source_target[self.source.classes[source_class]] = target_class
        logging.debug(f"Number of intersecting classes : {len(self.intersecting_source_class)} intersecting source classes are: {self.intersecting_source_class}")


    def push_to_mobile(self,):
        #### Removes existing source.type/img folder from the phone ####
        mobile_dataset_path = os.path.join(os.sep,'sdcard','mlperf_datasets', str(self.target))
        phone_dataset = subprocess.run(["adb", "shell", "ls", mobile_dataset_path], stderr=subprocess.DEVNULL)
        if phone_dataset.returncode == 0:
            print(f"{mobile_dataset_path} exists. Its elements will be deleted.")

            utils.check_remove_dir(path=mobile_dataset_path, force=self.target.force, mobile_shell=True)

        #### Push to mobile ####
        logging.info(f"Creating {os.path.join(mobile_dataset_path,'img')} directory on the phone.")
        try:
            subprocess.run(["adb", "shell", "mkdir", "-p", os.path.join(mobile_dataset_path,"img")], check=True,
                            stderr=subprocess.PIPE, universal_newlines=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.stderr))

        logging.info(f"Pushing {self.out_img_path} to the phone at {mobile_dataset_path}")
        try:
            subprocess.check_call(["adb", "push", self.out_img_path, mobile_dataset_path], stdout=sys.stdout, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))


    def create_all_annotations(self):
        """
        Creates self.all_annotations from the information of source.ann_dict.
        self.all_annotations keeps only images which have classes belonging both to source and target dataset, and images where target.min_nbox <= number_bbox <= target.max_nbox

        Structure of the dict:
        self.all_annotations = {img_path: (dict)
                                    {"objects": (list)
                                          [{"normalized_bbox": (dict) {"top": (float), "bot":,...},
                                           "target_label": (int),
                                           "source_label": (int),
                                           "normalized_area": float
                                           },
                                           ...],
                                     "number_bbox": (int)
                                    }
                                }

        """
        ann_dict = self.source.ann_dict

        img_sort_percentiles = [[] for k in range(len(self.target.nbox_percentile_grp))]

        for root, dirs, files in os.walk(self.source.input_data_path):
            for img_name in files:
                if img_name in ann_dict.keys():
                    img_objects = []
                    for obj_id in ann_dict[img_name].keys():
                        if ann_dict[img_name][obj_id]['source_label'] in self.mapping_source_target:
                            img_objects.append(ann_dict[img_name][obj_id])
                            img_objects[-1]["target_label"] = self.target.classes[self.mapping_source_target[ann_dict[img_name][obj_id]['source_label']]]

                    number_bbox = len(img_objects)
                    if not (number_bbox == 1 and img_objects[0]["normalized_area"] < self.target.min_normalized_bbox_area): # at least 1 bbox with non negligible area
                        for idx_grp in range(len(self.target.nbox_percentile_grp)):
                            lower, upper = self.target.nbox_percentile_grp[idx_grp]
                            keep = False
                            if lower <= number_bbox < upper:
                                keep = True
                                img_path = os.path.join(root, img_name)
                                areas = list(map(lambda obj_id: img_objects[obj_id]["normalized_area"], [obj_id for obj_id in range(len(img_objects))]))
                                #logging.debug(f"area {areas}: list of norm area for each obj in img ")
                                if self.target.mean_area_percentile_grp is not None:
                                    diff_area = abs(self.target.mean_area_percentile_grp[idx_grp] - np.mean(areas))
                                else:
                                    diff_area = 0
                                img_sort_percentiles[idx_grp].append([img_path, diff_area])
                                if img_path not in self.all_annotations.keys():
                                    self.all_annotations[img_path] = {}
                                self.all_annotations[img_path]['objects'] = img_objects
                                self.all_annotations[img_path]['number_bbox'] = number_bbox

        return img_sort_percentiles

    def compute_n_img_kept_grp(self, n_img_per_percentile, img_sort_percentiles):
        """
        This function is useful when self.target.percentile != 100. It is used when we try to match the distribution of number of bounding boxes of the target dataset.
        Args:
            img_sort_percentiles: list of tuple of length number of percentile groups
                img_sort_percentiles[i] = (img_path, diff_area) (diff_area = absolute value of difference between normalized area of bbox in img and normalized area of bbox of target dataset)
            n_img_per_percentile: int
                number of images we ideally would like to have in each percentile group
        Returns:
            n_kept_img_grp: list of int
                n_kept_img_grp[i] = number of images we will keep from group i in the final dataset. Ideally, should be close to n_img_per_percentile
        """
        debt = 0
        n_kept_img_grp = [0 for i in range(len(img_sort_percentiles))]

        for i in range(len(img_sort_percentiles)):
            n_img_grp = len(img_sort_percentiles[i])
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
                    debt += n_img_per_percentile - n_img_grp
                else:
                    n_img_give = min(n_img_grp - n_kept_img_grp[i], debt)
                    n_kept_img_grp[i] += n_img_give
                    debt -= n_img_give
        return n_kept_img_grp


    def subsample(self, N):
        """
        Subsamples from the source dataset.
        Args:
            N: int
                number of wanted samples.

        Returns:
            selected_img_path: list of path to images we want to keep in the new dataset
        """
        logging.info(f"Subsampling the new dataset...")
        img_sort_percentiles = self.create_all_annotations()
        selected_img_path = list(self.all_annotations.keys())
        logging.info(f"Number of images which have intersecting classes between source and target classes : {len(selected_img_path)}")

        if N >= len(selected_img_path):
            logging.info(f"Number of intersecting images < N(Number of images we want to keep): keeping all intersecting images. ({len(selected_img_path)})")
            return selected_img_path

        if self.target.percentile != 100:
            selected_img_path = []
            logging.info(f"Matching number of bbox distribution of {self.target.name} for each {self.target.percentile} percentile.")
            n_img_per_percentile = ceil(N*self.target.percentile/100)
            n_kept_img_grp = self.compute_n_img_kept_grp(n_img_per_percentile=n_img_per_percentile, img_sort_percentiles=img_sort_percentiles)
            logging.debug(f"Number of images kept in each percentile group: {n_kept_img_grp}. Ideally, it should match the Number of images per percentile group wanted in new dataset: {n_img_per_percentile}")

            for i in range(len(img_sort_percentiles)):
                img_sort_percentiles_grp = sorted(img_sort_percentiles[i],key=lambda pair:pair[1]) # Sorting per diff_area (closest to target dataset area)
                selected_img_path += list(map(lambda p:p[0],img_sort_percentiles_grp[:n_kept_img_grp[i]]))
            logging.debug(f"Number of selected images after matching number of bbox of target dataset: {len(selected_img_path)}")


        return random.sample(selected_img_path, N)

    def compute_stats_new_dataset(self, selected_img_path):
        """
        Computes statistics of number of bbox per image in the new dataset.
        """
        stats_new_dataset = np.zeros((len(selected_img_path,)))
        for i, img_path in enumerate(selected_img_path):
            stats_new_dataset[i] = self.all_annotations[img_path]["number_bbox"]
        logging.info("New dataset statistics:")
        for i in range(1, 20):
            p = 5*i
            logging.info(f"- Number of bboxes per image for {p}-th percentile: {np.percentile(stats_new_dataset, p)}")
        logging.info(f"- Mean of number of bboxes per image in new dataset: {np.mean(stats_new_dataset)}")


    def transform(self, N):
        """
        Transforms the source dataset into the target.
        Args:
            N: number of wanted images in new dataset
        """
        self.source.create_ann_dict()
        selected_img_path = self.subsample(N=N)
        self.compute_stats_new_dataset(selected_img_path)

        #### Process each image + write annotations in mobile_app txt file ####
        with open(self.target.out_ann_path,'w') as ann_file:
            for i, img_path in enumerate(selected_img_path):
                new_img_name = self.target.format_img_name(i + 1)
                utils.process_single_img(img_path=img_path, new_img_path=os.path.join(self.out_img_path, new_img_name), img_size = self.target.img_size)
                self.target.write_annotation(transformation_annotations=self.all_annotations, ann_file=ann_file, img_path=img_path, new_img_name=new_img_name)

        self.push_to_mobile()
        utils.check_remove_dir(path=self.target.tmp_path, force=self.target.force, remove_required=False)
