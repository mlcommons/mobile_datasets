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
import subprocess
from math import ceil
import shutil

class Transformation:
    def __init__(self, source, target):
        self.source = source
        self.target = target

        self.out_img_path = os.path.join(self.target.tmp_path, "img")
        logging.info(f"Creating {self.out_img_path} directory")
        os.makedirs(self.out_img_path)

        self.all_annotations = {}
        self.intersecting_classes()

    def intersecting_classes(self):
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
            delete = "n"
            while delete != "y":
                if not self.target.force:
                    delete = input("Do you want to continue? [y/n] \n")
                if self.target.force or delete == "y":
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

        logging.info(f"Pushing {self.out_img_path} to the phone at {mobile_dataset_path}")
        try:
            subprocess.check_call(["adb", "push", self.out_img_path, mobile_dataset_path], stdout=sys.stdout, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

    def remove_tmp_folder(self):
        #### Remove temporary folder ####
        if not self.target.force:
            remove_tmp_folder = input(f"Do you want to remove the temporary folder located at {self.target.tmp_path} which has been created by the script? [y/n] \n")
        if self.target.force or remove_tmp_folder == 'y':
            subprocess.run(["rm", "-r", self.target.tmp_path], check=True)
            logging.info(f"{self.target.tmp_path} folder has been removed.")


    def process_single_img(self, img_path, new_img_path, img_size):
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

    def create_all_annotations(self):
        """
        Creates self.all_annotations.
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
        self.all_annotations keeps only images which have classes belonging both to source and target dataset,
        and images where target.min_nbox <= number_bbox <= target.max_nbox
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
        logging.info(f"Subsampling with a {policy} policy...")
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


        if policy == "random":
            selected_img_path = random.sample(selected_img_path, N)

        elif policy == "balanced":
            if self.target.name == "imagenet":
                #img_in_class[imagenet_label].append(img_path)
                nb_total_img = len(selected_img_path)
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

    def compute_stats_new_dataset(self, selected_img_path):
        # Computes stats
        stats_new_dataset = np.zeros((len(selected_img_path,)))
        for i, img_path in enumerate(selected_img_path):
            stats_new_dataset[i] = self.all_annotations[img_path]["number_bbox"]
        logging.info("New dataset statistics:")
        for i in range(1, 20):
            p = 5*i
            logging.info(f"- Number of bboxes per image for {p}-th percentile: {np.percentile(stats_new_dataset, p)}")
        logging.info(f"- Mean of number of bboxes per image in new dataset: {np.mean(stats_new_dataset)}")


    def transform(self, N, policy):
        """
        DOC OF FORMER PROCESS_DATASET(removed):
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
        self.source.create_ann_dict()
        selected_img_path = self.subsample(N=N, policy=policy)
        self.compute_stats_new_dataset(selected_img_path)

        #### Process each image + write annotations in mobile_app txt file ####
        with open(self.target.out_ann_path,'w') as ann_file:
            for i, img_path in enumerate(selected_img_path):
                new_img_name = self.target.format_img_name(i + 1)
                self.process_single_img(img_path=img_path, new_img_path=os.path.join(self.out_img_path, new_img_name), img_size = self.target.img_size)
                self.target.write_annotation(transformation_annotations=self.all_annotations, ann_file=ann_file, img_path=img_path, new_img_name=new_img_name)

        self.push_to_mobile()
        self.remove_tmp_folder()
