import logging
import os
from enum import Enum
import numpy as np
import utils

class TargetDataset:
    """
    Class which represents the target dataset (e.g. coco) that one wants to mimic.
    Attributes:
        force: bool
            if True, answers yes to all questions asked by the script (such as permission to remove folders)
        mobile_app_path: str
            path to the folder containing the mobile_app repo
        tmp_path: str
            path to a temporary folder which will be created and removed at the end of the process
        out_ann_path: str
            path to the folder which contains the annotations files (in mobile_app repo)
        min_nbox/max_nbox: int
            respectively minimum and maximum number of bounding boxes wanted per image in the new
        percentile: int
            percentiles of number of bounding boxes per image that we want to match to target (used in self.compute_percentile_grp)

    """
    def __init__(self, mobile_app_path, tmp_path, force = False):

        self.force = force
        self.name = ""
        self.img_size = None
        self.out_ann_path = ""

        self.mobile_app_path = mobile_app_path
        self.tmp_path = tmp_path

        utils.check_remove_dir(self.tmp_path, force = force)
        os.mkdir(self.tmp_path)


        self.min_normalized_bbox_area = 0.2
        self.class_sep = ", "
        self.classification = False

        self.min_nbox = 1
        self.max_nbox = np.inf
        self.percentile = 100
        self.nbox_percentile_grp = [[self.min_nbox, self.max_nbox]]
        self.mean_area_percentile_grp = None

    def __str__(self):
        return ''

    def compute_percentile_grp(self, list_n_box_per_img=None):
        """
        Given a list of number of bboxes, computes self.nbox_percentile_grp
        Args:
            list_n_box_per_img: list of int
                list_n_box_per_img[i] = number of bbox contained in image i
        Example:
        self.percentile = 25
        self.nbox_percentile_grp = [[1,4], [4,8], [8,10], [10,50]]
        4 is the 25-percentile of list_n_box_per_img
        """
        if self.percentile == 100:
            self.nbox_percentile_grp = [[self.min_nbox, self.max_nbox+1]]
        else:
            percentiles = [self.percentile*i for i in range(1, int(100/self.percentile))]
            nbox_percentile = [np.percentile(list_n_box_per_img, p) for p in percentiles]
            self.nbox_percentile_grp = [[self.min_nbox, nbox_percentile[0]]] + [[nbox_percentile[i], \
                                        nbox_percentile[i+1]] for i in range(0,len(nbox_percentile)-1)] + [[nbox_percentile[-1], self.max_nbox+1]]
        logging.info(f"percentile {self.percentile}, percentile groups: {self.nbox_percentile_grp}")

    def load_classes(self):
        raise NotImplementedError

    def format_img_name(self, name):
        raise NotImplementedError


    def write_annotation(self, transformation_annotations, ann_file, img_path, new_img_name):
        """
        Write annotation of a given image, into the ann_file.
        Args:
            transformation_annotations: dict
                dict containing annotations corresponding to img_path
            ann_file: io.TextIOWrapper
                annotation file where the final annotations are written
            img_path: str
                path to the image
            new_img_name: str
                name of the new image
        """
        raise NotImplementedError
