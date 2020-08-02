import logging
import os
from enum import Enum
import numpy as np
import utils

class SubsamplingPolicy(Enum):
    random = 1
    balanced = 2

class TargetDataset:
    def __init__(self, mobile_app_path, force = False):
        self.force = force
        self.name = ""
        self.img_size = None
        self.out_ann_path = ""
        self.classes_url = ""
        self.ann_url = ""
        self.dataset_classes = []


        self.mobile_app_path = mobile_app_path
        self.tmp_path = os.path.join(self.mobile_app_path, "tmp_dataset_script") # temporary folder
        self.out_img_path = os.path.join(self.tmp_path, "img")
        utils.check_remove_dir(self.tmp_path, force = force)

        logging.info(f"Creating {self.out_img_path} directory")
        os.makedirs(self.out_img_path)

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
            ann_file: io.TextIOWrapper
                annotation file where the final annotations are written
            img_path: str
                path to the image
            new_img_name: str
                name of the new image
        """
        raise NotImplementedError
