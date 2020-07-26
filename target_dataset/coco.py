import logging
import urllib
import os
import zipfile
import json

import numpy as np

from .target_dataset import TargetDataset

class Coco(TargetDataset):
    def __init__(self, mobile_app_path, force = False):
        super().__init__(mobile_app_path=mobile_app_path,
                         force=force)
        self.name = "Coco"
        self.in_annotations = {}
        self.img_size = (300, 300)

        self.out_ann_path = os.path.join(self.mobile_app_path, "java", "org", "mlperf", "inference", "assets", "coco_val.pbtxt")

        # Parameters to mimic number of bbox of coco
        self.percentile = 25
        self.max_nbox_coco = 50
        self.min_nbox_coco = 1

        self.load_classes()

    def load_classes(self):
            coco_ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

            logging.info(f"Downloading coco annotation classes to {self.tmp_path}...")
            zip_path, hdrs = urllib.request.urlretrieve(coco_ann_url, os.path.join(self.tmp_path,
                                                                                   "annotations_trainval2017.zip"))
            logging.info(f"Extracting {zip_path} to temporary folder {self.tmp_path}...")
            with zipfile.ZipFile(f"{zip_path}", 'r') as z:
                z.extractall(f"{self.tmp_path}")
            annot_json_path =  os.path.join(self.tmp_path,
                                            "annotations", "instances_val2017.json")
            annot_json = json.load(open(annot_json_path, 'r'))
            categories = annot_json['categories']
            ids = list(map(lambda d: d['id'], categories))
            labels = list(map(lambda d: d['name'], categories))
            self.coco_classes_reverse = dict(zip(ids, labels))
            self.coco_classes = dict(zip(labels, ids))
            logging.debug(self.coco_classes)

            self.compute_n_box_per_img_coco(annot_json) # useful to match distribution of nb of bbox per img

    def bbox_area(self, bot, top, right, left):
        #TODO: move to utils
        return (bot - top) * (right - left)

    def compute_n_box_per_img_coco(self,coco_ann_dict):
        """
        n_box_per_img_coco : list of number of bbox per img in coco dataset
        """
        area_img_coco = {}
        for img_dict in coco_ann_dict["images"]:
            area_img_coco[img_dict["id"]] = img_dict["width"]*img_dict["height"]

        dict_img_coco = {}
        for annot in coco_ann_dict["annotations"]:
            if annot["image_id"] not in dict_img_coco:
                dict_img_coco[annot["image_id"]] = [0,0]
            # norm area, average over bbox in an img
            norm_area_annot = annot["bbox"][2]*annot["bbox"][3] / area_img_coco[annot["image_id"]]
            dict_img_coco[annot["image_id"]][1] *= dict_img_coco[annot["image_id"]][0]/(dict_img_coco[annot["image_id"]][0]+1)
            dict_img_coco[annot["image_id"]][1] += norm_area_annot/(dict_img_coco[annot["image_id"]][0]+1)
            dict_img_coco[annot["image_id"]][0] += 1 # number of bbox

        sorted_by_nbox = sorted(list(dict_img_coco.values()), key=lambda el:el[0])
        n_box_per_img_coco = np.array(list(map(lambda x: x[0], sorted_by_nbox)))
        percentiles = [self.percentile*i for i in range(1, int(100/self.percentile))] # TODO: can be modified??should it be chosen by user?
        nbox_percentile = [np.percentile(n_box_per_img_coco, p) for p in percentiles]
        self.coco_percentile_grp = [[self.min_nbox_coco, nbox_percentile[0]]] + [[nbox_percentile[i], \
                                    nbox_percentile[i+1]] for i in range(0,len(nbox_percentile)-1)] + [[nbox_percentile[-1], self.max_nbox_coco+1]]
        logging.debug(f"percentile {self.percentile}, coco per grp {self.coco_percentile_grp}")

        n_img_per_grp = int(len(dict_img_coco.keys())*self.percentile/100)
        mean_area_per_img_coco =  np.array(list(map(lambda x: x[1], sorted_by_nbox)))
        self.coco_mean_area_percentile_grp = []
        for i in range(len(self.coco_percentile_grp)):
            self.coco_mean_area_percentile_grp.append(np.mean(mean_area_per_img_coco[i*n_img_per_grp:(i+1)*n_img_per_grp]))
        logging.debug(f"coco_mean_area_percentile_grp {self.coco_mean_area_percentile_grp}")

    def format_img_name(self, name):
        return f"{name:012}.jpg"
