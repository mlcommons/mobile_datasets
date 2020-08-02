import logging
import urllib
import os
import zipfile
import json
import utils

import numpy as np

from .target_dataset import TargetDataset

class Coco(TargetDataset):
    def __init__(self, mobile_app_path, force = False):
        super().__init__(mobile_app_path=mobile_app_path,
                         force=force)
        self.name = "coco"
        self.out_ann_path = os.path.join(self.mobile_app_path, "java", "org", "mlperf", "inference", "assets", "coco_val.pbtxt")
        self.img_size = (300, 300)


        # Parameters to mimic number of bbox of coco
        self.percentile = 25
        self.max_nbox = 50
        self.min_nbox = 1

        self.load_classes()

    def __str__(self):
        return "coco"

    def load_classes(self):
        coco_ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

        logging.info(f"Downloading coco annotation classes to {self.tmp_path}...")
        zip_path, hdrs = urllib.request.urlretrieve(coco_ann_url, os.path.join(self.tmp_path, "annotations_trainval2017.zip"))
        logging.info(f"Extracting {zip_path} to temporary folder {self.tmp_path}...")
        with zipfile.ZipFile(f"{zip_path}", 'r') as z:
            z.extractall(f"{self.tmp_path}")
        annot_json = json.load(open( os.path.join(self.tmp_path, "annotations", "instances_val2017.json"), 'r'))
        categories = annot_json['categories']
        ids = list(map(lambda d: d['id'], categories))
        labels = list(map(lambda d: d['name'], categories))
        self.classes_reverse = dict(zip(ids, labels))
        self.classes = dict(zip(labels, ids))
        logging.debug(f"Classes of target dataset {self.name} are {self.classes}")

        self.compute_stats_coco(annot_json) # useful to match distribution of nb of bbox per img



    def compute_stats_coco(self,coco_ann_dict):
        """
        Computes statistics about coco bounding boxes that we want to mimic.
        -> computes number of bbox percentiles
        -> computes average of normalized bbox area for each group
        """

        # Create dict area_img_coco: area_img_coco[img_id] = area of image = width of img * height of img
        area_img_coco = {}
        for img_dict in coco_ann_dict["images"]:
            area_img_coco[img_dict["id"]] = img_dict["width"]*img_dict["height"]

        # Create dict dict_img_coco: dict_img_coco[img_id] = [number of bbox in img, average of normalized bbox over all bboxes in img]
        dict_img_coco = {}
        for annot in coco_ann_dict["annotations"]:
            if annot["image_id"] not in dict_img_coco:
                dict_img_coco[annot["image_id"]] = [0,0]

            # normalized area of annot bbox
            norm_area_annot = annot["bbox"][2]*annot["bbox"][3] / area_img_coco[annot["image_id"]]

            # add to average of normalized bbox over all bboxes in the img
            dict_img_coco[annot["image_id"]][1] *= dict_img_coco[annot["image_id"]][0]/(dict_img_coco[annot["image_id"]][0]+1)
            dict_img_coco[annot["image_id"]][1] += norm_area_annot/(dict_img_coco[annot["image_id"]][0]+1)

            # increment number of bboxes in the img
            dict_img_coco[annot["image_id"]][0] += 1


        sorted_by_nbox = sorted(list(dict_img_coco.values()), key=lambda el:el[0])

        # Compute ideal percentiles of number of bbox
        n_box_per_img_coco = np.array(list(map(lambda x: x[0], sorted_by_nbox)))
        self.compute_percentile_grp(list_n_box_per_img=n_box_per_img_coco)

        # Compute mean of normalized bbox area for each group of percentile_grp
        n_img_per_grp = int(len(dict_img_coco.keys())*self.percentile/100)
        mean_area_per_img_coco = np.array(list(map(lambda x: x[1], sorted_by_nbox)))
        self.mean_area_percentile_grp = []
        for i in range(len(self.nbox_percentile_grp)):
            self.mean_area_percentile_grp.append(np.mean(mean_area_per_img_coco[i*n_img_per_grp:(i+1)*n_img_per_grp]))
        logging.debug(f"coco_mean_area_percentile_grp {self.mean_area_percentile_grp}")


    def write_annotation(self, transformation_annotations, ann_file, img_path, new_img_name): # move coco
        ann_file.write("detection_results {\n")
        for obj in transformation_annotations[img_path]['objects'].keys():
            ann_file.write("  objects {\n")
            ann_file.write(f"    class_id: {transformation_annotations[img_path]['objects'][obj]['target_label']}\n")
            ann_file.write("    bounding_box {\n")
            ann_file.write(f"      normalized_top: {transformation_annotations[img_path]['objects'][obj]['normalized_bbox']['top']}\n")
            ann_file.write(f"      normalized_bottom: {transformation_annotations[img_path]['objects'][obj]['normalized_bbox']['bot']}\n")
            ann_file.write(f"      normalized_left: {transformation_annotations[img_path]['objects'][obj]['normalized_bbox']['left']}\n")
            ann_file.write(f"      normalized_right: {transformation_annotations[img_path]['objects'][obj]['normalized_bbox']['right']}\n")
            ann_file.write("    }\n")
            ann_file.write("  }\n")
        ann_file.write(f'  image_name: "{new_img_name}"\n')
        ann_file.write(f'  image_id: {int(new_img_name.split(".")[0])}\n')
        ann_file.write("}\n")


    def format_img_name(self, name):
        return f"{name:012}.jpg"
