
"""
Description:
This script takes as input a dataset (by default ADE20K). After subsampling this dataset,
it formats it so as to mimic "coco" or "imagenet" dataset for the mobile_app.
More specifically, it replaces the existing annotation file from mobile_app with
a new annotation file (corresponding to the new dataset) having the same format.
Then it pushes images of the new dataset to the mobile phone.

Remarks:
- coco not implemented yet
- Input dataset must be either ADE20K or kanter


Example list of commands for using ADE20K as classification test dataset:
```
git clone https://github.com/mlperf/mobile_app.git
python script.py --mobile_app_path=./mobile_app --N=300 --dataset=ADE20K --type=imagenet --subsampling_strategy=random
export ANDROID_HOME=Path/to/SDK # Ex: $HOME/Android/Sdk
export ANDROID_NDK_HOME=Path/to/NDK # Ex: $ANDROID_HOME/ndk/(your version)
bazel-2.2.0 build -c opt --cxxopt='--std=c++14' \
    --fat_apk_cpu=x86,arm64-v8a,armeabi-v7a \
    //java/org/mlperf/inference:mlperf_app
adb install -r bazel-bin/java/org/mlperf/inference/mlperf_app.apk
```
"""

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
ADE20K_URL =  'https://github.com/ctrnh/mlperf_misc/raw/master/ADE20K_subset.zip' ## For development, can use this link, otherwise it'll download the entire dataset (3GB)
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

        self.mobile_app_path = mobile_app_path
        self.tmp_path = os.path.join(self.mobile_app_path, "tmp_dataset_script") # temporary folder
        self.check_tmp_path()

        self.out_ann_path = os.path.join(self.mobile_app_path, "java", "org", "mlperf", "inference", "assets", self.type+"_val")
        if self.type == "imagenet":
            self.out_ann_path = self.out_ann_path + ".txt"
        elif self.type == "coco":
            self.out_ann_path = self.out_ann_path + ".pbtxt"

        self.out_img_path = os.path.join(self.tmp_path, "img")
        logger.info(f"Creating {self.out_img_path} directory")
        os.makedirs(self.out_img_path)

        self.input_data_path = input_data_path
        if self.input_data_path is None:
            self.download_dataset()
        self.load_classes()

    def check_tmp_path(self):
        """
        Checks whether the given tmp_path already exists or not. If so, asks the user to either remove it
        or change the path.
        """
        while os.path.isdir(self.tmp_path):
            if not self.yes_all:
                delete = input(f"{self.tmp_path} could not be created, folder already exists. Do you want to remove this folder? (y/n) \n")
            if self.yes_all or delete == "y":
                rm_tmp = subprocess.run(["rm", "-r", self.tmp_path],
                                        stderr=subprocess.PIPE,
                                        universal_newlines=True)
                if rm_tmp.returncode == 0:
                    logger.info(f"{self.tmp_path} has been removed.")
                else:
                    logger.error(f"{self.tmp_path} couldn't be removed.", rm_tmp.stderr)
            elif delete == "n":
                logger.error("Cannot pursue without deleting folder.")
            else:
                logger.error("Please enter a valid answer (y or n).")

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
            categories = json.load(open(annot_json_path, 'r'))['categories']
            ids = list(map(lambda d: d['id'], categories))
            labels = list(map(lambda d: d['name'], categories))
            self.COCO_CLASSES_reverse = dict(zip(ids, labels))
            self.COCO_CLASSES = dict(zip(labels, ids))
            logger.debug(self.COCO_CLASSES)
    def download_dataset(self):
        """
        Downloads dataset from a url to the temp folder self.tmp_path and updates self.input_data_path accordingly.
        """
        raise ValueError("input_data_path must not be None")


    def process_dataset(self,
                       selected_img_path
                       ):
        """
        Processes the input dataset to mimic the format of self.type.
        Tasks performed:
        - Copies, renames images from selected_img_path to temporary folder self.out_img_path.
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

        with open(self.out_ann_path,'w') as new_ann_file:
            for i, img_path in enumerate(selected_img_path):
                if self.type == "imagenet":
                    new_img_name = f"ILSVRC2012_val_{i+1:08}.JPEG"
                elif self.type == "coco":
                    new_img_name = f"{i+1:012}.jpg"
                logger.debug(f"Copying {img_path} to \n {os.path.join(self.out_img_path, new_img_name)}")
                shutil.copyfile(img_path,
                                os.path.join(self.out_img_path, new_img_name))

                self.write_annotation(new_ann_file=new_ann_file, img_path=img_path, new_img_name=new_img_name)



        #### Push to mobile ####
        mobile_dataset_path = os.path.join(os.sep,'sdcard','mlperf_datasets', self.type)

        # Removes existing imagenet folder from the phone
        phone_dataset = subprocess.run(["adb", "shell", "ls", mobile_dataset_path],
                                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                        universal_newlines=True)
        if phone_dataset.returncode == 0:
            print(f"{mobile_dataset_path} exists. Its elements will be deleted.")

            delete = "n"
            while delete != "y":
                if not self.yes_all:
                    delete = input("Do you want to continue? [y/n] \n")
                if self.yes_all or delete == "y":
                    rm_type = subprocess.run(["adb", "shell", "rm", "-r", mobile_dataset_path],
                                             stderr=subprocess.PIPE, universal_newlines=True)
                    if rm_type.returncode == 0:
                        logger.info(f"{mobile_dataset_path} folder has been removed from the phone.")
                    else:
                        logger.error(f"Cannot remove {mobile_dataset_path}.", rm_type.stderr)
                    break
                elif delete == "n":
                    logger.error("Cannot pursue without removing those elements.")
                    sys.exit()
                else:
                    logger.error("Please enter a valid answer (y or n).")

        logger.info(f"Creating {os.path.join(mobile_dataset_path,'img')} directory on the phone.")
        create_dir = subprocess.run(["adb", "shell", "mkdir", "-p", os.path.join(mobile_dataset_path,"img")])
        logger.info(f"Pushing {self.out_img_path} to the phone at {mobile_dataset_path}")

        try:
            subprocess.check_call(["adb", "push", self.out_img_path, mobile_dataset_path], stdout=sys.stdout, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

        assert create_dir.returncode == 0
        if not self.yes_all:
            remove_tmp_folder = input(f"Do you want to remove the temporary folder located at {self.tmp_path} which has been created by the script? [y/n] \n")
        if self.yes_all or remove_tmp_folder == 'y':
            clean = subprocess.run(["rm", "-r", self.tmp_path])
            logger.info(f"{self.tmp_path} folder has been removed.")


            #TODO: remove find_label
    def find_label(self, img_path):
        """
        Given an image path, returns the ground-truth class.
        Returns:
            label: index of corresponding class in type dataset
        """
        raise NotImplementedError

    def subsample(self,):
        """
        Policy for subsampling.
        Returns:
            selected_img_path: list of paths to images that we want to put in the output dataset
        """
        raise NotImplementedError




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
        Checks if ade_class belongs to imagenet. If so, returns the corresponding imagenet class index.
        """
        spaced_class = " ".join(ade_class.split("_"))
        for imagenet_class in self.IMAGENET_CLASSES:
           if spaced_class in imagenet_class.split(", "):
               logger.debug(f"ADE20K class: {spaced_class} is in imagenet: {imagenet_class.split(', ')}")
               return self.IMAGENET_CLASSES[imagenet_class]
        return None

    def is_in_coco(self, ade_class_idx):
        """
        Checks if ade_class belongs to coco. If so, returns the corresponding coco class index.
        """
        if ade_class_idx == 0:
            return None

        for single_label in self.ADE20K_CLASSES_reverse[ade_class_idx].split(", "):
            if single_label in self.COCO_CLASSES.keys():
                return self.COCO_CLASSES[single_label]
        return None


    def download_dataset(self):
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
            N: number of wanted samples.
            policy: type of policy: "random", "balanced".
                    "random": subsamples N images from the images class intersects with imagenet classes.
                    "balanced": subsamples N images from the images class intersects with imagenet classes, while keeping
                                the frequencies of each class.

        Returns:
            selected_img_path: list of path to images we want to keep in the new dataset
        """

        intersecting_img = []
        logger.info(f"Subsampling ADE20K with a {policy} policy...")
        img_in_class = defaultdict(list)
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
                                for r in range(n_row):
                                    for c in range(n_col):
                                        if class_seg_img[r,c] in label_correspondance:
                                            object_id = instance_seg_img[r,c]
                                            if object_id not in self.in_annotations[img_path]:
                                                self.in_annotations[img_path][object_id] = {}
                                                self.in_annotations[img_path][object_id]["label"] = label_correspondance[class_seg_img[r,c]]
                                                self.in_annotations[img_path][object_id]["normalized_bbox"] = [r/n_row, r/n_row, c/n_col, c/n_col]
                                            else:
                                                self.in_annotations[img_path][object_id]["normalized_bbox"] = [min(self.in_annotations[img_path][object_id]["normalized_bbox"][0], r/n_row),
                                                                                                               max(self.in_annotations[img_path][object_id]["normalized_bbox"][1], r/n_row),
                                                                                                               min(self.in_annotations[img_path][object_id]["normalized_bbox"][2], c/n_col),
                                                                                                               max(self.in_annotations[img_path][object_id]["normalized_bbox"][3], c/n_col)]
                                intersecting_img.append(img_path)

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

    def write_annotation(self, new_ann_file, img_path, new_img_name):
        """
        Write annotations for img_path
        à mettre dans ade20k class"""
        if self.type == "imagenet":
            label = self.in_annotations[img_path]
            logger.debug(f"Img {img_path}, imagenet label {label}")
            new_ann_file.write(str(label) + "\n")
        elif self.type == "coco":
            new_ann_file.write("detection_results {\n")
            for obj in self.in_annotations[img_path].keys():
                new_ann_file.write("  objects {\n")
                new_ann_file.write(f"    class_id: {self.in_annotations[img_path][obj]['label']}\n")
                new_ann_file.write("    bounding_box {\n")
                new_ann_file.write(f"      normalized_top: {self.in_annotations[img_path][obj]['normalized_bbox'][0]}\n")
                new_ann_file.write(f"      normalized_bottom: {self.in_annotations[img_path][obj]['normalized_bbox'][1]}\n")
                new_ann_file.write(f"      normalized_left: {self.in_annotations[img_path][obj]['normalized_bbox'][2]}\n")
                new_ann_file.write(f"      normalized_right: {self.in_annotations[img_path][obj]['normalized_bbox'][3]}\n")
                new_ann_file.write("    }\n")
                new_ann_file.write("  }\n")
            new_ann_file.write(f'  image_name: "{new_img_name}"\n')
            new_ann_file.write(f'  image_id: {int(new_img_name.split(".")[0])}\n')
            new_ann_file.write("}\n")

class KanterDataset(InputDataset):
    def __init__(self, input_data_path, mobile_app_path, type,yes_all):
        super().__init__(input_data_path=input_data_path,
                        mobile_app_path=mobile_app_path,
                        type=type,
                        yes_all=yes_all)
        self.in_img_path = os.path.join(self.input_data_path, "img")
        self.in_annotations = json.load(open(os.path.join(self.input_data_path, "annotations", "labels.json"), 'r'))


    def subsample(self, N, policy="random"):
        """Policy for selecting images.

        Args:
            N: number of images we want to select
            policy: subsampling policy (TODO)
        Returns:
            list of selected image paths
        """
        if policy == "random":
            all_img_names = [f for f in os.listdir(self.in_img_path) \
                            if f.lower().endswith(("jpg", "png", "jpeg"))]
            if N < len(all_img_names):
                selected_img = random.sample(all_img_names, N)
            else:
                selected_img = all_img_names
        return list(map(lambda x:os.path.join(self.in_img_path, x), selected_img))

    def find_label(self, img_path):
        if self.type == "imagenet":
            img_name = img_path.split("/")[-1]
            label = self.IMAGENET_CLASSES[self.in_annotations[img_name]]
        return label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_path", type=str, help="directory of input dataset (img + annotations). If None, download script", default=None)
    parser.add_argument("--mobile_app_path", type=str, help="path to root directory containing the mobile app repo")
    parser.add_argument("--N", type=int, help="number of samples wanted in the dataset")
    parser.add_argument("--type", type=str.lower, help="coco or imagenet", choices=["coco", "imagenet"])
    parser.add_argument("--dataset", type=str.lower, default="ade20k", help="Kanter or ADE20K or other to implement", choices=["kanter", "ade20k"])
    parser.add_argument("-y", action="store_true", help="automatically answer yes to all questions. If on, the script may remove folders without permission.")
    parser.add_argument("--subsampling_strategy", type=str.lower, help="random or balanced", choices=["random", "balanced"], default="random")
    args = parser.parse_args()

    adb_devices = subprocess.check_output(["adb", "devices"], universal_newlines=True)
    assert len(adb_devices.split('\n')) > 3, "No device attached. Please connect your phone."
    logger.info(adb_devices)

    input_data_path = args.input_data_path
    if args.dataset == "kanter":
        import json
        input_dataset = KanterDataset(input_data_path=input_data_path,
                                mobile_app_path=args.mobile_app_path,
                                type=args.type,
                                yes_all=args.y)
    elif args.dataset == "ade20k":
        input_dataset = ADE20KDataset(input_data_path=input_data_path,
                                mobile_app_path=args.mobile_app_path,
                                type=args.type,
                                yes_all=args.y)

    selected_img_path = input_dataset.subsample(N=args.N, policy=args.subsampling_strategy)
    input_dataset.process_dataset(selected_img_path=selected_img_path)



if __name__ == '__main__':
    main()
