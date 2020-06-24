import argparse
import os
import random
import shutil
import urllib
import zipfile
import requests
import subprocess
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
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
python script.py --mobile_app_path=./mobile_app --N=300 --dataset=ADE20K --type=imagenet
export ANDROID_HOME=Path/to/SDK # Ex: $HOME/Android/Sdk
export ANDROID_NDK_HOME=Path/to/NDK # Ex: $ANDROID_HOME/ndk/(your version)
bazel-2.2.0 build -c opt --cxxopt='--std=c++14' \
    --fat_apk_cpu=x86,arm64-v8a,armeabi-v7a \
    //java/org/mlperf/inference:mlperf_app
adb install -r bazel-bin/java/org/mlperf/inference/mlperf_app.apk
```
"""

class InputDataset:
    """
    Class which represents the input dataset (e.g. ADE20K) that one wants to subsample from and reformat
    into a new type of dataset (e.g. imagenet).

    Different input datasets come with different formats. For example, ADE20K contains jpg images which are
    saved in different folders depending on their class (for example, images/training/a/abbey/ADE_train_00000970.jpg).
    When dealing with a new input dataset, one has to write a new class which inherits from InputDataset,
    and implement the corresponding methods.

    Attributes:
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
        self.load_classes()
        self.mobile_app_path = mobile_app_path

        self.tmp_path = os.path.join(self.mobile_app_path, "tmp_dataset_script") # temporary folder
        self.check_tmp_path()


        self.out_ann_path = os.path.join(self.mobile_app_path, "java", "org", "mlperf", "inference", "assets")
        self.out_img_path = os.path.join(self.tmp_path, "img")
        logger.info(f"Creating {self.out_img_path} directory")
        os.makedirs(self.out_img_path)

        self.input_data_path = input_data_path
        if self.input_data_path is None:
            self.download_dataset()

    def check_tmp_path(self):
        """
        Checks whether the given tmp_path already exists or not. If so, asks the user to either remove it
        or change the path.
        """
        while os.path.isdir(self.tmp_path):
            if not self.yes_all:
                choice_tmp = input(f"{self.tmp_path} already exists. Do you want to:\n 1. Remove this folder \n 2. Create another temporary folder? [1/2] \n")
            if self.yes_all or choice_tmp == "1":
                rm_tmp = subprocess.run(["rm", "-r", self.tmp_path],
                                        stderr=subprocess.PIPE,
                                        universal_newlines=True)
                if rm_tmp.returncode == 0:
                    logger.info(f"{self.tmp_path} has been removed.")
                else:
                    logger.error(f"{self.tmp_path} couldn't be removed.", rm_tmp.stderr)
            elif choice_tmp == "2":
                new_tmp = input("Enter a name for the new tmp folder: ")
                self.tmp_path = os.path.join(self.mobile_app_path, new_tmp)
            else:
                logger.error("Please enter a valid choice (1 or 2)")

    def load_classes(self):
        """
        Loads classes labels and indices depending on self.type
        """
        if self.type == "imagenet":
            url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
            logger.info("Loading imagenet classes")
            imagenet_simple_labels = eval(requests.get(url).text)
            logger.debug(imagenet_simple_labels)
            # TODO: Check that there is no error in those classes
            self.IMAGENET_CLASSES = dict(zip(imagenet_simple_labels, [i for i in range(1,len(imagenet_simple_labels)+1)]))

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
        #### Process data locally ####
        if self.type == "coco":
            out_ann_name = os.path.join(self.out_ann_path, "coco_val.pbtxt")
            raise NotImplementedError
        elif self.type == "imagenet":
            new_ann_path = os.path.join(self.out_ann_path, "imagenet_val.txt")
            with open(new_ann_path,'w') as new_ann_file:
                for i, img_path in enumerate(selected_img_path):
                    #self.convert_img(img_path): not useful for ADE20K (already jpg image) but might be necessary for other datasets
                    label = self.find_label(img_path)
                    new_img_name = f"ILSVRC2012_val_{i+1:08}.JPEG"
                    logger.debug("Copying {img_path} to \n {os.path.join(self.out_img_path, new_img_name)}")
                    shutil.copyfile(img_path,
                                    os.path.join(self.out_img_path, new_img_name))
                    new_ann_file.write(str(label) + "\n")

        #### Push to mobile ####
        mobile_dataset_path = os.path.join('sdcard','mlperf_datasets', self.type)

        # Removes existing imagenet folder from the phone
        phone_dataset = subprocess.run(["adb", "shell", "ls", mobile_dataset_path],
                                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                        universal_newlines=True)
        if phone_dataset.returncode == 0:
            print(f"{mobile_dataset_path} exists. Its elements will be deleted.")
            dataset_folder = phone_dataset.stdout.split('\n')[:-1]
            for folder in dataset_folder:
                folder_elements = subprocess.run(["adb", "shell", "ls", os.path.join(mobile_dataset_path, folder)],
                                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                                universal_newlines=True)
                print(f"{os.path.join(mobile_dataset_path, folder)} will be deleted.")
                if folder_elements.returncode == 0:
                    print(f"Elements of {os.path.join(mobile_dataset_path, folder)} (listed below) will be deleted.")
                    print(folder_elements.stdout)

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
                else:
                    logger.error("Please enter a valid choice (y or n)")

        logger.info(f"Creating {os.path.join(mobile_dataset_path,'img')} directory on the phone.")
        create_dir = subprocess.run(["adb", "shell", "mkdir", "-p", os.path.join(mobile_dataset_path,"img")])
        logger.info(f"Pushing {self.out_img_path} to the phone at {mobile_dataset_path}")
        push = subprocess.run(["adb", "push", self.out_img_path, mobile_dataset_path])

        assert create_dir.returncode == push.returncode == 0
        if not self.yes_all:
            remove_tmp_folder = input(f"Do you want to remove the temporary folder located at {self.tmp_path} created by the script? [y/n] \n")
        if self.yes_all or remove_tmp_folder == 'y':
            clean = subprocess.run(["rm", "-r", self.tmp_path])
            logger.info(f"{self.tmp_path} folder has been removed.")



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

    def download_dataset(self):
        DEV = True
        if DEV: # TODO: remove
            dataset_name = "ADE20K_subset"
            ADE20K_url =  'https://github.com/ctrnh/mlperf_misc/raw/master/ADE20K_subset.zip'
        else:
            dataset_name = "ADE20K_2016_07_26"
            ADE20K_url = "https://groups.csail.mit.edu/vision/datasets/ADE20K/ADE20K_2016_07_26.zip"

        logger.info(f"Downloading dataset {dataset_name} at {ADE20K_url} to temporary folder {self.tmp_path}...")
        zip_path, hdrs = urllib.request.urlretrieve(ADE20K_url, f"{self.tmp_path}/{dataset_name}.zip")

        logger.info(f"Extracting {zip_path} to temporary folder {self.tmp_path}...")
        with zipfile.ZipFile(f"{zip_path}", 'r') as z:
            z.extractall(f"{self.tmp_path}")
        self.input_data_path = zip_path[:-4]

    def subsample(self, N, policy="random"):
        """
        Subsamples from ADE20K: it considers all images which class intersects with imagenet classes.
        Args:
            N: number of wanted samples.
            policy: type of policy :"random", "balanced" (TODO)

        Returns:
            selected_img_path: list of path to images we want to keep in the new dataset
        """
        selected_img_path = []
        logger.debug("Find classes which are common to imagenet and ADE20K")
        for set_path in [self.train_path, self.val_path]:
            for letter in os.listdir(set_path):
                logger.debug(letter)
                letter_path = os.path.join(set_path, letter)
                ADE_letter_classes = os.listdir(letter_path)
                for cur_class in ADE_letter_classes:
                   spaced_class = " ".join(cur_class.split("_"))
                   if spaced_class in self.IMAGENET_CLASSES:
                       logger.debug(f"Keep {spaced_class} which is shared both by imagenet and ADE20K")
                       class_path = os.path.join(letter_path, cur_class)
                       for f in os.listdir(class_path):
                           if f.endswith(".jpg"):
                               img_path = os.path.join(class_path, f)
                               selected_img_path.append(img_path)
                               self.in_annotations[img_path] = spaced_class
        if policy == "random":
            if N < len(selected_img_path):
                selected_img_path = random.sample(selected_img_path, N)

        return selected_img_path

    def find_label(self, img_path):
        if self.type == "imagenet":
            label = self.IMAGENET_CLASSES[self.in_annotations[img_path]]
            logger.debug(f"Img {img_path}, is {self.in_annotations[img_path]}, imagenet label {label}")
        return label


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
    parser.add_argument("--dataset", type=str.lower, help="Kanter or ADE20K or other to implement", choices=["kanter", "ade20k"])
    parser.add_argument("-y", action="store_true", help="automatically answer yes to all questions. If on, the script may remove folders without permission.")
    args = parser.parse_args()

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
    
    selected_img_path = input_dataset.subsample(N=args.N)
    input_dataset.process_dataset(selected_img_path=selected_img_path)

if __name__ == '__main__':
    main()
