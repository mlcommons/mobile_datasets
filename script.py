import argparse
import os
import random
import shutil
import urllib
import zipfile
import requests
import subprocess

"""
Description:
This script takes as input a dataset (by default ADE20K). After subsampling this dataset,
it formats it so as to mimic "coco" or "imagenet" dataset for the mobile_app.
More specifically, it replaces the existing annotation file from mobile_app with
a new annotation file (corresponding to the new dataset) having the same format.
Then it pushes images of the new dataset to the mobile phone.

Remarks:
- coco not implemented yet
- If the user wants to use an input dataset which is not ADE20K or kanter,
they will need to implement a new class which inherits from InputDataset.


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

parser = argparse.ArgumentParser()
parser.add_argument("--input_data_path", help="directory of input dataset (img + annotations). If None, download script", default=None)
parser.add_argument("--mobile_app_path", help="path to root directory containing the mobile app repo")
parser.add_argument("--N", type=int, help="number of samples wanted in the dataset")
parser.add_argument("--type", help="coco or imagenet")
parser.add_argument("--dataset", help="Kanter or ADE20K or other to implement", default="ade20k")
args = parser.parse_args()


class InputDataset:
    def __init__(self, input_data_path, mobile_app_path, type):
        assert type in ["coco", "imagenet"], "type must be coco or imagenet"
        self.type = type
        self.mobile_app_path = mobile_app_path

        self.tmp_path = os.path.join(self.mobile_app_path, "tmp_dataset_script") # temporary folder
        #TODO: take car of when tmp_path exists ?
        self.out_ann_path = os.path.join(self.mobile_app_path, "java", "org", "mlperf", "inference", "assets")
        self.out_img_path = os.path.join(self.tmp_path, "img")
        os.makedirs(self.out_img_path)

        self.input_data_path = input_data_path
        if self.input_data_path is None:
            self.download_dataset()



    def download_dataset(self):
        """Downloads dataset from a url to the temp folder self.tmp_path and updates self.path accordingly.
        """
        raise ValueError("input_data_path must not be None")


    def process_dataset(self,
               selected_img_path
               ):
        """Processes dataset from self.input_data_path to mimic the format of self.type.
         - Renames selected images, and push them to the mobile phone sdcard/mlperf_datasets/{self.type}/img
         (existing images of this folder are deleted)
         - Replaces ./mobile_app/java/org/mlperf/inference/assets/{self.type annotation file}.txt with corresponding new one.

        Args:
            selected_img_path: list of paths to images that we want to put in the new dataset
        """
        if self.type == "coco":
            out_ann_name = os.path.join(self.out_ann_path, "coco_val.pbtxt")
            raise NotImplementedError
        elif self.type == "imagenet":
            new_ann_path = os.path.join(self.out_ann_path, "imagenet_val.txt")
            with open(new_ann_path,'w') as new_ann_file:
                for i, img_path in enumerate(selected_img_path):
                    #self.convert_img(img_path)
                    label = self.find_label(img_path)
                    new_img_name = f"ILSVRC2012_val_{i+1:08}.JPEG"
                    #print("\n ***", i, img_path)
                    #print(f"Copies {img_path} to \n {os.path.join(self.out_img_path, new_img_name)}")
                    shutil.copyfile(img_path,
                                    os.path.join(self.out_img_path, new_img_name))
                    new_ann_file.write(str(label) + "\n")

        # Push to mobile
        mobile_dataset_path = os.path.join('sdcard','mlperf_datasets', self.type)

        # Remove existing imagenet images from the phone
        rm_type = subprocess.run(["adb", "shell", "rm","-r", mobile_dataset_path], stderr=subprocess.DEVNULL)
        if rm_type.returncode == 0:
            print(f"Removed {mobile_dataset_path} folder")

        # Push images to the right folder in the phone
        create_dir = subprocess.run(["adb", "shell", "mkdir", "-p", os.path.join(mobile_dataset_path,"img")])
        push = subprocess.run(["adb", "push", self.out_img_path, mobile_dataset_path])

        assert create_dir.returncode == push.returncode == 0
        clean = subprocess.run(["rm", "-r", self.tmp_path])

    def find_label(self, img_path):
        """ Given an image path, returns the ground-truth class.
        Returns:
            label: index of corresponding class in type dataset
        """
        raise NotImplementedError

    def subsample(self,):
        """Policy for subsampling.
        Returns:
            selected_img_path: list of paths to images that we want to put in the output dataset
        """
        raise NotImplementedError



class ADE20KDataset(InputDataset):
    def __init__(self, input_data_path, mobile_app_path, type):
        super().__init__(input_data_path=input_data_path, mobile_app_path=mobile_app_path, type=type)
        self.train_path = os.path.join(self.input_data_path, "images", "training")
        self.val_path = os.path.join(self.input_data_path, "images", "validation")
        self.in_annotations = {}

    def download_dataset(self):
        print("Downloading dataset to temporary folder...")
        #ADE20K_url = "https://groups.csail.mit.edu/vision/datasets/ADE20K/ADE20K_2016_07_26.zip"
        ADE20K_subset_url =  'https://github.com/ctrnh/mlperf_misc/raw/master/ADE20K_subset.zip'
        zip_path, hdrs = urllib.request.urlretrieve(ADE20K_subset_url, f"{self.tmp_path}/ADE20K_subset.zip")
        print("Extracting dataset to temporary folder...")
        with zipfile.ZipFile(f"{zip_path}", 'r') as z:
            z.extractall(f"{self.tmp_path}")
        self.input_data_path = zip_path[:-4]

    def subsample(self, N, policy="random"):
        """Subsamples from ADE20K: it considers all images which class intersects with imagenet classes.
        Args:
            N: number of wanted samples.
            policy: type of policy :"random", "balanced" (TODO)

        Returns:
            selected_img_path: list of path to images we want to keep in the new dataset
        """
        selected_img_path = []
        for set_path in [self.train_path, self.val_path]:
            for letter in os.listdir(set_path):
                letter_path = os.path.join(set_path, letter)
                ADE_letter_classes = os.listdir(letter_path)
                for cur_class in ADE_letter_classes:
                   spaced_class = " ".join(cur_class.split("_"))
                   if spaced_class in IMAGENET_CLASSES:
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
            label = IMAGENET_CLASSES[self.in_annotations[img_path]]
        return label


class KanterDataset(InputDataset):
    def __init__(self, input_data_path, mobile_app_path, type):
        super().__init__(input_data_path=input_data_path, mobile_app_path=mobile_app_path, type=type)
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
            label = IMAGENET_CLASSES[self.in_annotations[img_name]]
        return label


def main():

    input_data_path = args.input_data_path

    if args.dataset.lower() == "kanter":
        import json
        input_dataset = KanterDataset(input_data_path=input_data_path,
                                mobile_app_path=args.mobile_app_path,
                                type=args.type)
    elif args.dataset.lower() == "ade20k":
        input_dataset = ADE20KDataset(input_data_path=input_data_path,
                                mobile_app_path=args.mobile_app_path,
                                type=args.type)
    else:
        raise ValueError

    selected_img_path = input_dataset.subsample(N=args.N)
    input_dataset.process_dataset(selected_img_path=selected_img_path)



if args.type == "imagenet":
    print("Loading imagenet classes...")
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    imagenet_simple_labels = eval(requests.get(url).text)
    # TODO: Check that there is no error in those classes
    IMAGENET_CLASSES = dict(zip(imagenet_simple_labels, [i for i in range(1,len(imagenet_simple_labels)+1)]))

main()
