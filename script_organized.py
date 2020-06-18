import argparse
import os
import random
import shutil
import json #TODO: avoid import json?

"""
out_ADE20K_subset contains the output created by this script when run with the following command:
python script_organized.py --in_path=ADE20K_subset --out_path=out_ADE20K_subset --N=10 --dataset=ADE20K --type=imagenet
python script_organized.py --in_path=kanter_dataset --out_path=out_kanter --N=8 --dataset=kanter --type=imagenet

If the user wants to use an input dataset which is not ADE20K or kanter, they will need to implement the corresponding methods
"""

parser = argparse.ArgumentParser()
parser.add_argument("--in_path", help="directory of input dataset (img + annotations). If None, script downloads from ... ?", default=None)
parser.add_argument("--out_path", help="output directory to dump out the samples into")
parser.add_argument("--N", type=int, help="number of samples wanted in the dataset")
parser.add_argument("--type", help="coco or imagenet")
parser.add_argument("--dataset", help="Kanter or ADE20K or other to implement")
args = parser.parse_args()



imagenet_simple_labels = json.load(open("./imagenet-simple-labels.json",'r'))
# TODO: Check that there is no error in those classes (taken from a github repo)
IMAGENET_CLASSES = dict(zip(imagenet_simple_labels, [i for i in range(1,len(imagenet_simple_labels)+1)]))








class InputDataset:
    def __init__(self, path, out_path, type):
        assert type in ["coco", "imagenet"], "type must be coco or imagenet"
        self.type = type
        self.path = path
        self.out_path = out_path
        self.out_img_path = None
        self.out_ann_path = None

        self.dir_structure()

    def dir_structure(self):
        """Replicates type dataset directory structure.
        TODO: check if the output directory structure is correct

        Args:
            out_path: root folder from where one wants to replicate the directories structure
            type: "imagenet" or "coco"

        Defines:
            self.out_img_path: path to folder which will contain the new images
            self.out_ann_path: path to folder which will contain the new annotationssss
        """
        self.out_img_path = os.path.join(self.out_path, "sdcard", "mlperf_datasets", self.type)
        if not os.path.isdir(self.out_img_path):
            os.makedirs(self.out_img_path)

        # I put the following beause it seems like in mobile_app the annotations are saved in
        # the assets folder, as a .txt file. Maybe it isn't the same in the original imagenet
        # If not: what should this script replicate: imagenet or mobile_app?
        self.out_ann_path = os.path.join(self.out_path, "java", "org", "mlperf", "inference", "assets")

        if not os.path.isdir(self.out_ann_path):
            os.makedirs(self.out_ann_path)

    def convert_img(self, img_path):
        """Check if the img is in jpg, if not, converts (in place).
        This doesn't apply to ADE20K/Kanter (which img are jpg), but might be nice to have.
        """
        pass

    def format(self,
               selected_img_path,
               ):
        """Copies and formats images + annotations from in_path to out_path to mimick the format of type dataset.
        The tasks performed so far are:
         - rename images
         - create annotations file

        Args:
            selected_img_path: list of paths to images that we want to put in the output dataset
        """
        if self.type == "coco":
            out_ann_name = os.path.join(self.out_ann_path, "coco_val.pbtxt")
            raise NotImplementedError
        elif self.type == "imagenet":
            with open(os.path.join(self.out_ann_path, "imagenet_val.txt"),'w') as out_ann_file:
                for i, img_path in enumerate(selected_img_path):
                    self.convert_img(img_path)
                    label = self.find_label(img_path)
                    new_img_name = f"IMG_{i:012}.jpg"
                    #print("\n ***", i, img_path)
                    #print(f"Copies {img_path} to \n {os.path.join(self.out_img_path, new_img_name)}")
                    shutil.copyfile(img_path,
                                    os.path.join(self.out_img_path, new_img_name))
                    out_ann_file.write(str(label) + "\n")

    def find_label(self, img_path):
        """
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
    def __init__(self, path, out_path, type):
        super().__init__(path=path, out_path=out_path, type=type)
        self.train_path = os.path.join(path, "images", "training")
        self.val_path = os.path.join(path, "images", "validation")
        self.in_annotations = {}

    def subsample(self, N, policy="random"):
        """Subsamples from ADE20K: it considers all images which class intersect with imagenet classes.
        Args:
            N: number of wanted samples.
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
    def __init__(self, path, out_path, type):
        super().__init__(path=path, out_path=out_path, type=type)
        self.in_img_path = os.path.join(path, "img")
        self.in_annotations = json.load(open(os.path.join(self.path, "annotations", "labels.json"), 'r'))

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
    # TODO: check if in_path exists, if not, download from a link + change in_path
    in_path = args.in_path


    if args.dataset.lower() == "kanter":
        dataset = KanterDataset(path=in_path, out_path=args.out_path, type=args.type)
    elif args.dataset.lower() == "ade20k":
        dataset = ADE20KDataset(path=in_path, out_path=args.out_path, type=args.type)
    else:
        raise ValueError

    selected_img_path = dataset.subsample(N=args.N)
    dataset.format(selected_img_path=selected_img_path)



main()
