import argparse
import os
import random
import shutil
import json

"""
(not tested yet, probably buggy)
Maybe it would be better organized with one class for each different input dataset we give.
Then if the user wants to use an input dataset which is formatted a certain way, they will need to implement the corresponding methods
"""

parser = argparse.ArgumentParser()
parser.add_argument("--in_path", help="directory of input dataset (img + annotations). If None, script downloads from ... ?", default=None)
parser.add_argument("--out_path", help="output directory to dump out the samples into")
parser.add_argument("--N", type=int, help="number of samples wanted in the dataset")
parser.add_argument("--type", help="coco or imagenet")
parser.add_argument("--dataset", help="Kanter or ADE20K")
args = parser.parse_args()

imagenet_simple_labels = json.load(open("./imagenet-simple-labels.json",'r')) # TODO:Code better
imagenet_classes = dict(zip(imagenet_simple_labels, [i for i in range(1,len(imagenet_simple_labels)+1)]))


class InputDataset:
    def __init__(self, path):
        self.path = path

    def format(self,
               selected_img_path,
               out_img_path,
               out_ann_path,
               type):
        """Copies and formats images + annotations from in_path to out_path according to the type.
        Tasks: change folder structure, renames img,

        Args:
            type: "coco" or "imagenet"
        """
        if type == "coco":
            out_ann_name = os.path.join(out_ann_path, "coco_val.pbtxt")
            raise NotImplementedError
        elif type == "imagenet":
            out_ann_file = open(os.path.join(out_ann_path, "imagenet_val.txt"),'w')
            for i, img_path in enumerate(selected_img_path):
                label = find_label(img_path)
                new_img_name = f"IMG_{i:012}.jpg"
                # TODO: if not in right format (jpg, png, convert)
                shutil.copyfile(img_path,
                                os.path.join(out_img_path, new_img_name))
                out_ann_file.write(str(label) + "\n")
            out_ann_file.close()
        else:
            raise ValueError

    def find_label(self, img_path):
        raise NotImplementedError


class ADE20KDataset(InputDataset):
    def __init__(self, path):
        super().__init__(path)
        self.train_path = os.path.join(path,"training")
        self.val_path = os.path.join(path,"validation")
        self.annotations = {}

    def subsample(self, N, policy="all"):
        """Subsamples from ADE20K
        policy "all" : subsamples all images which class is in imagenet
        """
        if policy == "all":
            for set_path in [self.train_path, self.val_path]:
                for letter in os.listdir(set_path):
                    letter_path = os.path.join(set_path, letter)
                    ADE_letter_classes = os.listdir(letter_path)
                    for class in ADE_letter_classes:
                        spaced_class = " ".join(class.split("_"))
                        if spaced_class in imagenet_classes:
                            class_path = os.path.join(letter_path, class)
                            for f in os.listdir(class_path):
                                if f.endswith(".jpg"):
                                    img_path = os.path.join(class_path, f)
                                    selected_img_path.append(img_path)
                                    self.annotations[img_path] = spaced_class
        return selected_img_path

    def find_label(self, img_path):
        label = imagenet_classes[self.annotations[img_path]]
        return label


class KanterDataset(InputDataset):
    def __init__(self, path):
        super().__init__(path)
        self.img_path = os.path.join(path, "img")
        self.annotations = json.load(open(os.path.join(in_path, "annotations", "labels.json"), 'r'))

    def subsample(self, N, policy="random"):
        """Policy for selecting images.

        Args:
            in_img_path
        Returns:
            list of selected image paths
        """
        if policy == "random":
            selected_img = random.sample(os.listdir(self.img_path), N)
        return list(map(lambda x:os.path.join(self.img_path, x), selected_img))

    def find_label(self, img_path):
        img_name = img_path.split("/")[-1]
        label = imagenet_classes[self.annotations[img_name]]
        return label






def dir_structure(out_path, type):
    """Replicates datasets directories structure
    Args:
        type: "imagenet" or "coco"
    """
    out_img_path = os.path.join(out_path, "sdcard", "mlperf_datasets", type)
    if not os.path.isdir(out_img_path):
        os.makedirs(out_img_path)

    out_ann_path = os.path.join(out_path, "java", "org", "mlperf", "inference", "assets")


    if not os.path.isdir(out_ann_path):
        os.makedirs(out_ann_path)
    return out_img_path, out_ann_path





def main():
    # TODO: check if in_path exists, if not, download from a link
    in_path = args.in_path


    if args.dataset == "Kanter":
        dataset = KanterDataset(path=in_path)
    elif args.dataset == "ADE20K":
        dataset = ADE20KDataset(path=in_path)


    out_img_path, out_ann_path = dir_structure(out_path=args.out_path, type=args.type)

    selected_img_path = dataset.subsample(N=args.N)
    dataset.format(selected_img_path=selected_img_path,
                    out_img_path=out_img_path,
                    out_ann_path=out_ann_path,
                    type=args.type)



main()
