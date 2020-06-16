import argparse
import os
import random
import shutil
import json

"""
First simple script which converts Kanter dataset. (Draft of a possible structure for the code)
Example output folder after running the script with the following cmd:
python simple_script_custom_dataset.py --in_path=kanter_dataset --out_path=out_kanter --N=3 --type="imagenet"


Remarks:
 - in_path: path to dataset folder which contains two subfolders, img and annotations
 annotations here contains labels.json
"""

parser = argparse.ArgumentParser()
parser.add_argument("--in_path", help="directory of input dataset (img + annotations). If None, script downloads from ... ?", default=None)
parser.add_argument("--out_path", help="output directory to dump out the samples into")
parser.add_argument("--N", type=int, help="number of samples wanted in the dataset")
parser.add_argument("--type", help="coco or imagenet")
args = parser.parse_args()

imagenet_simple_labels = json.load(open("./imagenet-simple-labels.json",'r')) # TODO:Code better
imagenet_classes = dict(zip(imagenet_simple_labels, [i for i in range(1,len(imagenet_simple_labels)+1)]))

def subsample(in_path, N, policy="random"):
    """Policy for selecting images.

    Args:
        in_img_path
    Returns:
        list of selected image paths
    """
    in_img_path = os.path.join(in_path, "img")
    if policy == "random":
        selected_img = random.sample(os.listdir(in_img_path), N)
    return selected_img#list(map(lambda x:os.path.join(in_img_path, x), selected_img))


def format(in_path,
           selected_img,
           out_img_path,
           out_ann_path,
           type):
    """Copies and formats images + annotations from in_path to out_path according to the type.
    Tasks: change folder structure, renames img,

    Args:
        type: "coco" or "imagenet"
    """
    in_img_path = os.path.join(in_path, "img")
    in_ann_file = json.load(open(os.path.join(in_path, "annotations", "labels.json"), 'r'))
    if type == "coco":
        out_ann_name = os.path.join(out_ann_path, "coco_val.pbtxt")
    elif type == "imagenet":
        out_ann_file = open(os.path.join(out_ann_path, "imagenet_val.txt"),'w')
        for i, img in enumerate(selected_img):
            label = imagenet_classes[in_ann_file[img]]
            new_img_name = f"IMG_{i:012}.jpg"
            # TODO: if not in right format (jpg, png, convert)
            shutil.copyfile(os.path.join(in_img_path, img),
                            os.path.join(out_img_path, new_img_name))
            out_ann_file.write(str(label) + "\n")
        out_ann_file.close()

    else:
        raise ValueError


def format_img_imagenet(img_path,
                        label,
                        new_idx):
    """Fomats a single image into 
    """

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
    out_img_path, out_ann_path = dir_structure(args.out_path, type=args.type)

    selected_img = subsample(in_path, args.N)
    format(in_path=in_path,
           selected_img=selected_img,
           out_img_path=out_img_path,
           out_ann_path=out_ann_path,
           type=args.type)



main()
