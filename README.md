Script which converts a given dataset to imagenet/coco-like dataset for speed tests.

# Requirements
- python3
- cv2
- matplotlib, numpy, json (TODO: remove?)

# Description
This script takes as input a dataset (for example ADE20K). After subsampling this dataset,
it formats it so as to mimic "coco" or "imagenet" dataset for the mobile_app.
More specifically, it replaces the existing annotation file from mobile_app with
a new annotation file (corresponding to the new dataset) having the same format.
Then it pushes images of the new dataset to the mobile phone.



# Remarks
- Input dataset must be either ADE20K or kanter

## Remarks for coco
- For coco, images are resized to 300x300 (and bounding boxes are adapted) so as to follow the [mobile_app documentation](https://github.com/mlperf/mobile_app/blob/master/cpp/datasets/README.md). (The value 300x300 is hard coded for the moment.)

# Example

## Commands
List of commands for using ADE20K as classification test dataset in mobile app:
```
git clone https://github.com/mlperf/mobile_app.git
python script.py --mobile_app_path=./mobile_app --N=400 --dataset=ADE20K --type=imagenet --subsampling_strategy=random
export ANDROID_HOME=Path/to/SDK # Ex: $HOME/Android/Sdk
export ANDROID_NDK_HOME=Path/to/NDK # Ex: $ANDROID_HOME/ndk/(your version)
bazel-2.2.0 build -c opt --cxxopt='--std=c++14' \
    --fat_apk_cpu=x86,arm64-v8a,armeabi-v7a \
    //java/org/mlperf/inference:mlperf_app
adb install -r bazel-bin/java/org/mlperf/inference/mlperf_app.apk
```

Those commands are run from the root folder when the directory structure is like this:

```
root/
│   script.py
└───mobile_app/
│   │   ...
│   └───java/org/mlperf/inference/assets/
│       │   imagenet_val.txt
│       │   coco_val.pbtxt
│       │   ...
```

Running those commands and launching the app gave me ~57% accuracy.

## Main steps of the script
The main steps followed by the script, when `python script.py --mobile_app_path=./mobile_app --N=400 --dataset=ADE20K --type=imagenet` is run are:
* Create a temporary folder in ./mobile_app
* Download ADE20K from official [url](https://groups.csail.mit.edu/vision/datasets/ADE20K), and unzip in the tmp folder
* Subsample, format images from ADE20K and save them in the tmp folder
* Push new images to the sdcard of the phone
* Remove the temporary folder
* Update the annotation file ./mobile_app/java/org/mlperf/inference/assets/imagenet_val.txt accordingly to the new images
