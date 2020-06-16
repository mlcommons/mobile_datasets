Script which converts a given dataset to imagenet/coco-like dataset for speed tests.

# Questions/Remarks:
 - The input dataset should contain classes which also belong to imagenet classes
 - Formatting depends a lot on the structure of the input dataset. For example, the code might need to change if we give as input dataset the ADE20K dataset vs an artificial dataset. How to handle this? If we create an artificial dataset, should we define some conventions on the way annotations (labels or bounding boxes) are stored ?
 - Should we mimick imagenet/coco, or the way imagenet/coco are read from mlperf mobile app? (If in mobile app, we have already done some processing of imagenet, mimicking imagenet may not be what we want)
 - Should we subsample first and then only process the subsamples, or process the whole input dataset and then subsample? (The latter may require more computations and memory)

# TODO / TO CHANGE
 - Images folder / annotations folder : they may not be in the same "out_path" folder
 - Extract an imagenet like dataset from ADE20K dataset: should keep only images which contain classes which belong to imagenet classes.
 - Extract bounding boxes from mask images of ADE20K dataset
