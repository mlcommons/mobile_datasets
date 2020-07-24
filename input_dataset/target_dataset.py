from enum import Enum

class SubsamplingPolicy(Enum):
    random = 1
    balanced = 2

class TargetDataset:
    self.name
    self.img_size
    self.out_ann_path
    self.classes_url
    self.ann_url
    self.dataset_classes

    def load_classes(self):
    def format_img_name(self):
    def intersecting_classes(self):
    def read_annotations(self):
    def subsample(self, N, policy = SubsamplingPolicy.random):
    def write_annotations(self):
