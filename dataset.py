"""
    Dataset Segmentation Dataloader
    Options:PascalVOC Dataset
            Score Dataset

    To run: python dataset.py
"""

from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image

SCORE_CLASSES = ('-10', '-9', '-8', '-7','-6', '-5',
                 '-4', '-3', '-2', '-1', '0', '1', '2',
                 '3', '4', '5', '6')

NUM_SCORES  = len(SCORE_CLASSES)

# parser = argparse.ArgumentParser(description='Train a SegNet model')

# parser.add_argument('--VOC',default=0, type= int,help='whether to load the VOC dataset or not')

# args = parser.parse_args()


# VOC = args.VOC                 # whether to load VOC or Score
VOC=1
class ScoreDataset(Dataset):
    """The score dataset created by us"""
    def __init__(self, list_file, img_dir, mask_dir):
        self.images = open(list_file,"rt").read().split("\n")[:-1]
       
        self.image_root_dir = img_dir
        self.mask_root_dir  = mask_dir

        # self.counts = self.__compute_class_probability()
        self.counts =    {0: 100946816, 1:1, 2:1, 3:1, 4: 213530, 5: 2476976, 6: 32769, 7: 193461,
                         8: 359074, 9: 71519, 10: 46142, 11: 31482, 12: 15810, 13: 4947, 14: 3202,
                          15: 11343, 16: 129}
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        name = self.images[index]
        image_path = os.path.join(self.image_root_dir, name)
        mask_path = os.path.join(self.mask_root_dir, name)

        image = self.load_image(path=image_path)
        gt_mask = self.load_mask(path=mask_path)

        data = {
                    'image': torch.FloatTensor(image),
                    'mask' : torch.LongTensor(gt_mask)
                    }
        

        return data

    def __compute_class_probability(self):
        counts = dict((i, 0) for i in range(NUM_SCORES))
        print('COMPUTE PROBABS')
        for name in self.images:
            mask_path = os.path.join(self.mask_root_dir, name)

            imx_t = np.load(mask_path) + 10

            for i in range(NUM_SCORES):
                counts[i] += np.sum(imx_t == i)
        print("DONE")
        return counts

    def get_class_probability(self):
        values = np.array(list(self.counts.values()))
        p_values = values/np.sum(values)
        return torch.Tensor(p_values)

    def load_image(self, path=None):
        # raw_image = Image.open(path)
        raw_image = np.load(path)
        raw_image = np.transpose(raw_image, (2,0,1))
        imx_t = np.array(raw_image, dtype=np.float32)/45.0
        return imx_t

    def load_mask(self, path=None):
        # raw_image = Image.open(path)
        raw_image = np.load(path)
        # raw_image = raw_image
        imx_t = np.array(raw_image)
        return imx_t


##############################################################################################################
VOC_CLASSES = ('background',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

NUM_CLASSES = len(VOC_CLASSES) + 1


class PascalVOCDataset(Dataset):
    """Pascal VOC 2007 Dataset"""
    def __init__(self, list_file, img_dir, mask_dir, transform=None):
        self.images = open(list_file, "rt").read().split("\n")[:-1]
        self.transform = transform

        self.img_extension = ".jpg"
        self.mask_extension = ".png"

        self.image_root_dir = img_dir
        self.mask_root_dir = mask_dir

        self.counts = self.__compute_class_probability()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        name = self.images[index]
        image_path = os.path.join(self.image_root_dir, name + self.img_extension)
        mask_path = os.path.join(self.mask_root_dir, name + self.mask_extension)

        image = self.load_image(path=image_path)
        gt_mask = self.load_mask(path=mask_path)
        data = {
                    'image': torch.FloatTensor(image),
                    'mask' : torch.LongTensor(gt_mask)
                    }
        return data

    def __compute_class_probability(self):
        counts = dict((i, 0) for i in range(NUM_CLASSES))

        for name in self.images:
            mask_path = os.path.join(self.mask_root_dir, name + self.mask_extension)

            raw_image = Image.open(mask_path).resize((224, 224))
            imx_t = np.array(raw_image).reshape(224*224)
            imx_t[imx_t==255] = len(VOC_CLASSES)

            for i in range(NUM_CLASSES):
                counts[i] += np.sum(imx_t == i)
        return counts

    def get_class_probability(self):
        values = np.array(list(self.counts.values()))
        p_values = values/np.sum(values)
        return torch.Tensor(p_values)

    def load_image(self, path=None):
        raw_image = Image.open(path)
        raw_image = np.transpose(raw_image.resize((224, 224)), (2,1,0))
        imx_t = np.array(raw_image, dtype=np.float32)/255.0

        return imx_t

    def load_mask(self, path=None):
        raw_image = Image.open(path)
        raw_image = raw_image.resize((224, 224))
        imx_t = np.array(raw_image)
        imx_t[imx_t==255] = len(VOC_CLASSES)
        return imx_t


if __name__ == "__main__":
    if VOC:
        data_root = os.path.join("VOCdevkit", "VOC2007")
        list_file_path = os.path.join(data_root, "ImageSets", "Segmentation", "train.txt")
        img_dir = os.path.join(data_root, "JPEGImages")
        mask_dir = os.path.join(data_root, "SegmentationClass")


        objects_dataset = PascalVOCDataset(list_file=list_file_path,
                                        img_dir=img_dir,
                                        mask_dir=mask_dir)

    else:
        print('SCORE DATASET')
        data_root = "Data"
        list_file_path = os.path.join(data_root,"train.txt")
        img_dir = os.path.join(data_root,"Train")
        mask_dir = os.path.join(data_root,"Train_annot")

        objects_dataset = ScoreDataset(list_file=list_file_path,
                                        img_dir=img_dir,
                                        mask_dir=mask_dir)
    for i in range(10):
        k = np.random.randint(0,objects_dataset.__len__(),1)[0]
        print(k)
        sample = objects_dataset[k]
        image, mask = sample['image'], sample['mask']
        image.transpose_(0, 2)

        fig = plt.figure()

        a = fig.add_subplot(1,2,1)
        plt.imshow(image)

        a = fig.add_subplot(1,2,2)
        plt.imshow(mask)

        plt.show()

