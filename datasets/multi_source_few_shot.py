# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import json
import os
import torch
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import datasets.additional_transforms as add_transforms
from torch.utils.data import Dataset, DataLoader
from abc import abstractmethod
from torchvision.datasets import ImageFolder

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import sys
sys.path.append("../")
from configs import *

from datasets import miniImageNet_few_shot, DTD_few_shot, caltech256_few_shot, cifar_few_shot
from data.datamgr import SimpleDataManager, SetDataManager
from torchvision.datasets import CIFAR100

import configs

identity = lambda x:x

class multiDataset:
    def __init__(self, transform, target_transform=identity):
        self.transform = transform
        self.target_transform = target_transform
        
        ## miniImageNet ------------------------
        d = ImageFolder(miniImageNet_path)
        self.meta = {}

        self.meta['image_names'] = []
        self.meta['image_labels'] = []
        self.meta['domain_labels'] = []

        self.num_classes_list = []

        label_dict = {}
        label_index = 0
        for i, (data, label) in enumerate(d):
            if label not in label_dict.keys():
                label_dict[label] = label_index
                label_index += 1 
            data = data.resize((168, 168))
            self.meta['image_names'].append(data)
            self.meta['image_labels'].append(label_dict[label])
            self.meta['domain_labels'].append(0)
        print(label_dict)
        label_dict = {}
        self.num_classes_list.append(label_index)
        label_index = 0
        ## CIFAR100 -----------------------------
        d = CIFAR100("./", train=True, download=True)
        for i, (data, label) in enumerate(d):
            if label % 3 == 0:
                if label not in label_dict.keys():
                    label_dict[label] = label_index
                    label_index += 1
                self.meta['image_names'].append(data)
                self.meta['image_labels'].append(label_dict[label])
                self.meta['domain_labels'].append(1)
        print(label_dict)
        label_dict = {}
        self.num_classes_list.append(label_index)
        label_index = 0
        ## DTD ----------------------------------
        d = ImageFolder(DTD_path)
        for i, (data, label) in enumerate(d):
            if label not in label_dict.keys():
                label_dict[label] = label_index
                label_index += 1
            data = data.resize((168, 168))
            self.meta['image_names'].append(data)
            self.meta['image_labels'].append(label_dict[label])
            self.meta['domain_labels'].append(2)
        print(label_dict)
        label_dict = {}
        self.num_classes_list.append(label_index)
        label_index = 0
        ## CUB
        data_file = configs.data_dir['CUB'] + 'base.json'
        with open(data_file, 'r') as f:
            meta = json.load(f)
            for i in range(len(meta['image_names'])):
                label = meta['image_labels'][i]
                if label not in label_dict.keys():
                    label_dict[label] = label_index
                    label_index += 1
                image_path = os.path.join(meta['image_names'][i])
                img = Image.open(image_path).convert('RGB')
                img = img.resize((168, 168))
                self.meta['image_names'].append(img)
                self.meta['image_labels'].append(label_dict[label])
                self.meta['domain_labels'].append(3)
        print(label_dict)
        label_dict = {}
        self.num_classes_list.append(label_index)
        label_index = 0
        ## Caltech256
        d = caltech256_few_shot.Caltech256(root='../dataset/', download=False)
        for i, (data, label) in enumerate(d):
            if label not in label_dict.keys():
                label_dict[label] = label_index
                label_index += 1
            data = data.resize((168, 168))
            self.meta['image_names'].append(data)
            self.meta['image_labels'].append(label_dict[label])
            self.meta['domain_labels'].append(4)
        print(label_dict)
        self.num_classes_list.append(label_index)
        label_index = 0
        print("class number in datasets is", self.num_classes_list)

    def __getitem__(self, i):
        # image, label = self.d[i]

        img = self.transform(self.meta['image_names'][i])
        target = self.target_transform(self.meta['image_labels'][i])
        domain_label = self.target_transform(self.meta['domain_labels'][i])

        return img, target, domain_label

    def __len__(self):
        return len(self.meta['image_names'])

class SetDataset:
    def __init__(self, batch_size, transform):

        self.sub_meta = {}
        self.cl_list = range(64)

        for cl in self.cl_list:
            self.sub_meta[cl] = []

        d = ImageFolder(miniImageNet_path)

        for i, (data, label) in enumerate(d):
            self.sub_meta[label].append(data)

        for key, item in self.sub_meta.items():
            print (len(self.sub_meta[key]))
    
        self.sub_dataloader = [] 
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)        
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform = transform )
            self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.sub_dataloader)

class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self,i):

        img = self.transform(self.sub_meta[i])
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]

class TransformLoader:
    def __init__(self, image_size, 
                 normalize_param    = dict(mean= [0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
                 jitter_param       = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param
    
    def parse_transform(self, transform_type):
        if transform_type=='ImageJitter':
            method = add_transforms.ImageJitter( self.jitter_param )
            return method
        method = getattr(transforms, transform_type)
        if transform_type=='RandomResizedCrop':
            return method(self.image_size) 
        elif transform_type=='CenterCrop':
            return method(self.image_size) 
        elif transform_type=='Scale':
            return method([int(self.image_size*1.15), int(self.image_size*1.15)])
        elif transform_type=='Normalize':
            return method(**self.normalize_param )
        else:
            return method()

    def get_composed_transform(self, aug = False):
        if aug:
            transform_list = ['RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Scale','CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [ self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform

class DataManager(object):
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass 

class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size):        
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = multiDataset(transform)

        data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 8, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader

class SetDataManager(DataManager):
    def __init__(self, image_size, n_way=5, n_support=5, n_query=16, n_eposide = 100):        
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide

        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SetDataset(self.batch_size, transform)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide )  
        data_loader_params = dict(batch_sampler = sampler,  num_workers = 4, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader

if __name__ == '__main__':
    pass
