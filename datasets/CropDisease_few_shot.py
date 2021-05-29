# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

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

identity = lambda x:x
class SimpleDataset:
    def __init__(self, transform, target_transform=identity):
        self.transform = transform
        self.target_transform = target_transform

        self.meta = {}

        self.meta['image_names'] = []
        self.meta['image_labels'] = []


        d = ImageFolder(CropDisease_path + "/dataset/train/")

        for i, (data, label) in enumerate(d):
            self.meta['image_names'].append(data)
            self.meta['image_labels'].append(label)  

    def __getitem__(self, i):

        img = self.transform(self.meta['image_names'][i])
        target = self.target_transform(self.meta['image_labels'][i])

        return img, target

    def __len__(self):
        return len(self.meta['image_names'])


class SetDataset:
    def __init__(self, batch_size, transform):

        self.sub_meta = {}
        self.cl_list = range(38)

        for cl in self.cl_list:
            self.sub_meta[cl] = []

        d = ImageFolder(CropDisease_path + "/dataset/train/")


        for i, (data, label) in enumerate(d):
            data = data.resize((256, 256))
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

class SetDataset_fix:
    def __init__(self, batch_size, transform, n_episodes=600, n_support=5, n_way=5):

        self.sub_meta = {}
        self.cl_list = range(38)

        for cl in self.cl_list:
            self.sub_meta[cl] = []

        d = ImageFolder(CropDisease_path + "/dataset/train/")

        for i, (data, label) in enumerate(d):
            self.sub_meta[label].append(data)

        self.num_cats = len(self.cl_list)
        self.n_episodes=n_episodes
        self.n_way = n_way
        self.n_support = n_support

        for i, (data, label) in enumerate(d):
            data = data.resize((256, 256))
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
            self.sub_dataloader.append(sub_dataset)
            # self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )
        
        self.episode_list = []
        import random
        random.seed(2020)
        for i in range(n_episodes):
            cats_list = random.sample([sample_i for sample_i in range(self.num_cats)], n_way)
            episode_set = []
            for j in cats_list:
                sample_list = random.sample([sample_i for sample_i in range(len(self.sub_dataloader[j]))], batch_size)
                episode_set.append(sample_list)    #[n_way,batch_size]
            episode = {
                        "cats_list": cats_list,
                        "episode_set": episode_set
                    }
            self.episode_list.append(episode)

    def __getitem__(self, i):
        # return next(iter(self.sub_dataloader[i]))
        episode_img = []     # [n_way,support+query,img]
        episode_label = []
        for cat_num, img_num_list in zip(self.episode_list[i]["cats_list"],self.episode_list[i]["episode_set"]):
            img_list = []
            label_list = []
            for j in img_num_list:
                img, label =self.sub_dataloader[cat_num][j]
                img_list.append(img)
                label_list.append(torch.tensor(label))
            img_list = torch.stack(img_list,0)
            label_list = torch.stack(label_list,0)
            episode_img.append(img_list)
            episode_label.append(label_list)
            # single_cat = torch.cat([self.sub_dataloader[cat_num][j] for j in img_list],0)  # one list contains one category of both support and query [batch_size,img]
        return torch.stack(episode_img, 0), torch.stack(episode_label, 0)
    
    def get_support(self,i):
        img_list = []
        label_list = []
        for cat_num, img_num_list in zip(self.episode_list[i]["cats_list"],self.episode_list[i]["episode_set"]):
            support_list = img_num_list[:self.n_support]
            for j in support_list:
                img, label =self.sub_dataloader[cat_num][j]
                img_list.append(img)
                label_list.append(torch.tensor(label))
        return torch.stack(img_list, 0), torch.stack(label_list, 0)

    def get_query(self,i):
        # episode_img = []     # [n_way,support+query,img]
        # episode_label = []
        img_list = []
        label_list = []
        for cat_num, img_num_list in zip(self.episode_list[i]["cats_list"],self.episode_list[i]["episode_set"]):
            query_list = img_num_list[self.n_support:]
            for j in query_list:
                img, label =self.sub_dataloader[cat_num][j]
                img_list.append(img)
                label_list.append(torch.tensor(label))
        return torch.stack(img_list, 0), torch.stack(label_list, 0)

    def __len__(self):
        return self.n_episodes


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
        if transform_type=='RandomSizedCrop':
            return method(self.image_size) 
        elif transform_type=='CenterCrop':
            return method(self.image_size) 
        elif transform_type=='Resize':
            return method([int(self.image_size*1.15), int(self.image_size*1.15)])
        elif transform_type=='Normalize':
            return method(**self.normalize_param )
        else:
            return method()

    def get_composed_transform(self, aug = False):
        if aug:
            transform_list = ['RandomSizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Resize','CenterCrop', 'ToTensor', 'Normalize']

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
        dataset = SimpleDataset(transform)

        data_loader_params = dict(batch_size = self.batch_size, shuffle = True, num_workers = 8, pin_memory = True)       
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader

class SetDataManager(DataManager):
    def __init__(self, image_size, n_way=5, n_support=5, n_query=16, n_eposide = 100):        
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_support = n_support
        self.n_eposide = n_eposide

        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, aug): #parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)

        mean_pix = [0.485, 0.456, 0.406]
        std_pix = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)
        ToTensor = transforms.ToTensor()
        resize = transforms.Resize([int(self.image_size*1.5), int(self.image_size*1.5)])
        resize1 = transforms.Resize([self.image_size, self.image_size])
        HorizontalFlip = transforms.RandomHorizontalFlip(p=-1)
        transform = transforms.Compose([
            transforms.Lambda(lambda image: (resize(image),(resize1(image)))),
            transforms.Lambda(lambda image: (transforms.TenCrop(self.image_size)(image[0]), image[1], HorizontalFlip(image[1]))),
            transforms.Lambda(lambda crops: torch.stack([ToTensor(crop) for crop in crops[0]]+[ToTensor(crops[2]),ToTensor(crops[1])])), # returns a 4D tensor after tencrop
            transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
        ])
        
        dataset = SetDataset_fix(self.batch_size, transform, n_episodes=self.n_eposide, n_support=self.n_support, n_way=self.n_way)
        # sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide )  
        # data_loader_params = dict(batch_sampler = sampler,  num_workers = 4, pin_memory = True)       
        # data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        # return data_loader

        data_loader_params = dict(batch_size = 1, shuffle = False, num_workers = 2, pin_memory = True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader

if __name__ == '__main__':

    train_few_shot_params   = dict(n_way = 5, n_support = 5) 
    base_datamgr            = SetDataManager(224, n_query = 16)
    base_loader             = base_datamgr.get_data_loader(aug = True)

    cnt = 1
    for i, (x, label) in enumerate(base_loader):
        if i < cnt:
            print(label)
        else:
            break