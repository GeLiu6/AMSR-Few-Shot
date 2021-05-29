import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob

import logging

import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.protonet import ProtoNet
from methods.multi_source_train import Multi_Domain_Train

from io_utils import model_dict, parse_args, get_resume_file
from utils import set_logging_config  
from datasets import miniImageNet_few_shot, DTD_few_shot, multi_source_few_shot


def train(base_loader, model, optimization, start_epoch, stop_epoch, params, log):    
    # if optimization == 'Adam':
    #     optimizer = torch.optim.Adam(model.parameters())
    # else:
    #    raise ValueError('Unknown optimization, please define by yourself')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=1e-4)
    # optimizer = torch.optim.Adam(model.parameters())

    # scheduler = lr_scheduler.StepLR(optimizer, step_size = 100, gamma=0.1)
    lambda_epoch = lambda e: 1.0 if e < 100 else (0.1 if e<150 else (0.01 if e < 200 else 0.001))
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=-1)     

    for epoch in range(start_epoch,stop_epoch):
        model.train()
        model.train_loop(epoch, base_loader,  optimizer, log) 

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict(),'optimizer':optimizer.state_dict()}, outfile)

        scheduler.step()
        
    return model

if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')

    image_size = 84
    optimization = 'Adam'

    if params.method in ['baseline', 'multi_domain'] :

        if params.dataset == "miniImageNet":
        
            datamgr = miniImageNet_few_shot.SimpleDataManager(image_size, batch_size = 64)
            base_loader = datamgr.get_data_loader(aug = params.train_aug )

        elif params.dataset == "CUB":

            base_file = configs.data_dir['CUB'] + 'base.json' 
            base_datamgr    = SimpleDataManager(image_size, batch_size = 16)
            base_loader     = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
       
        elif params.dataset == "cifar100":
            base_datamgr    = cifar_few_shot.SimpleDataManager("CIFAR100", image_size, batch_size = 16)
            base_loader    = base_datamgr.get_data_loader( "base" , aug = True )
                
            params.num_classes = 100

        elif params.dataset == 'caltech256':
            base_datamgr  = caltech256_few_shot.SimpleDataManager(image_size, batch_size = 16)
            base_loader = base_datamgr.get_data_loader(aug = False )
            params.num_classes = 257

        elif params.dataset == "DTD":
            base_datamgr    = DTD_few_shot.SimpleDataManager(image_size, batch_size = 16)
            base_loader     = base_datamgr.get_data_loader( aug = True )
        
        elif params.dataset == "multi_source":
            datamgr = multi_source_few_shot.SimpleDataManager(image_size, batch_size = 128)
            base_loader = datamgr.get_data_loader(aug = params.train_aug )
            num_classes_list = base_loader.dataset.num_classes_list
        else:
           raise ValueError('Unknown dataset')

        # model           = BaselineTrain( model_dict[params.model], params.num_classes)
        if params.method == 'multi_domain':
            model = Multi_Domain_Train(model_dict[params.model], num_classes_list)
    elif params.method in ['protonet']:
        n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
        train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot) 
        test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot) 

        if params.dataset == "miniImageNet":

            datamgr            = miniImageNet_few_shot.SetDataManager(image_size, n_query = n_query,  **train_few_shot_params)
            base_loader        = datamgr.get_data_loader(aug = params.train_aug)

        else:
           raise ValueError('Unknown dataset')

        if params.method == 'protonet':
            model           = ProtoNet( model_dict[params.model], **train_few_shot_params )
       
    else:
       raise ValueError('Unknown method')

    model = model.cuda()
    save_dir =  configs.save_dir

    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(save_dir, params.dataset, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'

    if not params.method  in ['baseline', 'baseline++', 'multi_domain']: 
        params.checkpoint_dir += '_%dway_%dshot' %( params.train_n_way, params.n_shot)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)
    
    current_time = time.strftime('%y-%m-%d-%H:%M:%S',time.localtime(time.time()))
    log_file_path = os.path.join(params.checkpoint_dir, "log_%s.txt"%(current_time))
    set_logging_config(log_file_path)
    log = logging.getLogger('train')
    log.info(' '.join(os.sys.argv))
    log.info(params)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    # resume_file = '%s/%d%s' %(params.checkpoint_dir, 99, '.tar')
    # tmp = torch.load(resume_file)
    # start_epoch = tmp['epoch']+1
    # model.load_state_dict(tmp['state'])

    model = train(base_loader, model, optimization, start_epoch, stop_epoch, params, log)