import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob
from itertools import combinations

import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.protonet import ProtoNet

from io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file 

from utils import *

from datasets import ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot, Chest_few_shot

from methods.multi_source_train import Multi_Domain_Train, cosine_classifer


def entropy(out_t1, lamda = 0.1):
    out_t1 = F.softmax(out_t1, dim=1)
    loss_ent = -lamda * torch.mean(torch.sum(out_t1 *(torch.log(out_t1 + 1e-8)), 1))
    return loss_ent

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0) 

    total0 = total.unsqueeze(0).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]

    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

    return sum(kernel_val)

def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n = int(source.size()[0])
    m = int(target.size()[0])

    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:n, :n] 
    YY = kernels[n:, n:]
    XY = kernels[:n, n:]
    YX = kernels[n:, :n]

    XX = torch.div(XX, n * n).sum(dim=1).view(1,-1) 
    XY = torch.div(XY, -n * m).sum(dim=1).view(1,-1)

    YX = torch.div(YX, -m * n).sum(dim=1).view(1,-1)
    YY = torch.div(YY, m * m).sum(dim=1).view(1,-1)
        
    loss = (XX + XY).sum() + (YX + YY).sum()
    return loss

def doamin_adapation(log, novel_loader, n_query = 15, pretrained_dataset='miniImageNet', freeze_backbone = False, n_way = 5, n_support = 5):
    num_classes_list = [64, 34, 47, 100, 256]
    model = Multi_Domain_Train(model_dict[params.model], num_classes_list)
    acc_all = []
    acc_all_list = None
    timer = Timer()
    
    checkpoint_dir = '%s/checkpoints/%s/%s_%s_aug' %(configs.save_dir, pretrained_dataset, params.model, params.method)
    best_model_file = checkpoint_dir + '/249.tar'
    load_file = torch.load(best_model_file)

    for episode , (x, y) in enumerate(novel_loader):
        x = x.squeeze(0)
        n_query = x.size(1) - n_support
        x = x.cuda()
    
        batch_size = 25
        query_batch_size = 25
        n_crops = x.size(2)
        support_size = n_way * n_support * n_crops
        query_size = n_way * n_query * n_crops
        

        y_a_i = torch.from_numpy( np.repeat(range( n_way ), n_support*n_crops ) ).cuda() # (25,)
        x_b_i = x[:, n_support:].contiguous().view( n_way* n_query*n_crops,   *x.size()[3:]) 
        x_a_i = x[:, :n_support].contiguous().view( n_way* n_support*n_crops, *x.size()[3:]) # (25, 3, 224, 224)

        acc_list_in_this_task = []
        overall_scores_in_epochs = []
        
        ###############################################################################################
        # fourth model fine-tuning with classification, MMD, Entropy losses
        ###############################################################################################
        model.load_state_dict(load_file['state'])
        model = model.cuda()

        if freeze_backbone is False:
            model.train()
        else:
            model.eval()

        ###############################################################################################
        # create classifier for each domain
        novel_classifiers = nn.ModuleList([cosine_classifer(model.feat_dim , n_way).cuda() for i in model.num_class_list]).cuda()
        
        # classifier initialization, Pseudo labeling and cherry picking.
        with torch.no_grad():
            shared_features, BP_features_list = model.forward_features(x_a_i, batch_size = forward_batch_size)
            shared_features_query, BP_features_list_query = model.forward_features(x_b_i, batch_size = forward_batch_size)
            for i, support_features in enumerate(BP_features_list):
                novel_classifiers[i].weight.data = support_features.view(n_way, n_support*n_crops, -1).mean(1)
            domain_scores = model.domain_classifier(torch.cat((shared_features,shared_features_query), dim =0))
            sample_weights = domain_scores.softmax(dim = -1)
            domain_weights = sample_weights.view(n_way*(n_support+n_query)*n_crops, n_way).mean(0)

            prob_list = []
            for i, query_features in enumerate(BP_features_list_query):
                scores = novel_classifiers[i].forward(query_features)
                prob = scores.softmax(dim = -1)
                prob =  prob.view(n_way*n_query, n_crops, n_way).mean(1)
                prob_list.append(prob)
            
            scores = 0
            for i in range(len(prob_list)):
                scores+= domain_weights[i]*prob_list[i]
            
            topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
            topk_ind = topk_labels.cpu().numpy()

            ## performance of fixed representations.
            y_query = np.repeat(range( n_way ), n_query)
            top1_correct = np.sum(topk_ind[:,0] == y_query)
            acc=float(top1_correct)/len(y_query)*100
            acc_list_in_this_task.append(acc)

            for j in range(n_way):
                # selecting top k samples for the j-th category
                label_index = np.argwhere(topk_ind[:,0] == j)[:,0]
                k = min(int(topk_scores.size(0)/n_way*ratio),label_index.size)  # number of selected query samples for j-th pseudo label
                _, topk_index = topk_scores[label_index].topk(k=k,dim=0) #size of topk_index is [k]

                selected_index =  label_index[topk_index[:,0].cpu()]   # seleted sample index in the original label tensor
                unselected_index = np.setdiff1d(label_index, selected_index)

                if j == 0: rest_query = x_b_i.view(n_way*n_query,n_crops, *x_b_i.size()[1:])[unselected_index]
                else: rest_query = torch.cat( (rest_query, x_b_i.view(n_way*n_query,n_crops, *x_b_i.size()[1:])[unselected_index]), 0 )
                
                selected_query_samples = x_b_i.view(n_way*n_query,n_crops, *x_b_i.size()[1:])[selected_index]
                selected_query_labels = topk_labels[selected_index].repeat(1,n_crops).view(-1)  
            
                # add selected_query_samples to support set
                selected_query_samples = selected_query_samples.view(k*n_crops, *x_a_i.size()[1:])
                x_a_i = torch.cat((x_a_i, selected_query_samples), dim = 0)
                y_a_i = torch.cat((y_a_i, selected_query_labels), dim = 0)
                
            rest_query = rest_query.view(-1 , *rest_query.size()[2:])
            support_size = y_a_i.size(0)
            query_size = rest_query.size(0)

        loss_fn = nn.CrossEntropyLoss().cuda()
        classifier_opt = torch.optim.SGD(novel_classifiers.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)
        if freeze_backbone is False:
            delta_opt = torch.optim.SGD([{'params':model.feature.parameters()}, {'params':model.BP_list.parameters()}],lr = 0.01)
        ###############################################################################################
        
        rand_query_id = np.random.permutation(query_size)
        query_index = 0

        for epoch in range(total_epoch):
            rand_id = np.random.permutation(support_size)
            model.train()
            novel_classifiers.train()
            for j in range(0, support_size, batch_size):
                classifier_opt.zero_grad()
                if freeze_backbone is False:
                    delta_opt.zero_grad()

                #####################################
                selected_id = torch.from_numpy( rand_id[j: min(j+batch_size, support_size)]).cuda()
                z_batch = x_a_i[selected_id]
                y_batch = y_a_i[selected_id]
                
                selected_query_id = rand_query_id[query_index : query_index + query_batch_size]
                query_batch = rest_query[selected_query_id]
                query_index += query_batch_size

                if (query_index + query_batch_size +1) > query_size:    
                    rand_query_id = np.random.permutation(query_size)
                    query_index = 0
                
                #####################################

                _, BP_features_list = model.forward_features(z_batch)
                _, BP_features_list_query = model.forward_features(query_batch)
                loss = 0
                for k in range(len(domain_weights)):
                    novel_scores = novel_classifiers[k](BP_features_list[k])
                    base_scores = model.classifier_list[k](BP_features_list[k])

                    score_cat = torch.cat((base_scores, novel_scores),1)
                    y_joint = (y_batch + model.num_class_list[k]).view(-1, 1)
                    m = 0.1
                    delt_cos = torch.zeros(score_cat.size()).scatter_(1, y_joint.cpu(), m)
                    score_cat = score_cat - delt_cos.cuda()*novel_classifiers[k].scale

                    classification_loss = loss_fn(score_cat, y_batch + model.num_class_list[k])

                    rand_id_class = np.random.permutation(model.classifier_list[k].weight.size(0))
                    selected_class_centers = model.classifier_list[k].weight.data[rand_id_class[:34]]

                    BP_features = torch.cat((BP_features_list_query[k], BP_features_list[k]), dim = 0)

                    novel_scores = novel_classifiers[k](BP_features_list_query[k])
                    entropy_loss = entropy(novel_scores, lamda=0.1)

                    mmd_loss = mmd(selected_class_centers, BP_features, kernel_num=17, fix_sigma=1)
                    loss = loss + domain_weights[k] * (classification_loss + 10* mmd_loss + entropy_loss)

                #####################################
                loss.backward()

                classifier_opt.step()
                
                if freeze_backbone is False:
                    delta_opt.step()

            if epoch == 4:
                with torch.no_grad():
                    shared_features, BP_features_list = model.forward_features(x_a_i, batch_size = forward_batch_size)
                    shared_features_query, BP_features_list_query = model.forward_features(rest_query, batch_size = forward_batch_size)
                    prob_list = []
                    for i, query_features in enumerate(BP_features_list_query):
                        scores = novel_classifiers[i].forward(query_features)
                        prob = scores.softmax(dim = -1)
                        prob =  prob.view(-1, n_crops, n_way).mean(1)
                        prob_list.append(prob)
                    
                    scores = 0
                    for i in range(len(prob_list)):
                        scores+= domain_weights[i]*prob_list[i]
                    topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
                    topk_ind = topk_labels.cpu().numpy()
                    for j in range(n_way):
                        # selecting top k samples for the j-th category
                        label_index = np.argwhere(topk_ind[:,0] == j)[:,0]
                        k = min(int(topk_scores.size(0)/n_way*ratio*2), label_index.size)  # number of selected query samples for j-th pseudo label
                        _, topk_index = topk_scores[label_index].topk(k=k,dim=0) #size of topk_index is [k]

                        selected_index =  label_index[topk_index[:,0].cpu()]   # seleted sample index in the original label tensor
                        unselected_index = np.setdiff1d(label_index, selected_index)

                        if j == 0: rest_query_second = rest_query.view(-1,n_crops, *rest_query.size()[1:])[unselected_index]
                        else: rest_query_second = torch.cat( (rest_query_second, rest_query.view(-1,n_crops, *rest_query.size()[1:])[unselected_index]), 0)
                        
                        selected_query_samples = rest_query.view(-1,n_crops, *rest_query.size()[1:])[selected_index]
                        selected_query_labels = topk_labels[selected_index].repeat(1,n_crops).view(-1)    
                    
                        # add selected_query_samples to support set
                        selected_query_samples = selected_query_samples.view(k*n_crops, *rest_query.size()[1:])
                        x_a_i = torch.cat((x_a_i, selected_query_samples), dim = 0)
                        y_a_i = torch.cat((y_a_i, selected_query_labels), dim = 0)
                    rest_query = rest_query_second.view(-1 , *rest_query_second.size()[2:])
                    support_size = y_a_i.size(0)
                    query_size = rest_query.size(0)
                    rand_query_id = np.random.permutation(query_size)
                    query_index = 0
        
        #######################
        #     evaluate performance on this epoch      
        model.eval()
        novel_classifiers.eval()
        ###############################################################################################
        with torch.no_grad():
            shared_features, BP_features_list = model.forward_features(x_b_i, batch_size = forward_batch_size)
            scores = 0
            scores_list = []
            for i, features in enumerate(BP_features_list):
                scores =  novel_classifiers[i](features)
                scores = scores.view(n_way*n_query,n_crops,n_way).mean(dim=1)
                scores_list.append(scores)

            y_query = np.repeat(range( n_way ), n_query )                  
            scores = 0
            for i in range(len(domain_weights)):
                scores+= domain_weights[i]*scores_list[i]
            overall_scores_in_epochs.append(scores)
            topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
            topk_ind = topk_labels.cpu().numpy()
            top1_correct = np.sum(topk_ind[:,0] == y_query)
            acc=float(top1_correct)/len(y_query)*100
            acc_list_in_this_task.append(acc)
        
        #############################################################
        # results on this episodes
        #############################################################

        log_string = "Acc %d : "%(episode)

        if acc_all_list is None:
            acc_all_list = [ [] for acc in acc_list_in_this_task]

        for j in range(len(acc_list_in_this_task)):
            acc_all_list[j].append(acc_list_in_this_task[j])
            log_string += "%d:%4.2f%%(%4.2f%%),"%(j, np.mean(acc_all_list[j]), acc_list_in_this_task[j])

        log_string += "Time:%s/%s"%(timer.measure(), timer.measure((episode+1)/ iter_num))
        log.info(log_string)
    #####
    # output results with confidence interval
        if episode% 10==9:
            log_string = "Acc %d : "%(episode)
            for j in range(len(acc_all_list)):
                acc_all  = np.asarray(acc_all_list[j])
                acc_mean = np.mean(acc_all)
                acc_std  = np.std(acc_all)
                log_string += "%d:%4.2f%%(%4.2f%%),"%(j, acc_mean, 1.96* acc_std/np.sqrt(iter_num))
            log.info(log_string)    


if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')
    ##################################################################
    if params.model == 'ResNet12' or params.model == 'WResNet12':
        image_size = 84
    elif params.model == 'ResNet10':
        image_size = 224

    iter_num = 100

    n_query = 15
    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot) 
   
    freeze_backbone = params.freeze_backbone
    ##################################################################
    pretrained_dataset =  params.dataset #"miniImageNet" #"multi_source"

    checkpoint_dir = '%s/checkpoints/%s/%s_%s_aug' %(configs.save_dir, pretrained_dataset, params.model, params.method)
    current_time = time.strftime('%y-%m-%d-%H:%M:%S',time.localtime(time.time()))
    log_file_path = os.path.join(checkpoint_dir, "log_%s.txt"%(current_time))
    set_logging_config(log_file_path)
    log = logging.getLogger()
    log.info(' '.join(os.sys.argv))
    log.info(params)

    dataset_names = ["ISIC", "EuroSAT", "CropDisease", "ChestX"]
    novel_loaders = []
    if params.test_dataset == "ISIC":
        log.info("Loading ISIC")
        datamgr             =  ISIC_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
        novel_loader        = datamgr.get_data_loader(aug =False)
        novel_loaders.append(novel_loader)
        ratio = 0.1
    elif params.test_dataset == "EuroSAT":
        print ("Loading EuroSAT")
        datamgr             =  EuroSAT_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
        novel_loader        = datamgr.get_data_loader(aug =False)
        novel_loaders.append(novel_loader)
        ratio = 0.15
    elif params.test_dataset == "CropDisease":
        print ("Loading CropDisease")
        datamgr             =  CropDisease_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
        novel_loader        = datamgr.get_data_loader(aug =False)
        novel_loaders.append(novel_loader)
        ratio = 0.2
    elif params.test_dataset == "ChestX":
        log.info("Loading ChestX")
        datamgr             =  Chest_few_shot.SetDataManager(image_size, n_eposide = iter_num, n_query = 15, **few_shot_params)
        novel_loader        = datamgr.get_data_loader(aug =False)
        novel_loaders.append(novel_loader)
        ratio = 0.05
    else:
        raise ValueError('Unknown dataset')

    if params.n_shot > 5:
        total_epoch = 10
    else:
        total_epoch = 20

    forward_batch_size = 300

    #########################################################################
    for idx, novel_loader in enumerate(novel_loaders):
        log.info(params.test_dataset)
        start_epoch = params.start_epoch
        stop_epoch = params.stop_epoch
        print (freeze_backbone)

        doamin_adapation(log, novel_loader, n_query = 15, pretrained_dataset=pretrained_dataset, freeze_backbone=freeze_backbone, **few_shot_params)
