import backbone
import utils

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from utils import AverageMeter


class cosine_classifer(nn.Module): 
    def __init__(self, indim, outdim, scale = 20, imprint_weight = None):
        super(cosine_classifer, self).__init__()

        if imprint_weight is None:
            weight = torch.FloatTensor(outdim,indim).normal_(0.0,np.sqrt(2.0/indim))
        else:
            weight = imprint_weight
        self.scale = scale
        self.weight=nn.Parameter(weight.data,requires_grad=True)

    def forward(self, x):
        x_norm = F.normalize(x,p=2, dim=1)
        weight_norm = F.normalize(self.weight,p=2,dim=1)
        cos_sim = torch.mm(x_norm, weight_norm.t())
        return self.scale * cos_sim

class BPLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(BPLayer, self).__init__()
        self.proj0 = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, bias=False).cuda()
        self.proj1 = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, bias=False).cuda()
        nn.init.kaiming_normal_(self.proj0.weight)
        nn.init.kaiming_normal_(self.proj1.weight)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x ):
        feature0 = self.proj0(x)
        feature1 = self.proj1(x)
        bp_features = feature0 * feature1
        out = self.avgpool(bp_features).view(x.size(0), -1)
        return out

class Multi_Domain_Train(nn.Module):
    def __init__(self, model_func, num_class_list, loss_type = 'softmax'):
        super(Multi_Domain_Train, self).__init__()
        self.feature    = model_func(flatten = False)

        self.feat_dim = 8192
        self.num_domains = len(num_class_list)
        self.num_class_list = num_class_list

        self.BP_list = nn.ModuleList([BPLayer(self.feature.final_feat_dim[0], self.feat_dim).cuda() for i in range(self.num_domains)])
        self.classifier_list = nn.ModuleList([cosine_classifer(self.feat_dim , num_class).cuda() for num_class in self.num_class_list])

        self.domain_classifier = cosine_classifer(self.feature.final_feat_dim[0] , self.num_domains).cuda()
        
        self.loss_type = loss_type  #'softmax' #'dist'
        self.num_class = sum(num_class_list)
        self.loss_fn = nn.CrossEntropyLoss()
        self.top1 = utils.AverageMeter()

    def forward(self,x):
        x    = x.cuda()
        out  = self.feature.forward(x)
        feature0 = self.proj0(out)
        feature1 = self.proj1(out)
        bp_features = feature0 * feature1
        
        out = self.avgpool(bp_features).view(x.size(0), -1)
        # out = F.normalize(out, p =2, dim = -1)
        scores  = self.classifier.forward(out)
        return scores
    
    def forward_train(self, x, y, y_domain):
        x    = x.cuda()
        out  = self.feature.forward(x)
        classification_loss = 0
        loss_list = []
        for i in range(self.num_domains):
            a = y_domain.cpu().numpy()
            domain_index = np.argwhere( a == i )
            if domain_index.size > 0 :
                BP_features = self.BP_list[i](out[domain_index[:,0]])
                scores = self.classifier_list[i](BP_features)
                classification_loss = self.loss_fn(scores, y[domain_index[:,0]])
                loss_list.append(classification_loss)
        domain_scores  = self.domain_classifier.forward(out.mean(-1).mean(-1))
        domain_loss = self.loss_fn(domain_scores, y_domain)
        return loss_list, domain_loss

    def forward_features(self,x, batch_size = None):
        if batch_size is None:
            x    = x.cuda()
            BP_features_list = []
            out  = self.feature.forward(x)
            for BP in self.BP_list:
                BP_features = BP(out)
                BP_features_list.append(BP_features)
            return out.mean(-1).mean(-1), BP_features_list
    
        else:
            x    = x.cuda()
            BP_features_list = []
            sample_size = x.size(0)
            for index in range(0, sample_size, batch_size):
                sample_batch = x[index: min(index+batch_size,sample_size)]
                if index == 0:
                    out  = self.feature.forward(sample_batch)
                    for BP in self.BP_list:
                        BP_features = BP(out)
                        BP_features_list.append(BP_features)
                else:
                    out_this = self.feature.forward(sample_batch)
                    out = torch.cat((out,out_this), dim=0)
                    for i, BP in enumerate(self.BP_list):
                        BP_features = BP(out_this)
                        BP_features_list[i] = torch.cat((BP_features_list[i],BP_features), dim = 0)
            return out.mean(-1).mean(-1), BP_features_list
    
    def forward_loss(self, x, y):
        y = Variable(y.cuda())
        scores = self.forward(x)

        _, predicted = torch.max(scores.data, 1)
        correct = predicted.eq(y.data).cpu().sum()
        self.top1.update(correct.item()*100 / (y.size(0)+0.0), y.size(0))  

        return self.loss_fn(scores, y)
    
    def train_loop(self, epoch, train_loader, optimizer, log):

        acc = AverageMeter()
        loss_record = AverageMeter()
        iterations = len(train_loader)

        # for i, (x,y) in enumerate(train_loader):
        with tqdm(total=iterations, bar_format='{l_bar} {bar} {n_fmt}/{total_fmt} {postfix}',\
            desc='Epoch {}'.format(epoch)) as tq: 
            for (x,y,y_domain) in train_loader:

                optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                y_domain = y_domain.cuda()

                loss_list, domain_loss = self.forward_train(x,y,y_domain)

                loss = sum(loss_list) + domain_loss

                loss_record.update(loss.item(), 1)

                loss.backward()
                optimizer.step()
                # optimizer1.step()

                # prec = self.accuracy(scores,y)

                # acc.update(prec, 1)

                tq.set_postfix_str('Loss = {:.5f}'.format(loss_record.avg))
                tq.update()

        log.info('Epoch {:d} | iterations {:d} | Loss {:4f}'.format(epoch, iterations, loss_record.avg))
        print(loss)

    def accuracy(self, output, target):
        with torch.no_grad():
            batch_size = target.size(0)
            correct = 0
            _, predicted = torch.max(output, 1)
            c = (predicted == target).squeeze()
            correct = c.sum().item()
            return (correct / batch_size * 100.0)

    def test_loop(self, val_loader):
        return -1 #no validation, just save model during iteration