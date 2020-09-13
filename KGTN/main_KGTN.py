import torch
import torch.nn as nn
import torch.optim
import torch.utils.data as Data
import torch.nn.functional as F

import h5py
import json
import argparse
import os
import json
import numpy as np

from KGTN import KGTN
from util import process_adjacent_matrix

import random

seed = 192
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

class SimpleHDF5Dataset:
    def __init__(self, file_handle):
        self.f = file_handle
        self.all_feats_dset = self.f['all_feats'][...]
        self.all_labels = self.f['all_labels'][...]
        self.total = self.f['count'][0]
        print('here')
    def __getitem__(self, i):
        return torch.Tensor(self.all_feats_dset[i,:]), int(self.all_labels[i])

    def __len__(self):
        return int(self.total)

# a dataset to allow for category-uniform sampling of base and novel classes.
# also incorporates hallucination
class LowShotDataset:
    def __init__(self, file_handle, base_classes, novel_classes, novel_idx):
        self.f = file_handle
        self.all_feats_dset = self.f['all_feats']
        all_labels_dset = self.f['all_labels']
        self.all_labels = all_labels_dset[...]

        #base class examples
        self.base_class_ids = np.where(np.in1d(self.all_labels, base_classes))[0]
        total = self.f['count'][0]
        self.base_class_ids = self.base_class_ids[self.base_class_ids<total]


        # novel class examples
        novel_feats = self.all_feats_dset[novel_idx,:]
        novel_labels = self.all_labels[novel_idx]

        # hallucinate if needed
        self.novel_feats = novel_feats
        self.novel_labels = novel_labels

        self.base_classes = base_classes
        self.novel_classes = novel_classes
        self.frac = 0.5
        self.all_classes = np.concatenate((base_classes, novel_classes))

    def sample_base_class_examples(self, num):
        sampled_idx = np.sort(np.random.choice(self.base_class_ids, num, replace=False))
        return torch.Tensor(self.all_feats_dset[sampled_idx,:]), torch.LongTensor(self.all_labels[sampled_idx].astype(int))

    def sample_novel_class_examples(self, num):
        sampled_idx = np.random.choice(self.novel_labels.size, num)
        return torch.Tensor(self.novel_feats[sampled_idx,:]), torch.LongTensor(self.novel_labels[sampled_idx].astype(int))

    def get_sample(self, batchsize):
        # num_base = round(self.frac*batchsize)
        num_base = int(round(self.frac*batchsize))
        num_novel = batchsize - num_base
        base_feats, base_labels = self.sample_base_class_examples(num_base)
        novel_feats, novel_labels = self.sample_novel_class_examples(num_novel)
        return torch.cat((base_feats, novel_feats)), torch.cat((base_labels, novel_labels))

    def featdim(self):
        return self.novel_feats.shape[1]

# simple data loader for test
def get_test_loader(file_handle, batch_size=1000):
    testset = SimpleHDF5Dataset(file_handle)
    data_loader = Data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return data_loader

def training_loop(lowshot_dataset, model, num_classes, params, batchsize=1000, maxiters=1000):
    featdim = lowshot_dataset.featdim()
    optimizer = torch.optim.SGD(filter(lambda p:p.requires_grad==True, model.parameters()), params.lr, momentum=params.momentum, dampening=params.momentum, weight_decay=params.wd)

    loss_function = nn.CrossEntropyLoss()
    loss_function = loss_function.cuda()

    for i in range(maxiters):  
        model.train()      
        (x,y) = lowshot_dataset.get_sample(batchsize)
        
        optimizer.zero_grad()

        output, l2_reg = model(x.cuda())

        cls_loss = loss_function(output,y.cuda())
        total_loss = cls_loss + params.l2_reg * l2_reg

        total_loss.backward()
        optimizer.step()

        if (i%100==0):
            print('[Batch Idx]:{:d}: [cls loss]:{:f} [reg loss]:{:f}'.format(i, cls_loss.item(), params.l2_reg * l2_reg.item()))
        if (i%1000==0):
            with h5py.File(params.testfile, 'r') as f:
                test_loader = get_test_loader(f)
                with torch.no_grad():
                    eval_loop_step(test_loader, model, base_classes, novel_classes)
    return model

def perelement_accuracy(scores, labels):
    topk_scores, topk_labels = scores.topk(5, 1, True, True)
    label_ind = labels.cpu().numpy()
    topk_ind = topk_labels.cpu().numpy()
    top1_correct = topk_ind[:,0] == label_ind
    top5_correct = np.sum(topk_ind == label_ind.reshape((-1,1)), axis=1)
    return top1_correct.astype(float), top5_correct.astype(float)


def eval_loop_step(data_loader, model, base_classes, novel_classes):

    model = model.eval()
    top1 = None
    top5 = None
    top1_new_metrix = None
    top5_new_metrix = None
    top1_with_prior = None
    top5_with_prior = None
    
    all_labels = None

    for i, (x,y) in enumerate(data_loader):
        output, _ = model(x.cuda())

        output_join = output * 0 - 999
        output_join[:, base_classes + novel_classes] = output[:, base_classes + novel_classes]
        top1_this, top5_this = perelement_accuracy(output_join.data, y)
        top1 = top1_this if top1 is None else np.concatenate((top1, top1_this))
        top5 = top5_this if top5 is None else np.concatenate((top5, top5_this))

        # new metrix, evaluate novel class on novel space
        output_new_metrix = output * 0 - 999
        output_new_metrix[:, novel_classes] = output[:, novel_classes]
        top1_this_new_metrix, top5_this_new_metrix = perelement_accuracy(output_new_metrix.data, y)
        top1_new_metrix = top1_this_new_metrix if top1_new_metrix is None else np.concatenate((top1_new_metrix, top1_this_new_metrix))
        top5_new_metrix = top5_this_new_metrix if top5_new_metrix is None else np.concatenate((top5_new_metrix, top5_this_new_metrix))
            
        # new metrix, evaluate all class on space with prior
        mu = 0.8
        output_with_prior = output * 0 - 999
        output_with_prior[:, novel_classes] = F.softmax(output[:, novel_classes], 1) * mu
        output_with_prior[:, base_classes] = F.softmax(output[:, base_classes], 1) * ( 1 - mu )

        top1_this_with_prior, top5_this_with_prior = perelement_accuracy(output_with_prior.data, y)
        top1_with_prior = top1_this_with_prior if top1_with_prior is None else np.concatenate((top1_with_prior, top1_this_with_prior))
        top5_with_prior = top5_this_with_prior if top5_with_prior is None else np.concatenate((top5_with_prior, top5_this_with_prior))
    
        all_labels = y.numpy() if all_labels is None else np.concatenate((all_labels, y.numpy()))

    is_novel = np.in1d(all_labels, novel_classes)
    is_base = np.in1d(all_labels, base_classes)
    is_either = is_novel | is_base

    top1_novel = np.mean(top1[is_novel]) 
    top1_all = np.mean(top1[is_either]) 
    top5_novel = np.mean(top5[is_novel]) 
    top5_all = np.mean(top5[is_either]) 

    # new metrix
    top1_novel_new_metrix = np.mean(top1_new_metrix[is_novel]) 
    top5_novel_new_metrix = np.mean(top5_new_metrix[is_novel]) 
    # all with prior
    top1_all_with_prior = np.mean(top1_with_prior[is_either]) 
    top5_all_with_prior = np.mean(top5_with_prior[is_either]) 
    
    print("**********************************Testing**********************************")
    print("novel in all  :  {:<8.2f}".format(top5_novel*100))
    print("novel in novel:  {:<8.2f}".format(top5_novel_new_metrix*100))
    print("all           :  {:<8.2f}".format(top5_all*100))
    print("all with prior:  {:<8.2f}".format(top5_all_with_prior*100))

    print("***************************************************************************")
    
    return np.array([top5_novel, top5_novel_new_metrix, top5_all, top5_all_with_prior])

def parse_args():
    parser = argparse.ArgumentParser(description='Low shot benchmark')
    parser.add_argument('--lowshotmeta', required=True, type=str, help='set of base and novel classes')
    parser.add_argument('--experimentpath', required=True, type=str, help='path of experiments')
    parser.add_argument('--experimentid', default=1, type=int, help='id of experiment')
    parser.add_argument('--lowshotn', required=True, type=int, help='number of examples per novel class')
    parser.add_argument('--trainfile', required=True, type=str)
    parser.add_argument('--testfile', required=True, type=str)
    parser.add_argument('--testsetup', default=0, type=int, help='test setup or validation setup?')
    parser.add_argument('--numclasses', default=1000, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=0.001, type=float)
    parser.add_argument('--l2_reg', default=0.001, type=float)
    parser.add_argument('--maxiters', default=10000, type=int)
    parser.add_argument('--batchsize', default=1000, type=int)
    parser.add_argument('--outdir', type=str, help='output directory for results')
    parser.add_argument('--use_knowledge_propagation',action='store_true', help='whether use KGTN')
    parser.add_argument('--use_all_base',action='store_true', help='whether use all base category to do propagation')
    parser.add_argument('--ggnn_time_step',default=2, type=int, help='ggnn propagation time')
    parser.add_argument('--ggnn_coefficient',default=0.5, type=float, help='ggnn ggnn_coefficient')
    parser.add_argument('--process_type',default='wordnet', type=str, help='output directory for results')
    parser.add_argument('--adjacent_matrix_file', type=str, help='adjacent matrix file')
    parser.add_argument('--kg_ratio',default=100, type=int, help='ratio of the kg information')
    parser.add_argument('--classifier_type',default='inner_product', type=str, help='classifier_type')
    return parser.parse_args()

if __name__ == '__main__':
    params = parse_args()
    with open(params.lowshotmeta, 'r') as f:
        lowshotmeta = json.load(f)
    accs = np.zeros(6)

    with open(params.experimentpath.format(params.experimentid),'r') as f:
        exp = json.load(f)
    novel_idx = np.array(exp)[:,:params.lowshotn]
    if params.testsetup:
        novel_classes = lowshotmeta['novel_classes_2']
        base_classes = lowshotmeta['base_classes_2']
    else:
        novel_classes = lowshotmeta['novel_classes_1']
        base_classes = lowshotmeta['base_classes_1']

    if params.use_all_base:
        train_base_classes = lowshotmeta['base_classes_1'] + lowshotmeta['base_classes_2']
    else:
        train_base_classes = base_classes

    novel_idx = np.sort(novel_idx[novel_classes,:].reshape(-1))
    
    with h5py.File(params.trainfile, 'r') as f:
        lowshot_dataset = LowShotDataset(f, train_base_classes, novel_classes, novel_idx)
        adjacent_matrix = process_adjacent_matrix(params.lowshotmeta, params.testsetup, params.adjacent_matrix_file, params.ggnn_coefficient, params.process_type, params.kg_ratio, use_all_base=params.use_all_base) if params.use_knowledge_propagation else None
        model = KGTN(
                           lowshot_dataset.featdim(), 
                           params.numclasses,
                           use_all_base=params.use_all_base,
                           use_knowledge_propagation=params.use_knowledge_propagation,
                           ggnn_time_step=params.ggnn_time_step,
                           pretrain=False,
                           adjacent_matrix = adjacent_matrix,
                           classifier_type=params.classifier_type,
                           )
        model = model.cuda()

        model = training_loop(lowshot_dataset, model, params.numclasses, params, params.batchsize, params.maxiters)

    print('trained')
    with h5py.File(params.testfile, 'r') as f:
        test_loader = get_test_loader(f)
        with torch.no_grad():
            accs = eval_loop_step(test_loader, model, base_classes, novel_classes)
    
    if not os.path.exists(params.outdir):
        os.makedirs(params.outdir)
        
    modelrootdir = os.path.basename(os.path.dirname(params.trainfile))
    outpath = os.path.join(params.outdir, modelrootdir+'_lr_{:.3f}_wd_{:.3f}_expid_{:d}_lowshotn_{:d}.json'.format(
                                    params.lr, params.wd, params.experimentid, params.lowshotn))
    with open(outpath, 'w') as f:
        json.dump(dict(lr=params.lr,wd=params.wd, expid=params.experimentid, lowshotn=params.lowshotn, accs=accs.tolist()),f)


