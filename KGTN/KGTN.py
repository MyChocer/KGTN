import numpy as np
import math

import torch
import torch.nn.functional as F
import torch.nn as nn

class KGTN(nn.Module):
    def __init__(self, 
                 feature_dim, 
                 num_classes,
                 use_all_base,
                 use_knowledge_propagation,
                 ggnn_time_step=None,
                 pretrain=False,
                 pretrain_model=None,
                 graph_learnable=False,
                 classifier_type='inner_product',
                 adjacent_matrix=None):
        super(KGTN, self).__init__()

        self.feature_dim = feature_dim
        self.use_knowledge_propagation = use_knowledge_propagation
        self.use_all_base = use_all_base
        self.ggnn_time_step = ggnn_time_step

        self.last_fc_weight = nn.Parameter(torch.rand(feature_dim, num_classes))

        if use_knowledge_propagation:
            self.ggnn = KGTM(
                num_nodes = num_classes, 
                use_all_base = use_all_base,
                hidden_state_channel = feature_dim,
                output_channel = feature_dim,
                time_step = self.ggnn_time_step,
                adjacent_matrix = adjacent_matrix,
                graph_learnable=graph_learnable 
            )
        # initialize parameters and load pretrain
        self.param_init()
        self.load_pretrain(pretrain_model, pretrain)
        
        assert classifier_type in ['inner_product', 'cosine', 'pearson']
        self.classifier_type=classifier_type
        if self.classifier_type == 'cosine' or self.classifier_type == 'pearson':
            init_scale_cls = 10
            self.scale_cls = nn.Parameter(
                torch.FloatTensor(1).fill_(init_scale_cls),
                requires_grad=True)

    def forward(self, input):
        if self.use_knowledge_propagation:
            step_fc_weight = self.ggnn(self.last_fc_weight.transpose(0, 1).unsqueeze(0))
            weight = step_fc_weight[-1]
            weight = weight.squeeze().transpose(0, 1)
            if self.classifier_type == 'cosine':
                # cos sim
                input = F.normalize(input, p=2, dim=1, eps=1e-12)
                weight = F.normalize(weight, p=2, dim=0, eps=1e-12)
                output = torch.matmul(input, weight) * self.scale_cls
            elif self.classifier_type == 'pearson':
                # pearson corr
                input = input - torch.mean(input, 1, keepdim=True)
                weight = weight - torch.tensor(torch.mean(weight, 0, keepdim=True), requires_grad=False)
                input = F.normalize(input, p=2, dim=1, eps=1e-12)
                weight = F.normalize(weight, p=2, dim=0, eps=1e-12)
                output = torch.matmul(input, weight) * self.scale_cls
            else:
                output = torch.matmul(input, weight)
            l2_reg = self.l2_reg(weight)
        else:
            output = torch.matmul(input, self.last_fc_weight)
            l2_reg = self.l2_reg(self.last_fc_weight)
        return output, l2_reg
    
    def l2_reg(self, input):
        return input.pow(2).sum()

    def load_pretrain(self,pretrain_model, pretrain):
        if pretrain:
            pretrain = torch.load('checkpoints/{}'.format(pretrain_model))['state_dict']
            self_param = self.state_dict()
            self_param.update(pretrain)

            self.load_state_dict(self_param)

    def param_init(self):
        # init parameters
        self.last_fc_weight.data.normal_(0.0, np.sqrt(2.0/self.feature_dim))

class KGTM(nn.Module):
    def __init__(self,
                 num_nodes = 512, 
                 use_all_base = False,
                 hidden_state_channel = 20,
                 output_channel = 20,
                 time_step = 3,
                 adjacent_matrix = None,
                 graph_learnable=False):

        super(KGTM, self).__init__()
        self.num_nodes = num_nodes
        self.use_all_base = use_all_base
        self.time_step = time_step
        self.hidden_state_channel = hidden_state_channel
        self.output_channel = output_channel
        self.adjacent_matrix = adjacent_matrix
        self.graph_learnable = graph_learnable
        #  form the connect matrix 
        self._in_matrix,self._out_matrix = self.load_adjacent_matrix(self.adjacent_matrix)
        
        self._in_matrix = nn.Parameter(torch.from_numpy(self._in_matrix), requires_grad=graph_learnable)
        self._out_matrix = nn.Parameter(torch.from_numpy(self._out_matrix), requires_grad=graph_learnable)

        self.fc_eq3_w = nn.Linear(2 * hidden_state_channel,hidden_state_channel, bias = False)
        self.fc_eq3_u = nn.Linear(hidden_state_channel,hidden_state_channel, bias = False)
        self.fc_eq4_w = nn.Linear(2 * hidden_state_channel,hidden_state_channel, bias = False)
        self.fc_eq4_u = nn.Linear(hidden_state_channel,hidden_state_channel, bias = False)
        self.fc_eq5_w = nn.Linear(2 * hidden_state_channel,hidden_state_channel, bias = False)
        self.fc_eq5_u = nn.Linear(hidden_state_channel,hidden_state_channel, bias = False)

        self.transform_fc = nn.Linear(hidden_state_channel, hidden_state_channel, bias=False)

        self.fc_output = nn.Linear(2 * hidden_state_channel,output_channel)

        self._initialize_weights()

    def forward(self, input):
        outputs_per_step = []
        # input : batch * 512 * 10
        batch_size = input.size()[0]
        input = input.view(-1,self.hidden_state_channel)

        batch_aog_nodes = input.view(-1,self.num_nodes,self.hidden_state_channel)

        batch_in_matrix = self._in_matrix.repeat(batch_size,1).view(batch_size,self.num_nodes,-1)
        batch_out_matrix = self._out_matrix.repeat(batch_size,1).view(batch_size,self.num_nodes,-1)

        # propogation process
        for t in range(self.time_step):
             # eq(2)
            av = torch.cat((torch.bmm(batch_in_matrix,batch_aog_nodes) ,torch.bmm(batch_out_matrix,batch_aog_nodes)),2)
            av = av.view(batch_size * self.num_nodes,-1)
            flatten_aog_nodes = batch_aog_nodes.view(batch_size * self.num_nodes,-1)
            # eq(3)
            zv = torch.sigmoid(self.fc_eq3_w(av) + self.fc_eq3_u(flatten_aog_nodes))
            # eq(4)
            rv = torch.sigmoid(self.fc_eq4_w(av) + self.fc_eq4_u(flatten_aog_nodes))
            # eq(5)
            hv = torch.tanh(self.fc_eq5_w(av) + self.fc_eq5_u(rv * flatten_aog_nodes))
            # hv = self.fc_eq5_w(av) + self.fc_eq5_u(rv * flatten_aog_nodes)
            # eg(6)
            flatten_aog_nodes = (1 - zv) * flatten_aog_nodes + zv *  hv

            batch_aog_nodes = flatten_aog_nodes.view(batch_size,self.num_nodes,-1)

            # compute the output of each step
            step_output = torch.cat((flatten_aog_nodes,input),1)
            # step_output = flatten_aog_nodes
            step_output = self.fc_output(step_output)
            step_output = step_output.view(batch_size,self.num_nodes,-1)
            outputs_per_step.append(step_output)

        return outputs_per_step


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
        
        self.fc_eq3_w.weight.data.zero_()
        self.fc_eq3_u.weight.data.zero_()
        self.fc_eq4_w.weight.data.zero_()
        self.fc_eq4_u.weight.data.zero_()
        self.fc_eq5_w.weight.data.zero_()
        self.fc_eq5_u.weight.data.zero_()

        self.transform_fc.weight.data = torch.eye(self.hidden_state_channel)
  
    def load_adjacent_matrix(self, mat):
        in_matrix = mat
        out_matrix = in_matrix.transpose()

        return in_matrix.astype(np.float32),out_matrix.astype(np.float32)





