# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 18:58:09 2022

@author: wangz
"""
import time 

import torch
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops

from layers import BiGCNConv, indBiGCNConv, BiSAGEConv, BiGraphConv
from Function import BinActive, BinActive0,BinLinear

import torch.nn.functional as F
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.quantized import QFunctional


class BernoulliDropout(nn.Module):
    def __init__(self, p=0.0):
        super(BernoulliDropout, self).__init__()
        self.p = torch.nn.Parameter(torch.ones((1,))*p, requires_grad=False)
        if self.p < 1:
            self.multiplier = torch.nn.Parameter(
                torch.ones((1,))/(1.0 - self.p), requires_grad=False)
        else:
            self.multiplier = torch.nn.Parameter(
                torch.ones((1,))*0.0, requires_grad=False)

        self.mul_mask = torch.nn.quantized.FloatFunctional()
        self.mul_scalar = torch.nn.quantized.FloatFunctional()
        
    def forward(self, x):
        if self.p <= 0.0:
            return x
        mask_ = None
        if len(x.shape) <= 2:
            if x.is_cuda:
                mask_ = torch.cuda.FloatTensor(x.shape).bernoulli_(1.-self.p)
            else:
                mask_ = torch.FloatTensor(x.shape).bernoulli_(1.-self.p)
        else:
            
            if x.is_cuda:
                mask_ = torch.cuda.FloatTensor(x.shape[:2]).bernoulli_(
                    1.-self.p)
            else:
                mask_ = torch.FloatTensor(x.shape[:2]).bernoulli_(
                    1.-self.p)
        if isinstance(self.mul_mask, QFunctional):
            scale = self.mul_mask.scale
            zero_point = self.mul_mask.zero_point
            mask_ = torch.quantize_per_tensor(
                mask_, scale, zero_point, dtype=torch.quint8)
        if len(x.shape) > 2:
            
            mask_ = mask_.view(
                mask_.shape[0], mask_.shape[1], 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
        x = self.mul_mask.mul(x, mask_)
        x = self.mul_scalar.mul_scalar(x, self.multiplier.item())
        return x

    def extra_repr(self):
        return 'p={}, quant={}'.format(
            self.p.item(), isinstance(
                self.mul_mask, QFunctional)
        )
        
        
class BiGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, layers, dropout, print_log=True):
        super(BiGCN, self).__init__()

        if print_log:
            print("Create a {:d}-layered Bi-GCN.".format(layers))

        self.layers = layers
        self.dropout = dropout
        self.bn1 = torch.nn.BatchNorm1d(in_channels, affine=False)

        convs = []
        for i in range(self.layers):
        
            if i==0 :
              in_dim = in_channels  
            if i==1:  
              in_dim =hidden_channels 
            if 1<i< self.layers-1:
              in_dim= 128
            
            if i==0 :
              out_dim = hidden_channels  
            if 0 < i < self.layers-1:  
              out_dim =128 
            if i== self.layers-1:
              out_dim= out_channels
            if print_log:
                print("Layer {:d}, in_dim {:d}, out_dim {:d}".format(i, in_dim, out_dim))
            convs.append(BiGCNConv(in_dim, out_dim, cached=True, bi=True))
            #if i < self.layers -1:
             #  convs.append (BernoulliDropout(self.dropout))
     
        self.convs = torch.nn.ModuleList(convs)

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):
       
       
        x, edge_index = data.x, data.edge_index 
        #print(edge_index.shape)  
        #print(edge_index)     
        #begin6 = time.time()
        x = self.bn1(x)
        #end6 = time.time()
        #print(end6-begin6)
        

        for i, conv in enumerate(self.convs):
            
            #begin4 = time.time()
            #x = x - x.mean(dim=0, keepdim=True)
            #x = x / (x.std(dim=0, keepdim=True) + 0.0001)
            x = BinActive()(x) 
            #print(x)
            #x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            if i != self.layers - 1:
                 #x = F.relu(x)
                 x = BernoulliDropout(0.9)(x)
            
                 #print(x)
            
            
            #end4 = time.time()
            #print(end4-begin4)

            #print(x.shape)
            #print(edge_index.shape)
            # begin = time.time()
                     
            #end = time.time()
            #print(end-begin)
                     
        x.cpu()              

        return F.log_softmax(x, dim=1)


# indGCN and GraphSAGE
class NeighborSamplingGCN(torch.nn.Module):
    def __init__(self, model: str, in_channels, hidden_channels, out_channels, binarize, dropout=0.):
        super(NeighborSamplingGCN, self).__init__()

        assert model in ['indGCN', 'GraphSAGE'], 'Only indGCN and GraphSAGE are available.'
        GNNConv = indBiGCNConv if model == 'indGCN' else BiSAGEConv

        self.num_layers = 2
        self.model = model
        self.binarize = binarize
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(GNNConv(in_channels, hidden_channels, binarize=binarize))
        self.convs.append(GNNConv(hidden_channels, out_channels, binarize=binarize))
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):

        for i, (edge_index, _, size) in enumerate(adjs):
            
            x_target = x[:size[1]]
            
            x = x - x.mean(dim=0, keepdim=True)
            x = x / (x.std(dim=0, keepdim=True) + 0.0001)
            x = BinActive0()(x)
            
            
            x_target = x_target - x_target.mean(dim=0, keepdim=True)
            x_target = x_target / (x_target.std(dim=0, keepdim=True) + 0.0001)
            x_target = BinActive0()(x_target)
           
            # if self.model == 'GraphSAGE':
            #     edge_index, _ = add_self_loops(edge_index, num_nodes=x[0].size(0))
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all, subgraph_loader, device, return_lat=False):
        begin_time = time.time()
        load_time = 0.0
        bin_active_time = 0.0
        conv_time = 0.0
        #torch.no_grad()
        #torch.set_num_interop_threads(72)
        #torch.set_num_threads(72)
        #print ("Number of Inter Thred:", torch.get_num_interop_threads())
        #print ("Number of Intra Thread:", torch.get_num_threads())
        for i in range(self.num_layers):
            xs = []
            #print ("layer no.:", i)
            layer_begin_time = time.time()
            for batch_size, n_id, adj in subgraph_loader:
                load_begin_time = time.time()
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                load_end_time = time.time()
                load_time += load_end_time - load_begin_time
                
                bin_begin_time = time.time()
                x = x - x.mean(dim=0, keepdim=True)
                x = x / (x.std(dim=0, keepdim=True) + 0.0001)
                x = BinActive()(x)

                # bn x_target
                x_target = x_target - x_target.mean(dim=0, keepdim=True)
                x_target = x_target / (x_target.std(dim=0, keepdim=True) + 0.0001)
                x_target = BinActive()(x_target)
                bin_end_time = time.time()
                bin_active_time += bin_end_time - bin_begin_time
                
                #print("input shape:", x.shape)
                #print("edge_index shape:", edge_index.shape)
                #print("target_shape:", x_target.shape)
                conv_begin_time = time.time()                 
                x = self.convs[i]((x,x_target), edge_index)
                
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())
                
                conv_end_time = time.time()
                conv_time += conv_end_time - conv_begin_time
                 
            x_all = torch.cat(xs, dim=0)
            layer_end_time = time.time()
            #print ("No. %d consumes %f s"%(i, layer_end_time - layer_begin_time))
        end_time = time.time()
        total_time = end_time - begin_time
        if return_lat:
            return x_all, total_time, load_time, bin_active_time, conv_time 
        else:
            return x_all


class SAINT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, binarize):
        super(SAINT, self).__init__()
        self.dropout = dropout
        self.binarize = binarize
        self.conv1 = BiGraphConv(in_channels, hidden_channels, binarize=self.binarize)
        self.conv2 = BiGraphConv(hidden_channels, hidden_channels, binarize=self.binarize)
        # if self.binarize:
        #     self.lin = BinLinear(2 * hidden_channels, out_channels)
        # else:
        self.lin = torch.nn.Linear(2 * hidden_channels, out_channels)

    def set_aggr(self, aggr):
        self.conv1.aggr = aggr
        self.conv2.aggr = aggr

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x0, edge_index, edge_weight=None):
        if self.binarize:
            x0 = x0 - x0.mean(dim=1, keepdim=True)
            x0 = x0 / (x0.std(dim=1, keepdim=True) + 0.0001)
            x0 = BinActive()(x0)

        x1 = self.conv1(x0, edge_index, edge_weight)
        if not self.binarize:
            x1 = F.relu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)

        if self.binarize:
            x2 = BinActive()(x1)
        else:
            x2 = x1
        x2 = self.conv2(x2, edge_index, edge_weight)
        if not self.binarize:
            x2 = F.relu(x2)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)

        x = torch.cat([x1, x2], dim=-1)
        x = self.lin(x)
        return x.log_softmax(dim=-1)
