# -*- coding: utf-8 -*-
"""
Created on Fri May 20 22:05:20 2022

@author: wangz
"""

import time 

import numpy as np
import argparse

import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Reddit, Flickr
from torch_geometric.data import NeighborSampler
from sklearn.metrics import f1_score
from utils import uncertainty_mrun,accuracy_mrun_np

from models import NeighborSamplingGCN
num_run=2




def train(model, data, train_loader, optimizer, device):
    x = data.x.to(device)
    y = data.y.squeeze().to(device)
    print('train')
    model.train()

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]

        optimizer.zero_grad()
        out = model(x[n_id], adjs)
        loss = F.nll_loss(out, y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / int(data.train_mask.sum())

    return loss, approx_acc


@torch.no_grad()
def test(model, data, subgraph_loader, device):
    print('test')
    x = data.x.to(device)
    y = data.y.squeeze().to(device)
    model.eval()
    
    
    outs = [None]*num_run 
    for j in range(num_run):
        outstmp = model.inference(x, subgraph_loader, device)
        outs[j] = outstmp.cpu().data.numpy()
        
        
    
    y_true = y.cpu().unsqueeze(-1)
    y_pred = outs.argmax(dim=-1, keepdim=True)

    results = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        results.append(f1_score(y_true[mask], y_pred[mask], average='micro') if y_pred[mask].sum() > 0 else 0)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--model', type=str, default='indGCN')            # indGCN, GraphSAGE
    parser.add_argument('--dataset', type=str, default='Flickr')          # Reddit; Flickr
    parser.add_argument('--batch', type=int, default=1024)                 # 512; 1024
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--binarize', action='store_true')
    args = parser.parse_args()
    print(args)

    assert args.dataset in ['Flickr', 'Reddit'], 'For dataset, only Flickr and Reddit are available'

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Flickr')
    if args.dataset == 'Flickr':
        dataset = Flickr(path)
    else:
        dataset = Reddit(path)
    
    data = dataset[0]
    
    labels_np = data.y.cpu().numpy().astype(np.int32)
    idx_train_np = data.train_mask.cpu().numpy()

    idx_val_np = data.val_mask.cpu().numpy()
    idx_test_np = data.test_mask.cpu().numpy()
    
    
   


    train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                                   sizes=[25, 10], batch_size=args.batch, shuffle=True,
                                   num_workers=4)
    subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                      batch_size=args.batch, shuffle=False,
                                      num_workers=4)

    device = torch.device( 'cpu')
    assert args.model in ['indGCN', 'GraphSAGE'], 'Only indGCN and GraphSAGE are available.'
    model = NeighborSamplingGCN(args.model, dataset.num_features, args.hidden, dataset.num_classes, args.binarize,
                                args.dropout).to(device)


    test_accs = []
    print(data.edge_index)
    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_test = 0.0
        best_val = 0.0
        for epoch in range(1, args.epochs+1):
             loss, acc = train(model, data, train_loader, optimizer, device)
             train_f1, val_f1, test_f1 = test(model, data, subgraph_loader, device)
             print(test_f1)
             
             
               
            
            
            
            
            


if __name__ == '__main__':
    main()
    

if __name__ == '__main__':
    main()
