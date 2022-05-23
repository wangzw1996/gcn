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
from numpy import *
from models import NeighborSamplingGCN
num_run=5

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data','Flickr')

dataset = Flickr(path)

data = dataset[0]
torch.cuda.get_device_name(0)  

labels_np = data.y.cpu().numpy().astype(np.int32)
idx_test_np = data.test_mask.cpu().numpy()

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
    
    
    
    acc_test = []*num_run
    acc_val = []*num_run
    acc_train = []*num_run
    outs=[None]*num_run
    for j in range(num_run):
        
        out = model.inference(x, subgraph_loader, device)
        
        outs[j]=out.cpu().data.numpy()
        
        
        y_true = y.cpu().unsqueeze(-1)
        y_pred = out.argmax(dim=-1, keepdim=True)
        #print(y_pred.shape)
        
        
        acc_test.append(f1_score(y_true[data.test_mask], y_pred[data.test_mask], average='micro') 
          if y_pred[data.test_mask].sum() >0 else 0)
        acc_val.append(f1_score(y_true[data.val_mask], y_pred[data.val_mask], average='micro') 
          if y_pred[data.val_mask].sum() >0 else 0)
        acc_test.append(f1_score(y_true[data.train_mask], y_pred[data.train_mask], average='micro') 
          if y_pred[data.train_mask].sum() >0 else 0)
          
          
    outs_opt= np.stack(outs)
    c_opt=torch.tensor(outs_opt)
    c_opt=c_opt.log_softmax(dim=-1) 
    print(c_opt)
    labels_opt=torch.tensor(labels_np)      
    pavpu_opt=uncertainty_mrun(c_opt, labels_opt, idx_test_np)
    print(pavpu_opt)
        
        
        
   
    #y_true = y.cpu().unsqueeze(-1)
    #y_pred = outs.argmax(dim=-1, keepdim=True)

   # results = []
    #for mask in [train_mask1, val_mask1, test_mask1]:
     #   results.append(f1_score(y_true[mask], y_pred[mask], average='micro') if y_pred[mask].sum() > 0 else 0)
    return mean(acc_test) ,mean(acc_train),mean(acc_val),pavpu_opt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='1')
    parser.add_argument('--model', type=str, default='indGCN')            # indGCN, GraphSAGE
    parser.add_argument('--dataset', type=str, default='Flickr')          # Reddit; Flickr
    parser.add_argument('--batch', type=int, default=89250)                 # 512; 1024
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--binarize', action='store_true')
    args = parser.parse_args()
    print(args)

    assert args.dataset in ['Flickr', 'Reddit'], 'For dataset, only Flickr and Reddit are available'

    
   
    train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                                   sizes=[25, 10], batch_size=args.batch, shuffle=True,
                                   num_workers=4)
    subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                      batch_size=args.batch, shuffle=False,
                                      num_workers=4)

    device = torch.device( 'cuda:1')
    assert args.model in ['indGCN', 'GraphSAGE'], 'Only indGCN and GraphSAGE are available.'
    model = NeighborSamplingGCN(args.model, dataset.num_features,args.hidden, dataset.num_classes, args.binarize,
                                args.dropout).to(device)


    test_accs = []
    #print(data.edge_index)
    
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_test = 0.0
    best_val = 0.0
    for epoch in range(1, args.epochs+1):
             loss, acc = train(model, data, train_loader, optimizer, device)
             test_f1,train_f1, val_f1,pavpu_opt= test(model, data, subgraph_loader, device)
             print(test_f1)
             
             if val_f1>best_val:
               best_val=val_f1
               acc1 = test_f1
               a6=pavpu_opt[6]
               a7=pavpu_opt[7]
               a8=pavpu_opt[8]
               a9=pavpu_opt[9]
               
    print(acc1)
    print(acc1)
    print(a6)
    print(a7)
    print(a8)
    print(a9)
               
            
            
            
            
            


if __name__ == '__main__':
    main()
    

if __name__ == '__main__':
    main()
