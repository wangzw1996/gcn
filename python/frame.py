from __future__ import division

import time
import argparse
import os.path as osp
import torch
from torch import tensor
from torch.optim import Adam
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from models import BiGCN
import numpy as np
import torch_geometric.transforms as T
from utils import load_data, accuracy_mrun_np, normalize_torch,uncertainty_mrun


seed = 5
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    



adj, features, labels, idx_train, idx_val, idx_test = load_data('cora')

adj = torch.FloatTensor(adj.todense())
adj_normt = normalize_torch(adj + torch.eye(adj.shape[0]))

if torch.cuda.is_available():
    features = features.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    adj = adj.cuda()
    adj_normt = adj_normt.cuda()
    
labels_np = labels.cpu().numpy().astype(np.int32)
idx_train_np = idx_train.cpu().numpy()
#print(idx_train_np.shape)
idx_val_np = idx_val.cpu().numpy()
idx_test_np = idx_test.cpu().numpy()
#print(idx_test_np.shape)

def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()

    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    
    pred = out[data.train_mask].max(1)[1]
        
    acc = pred.eq(data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()
    loss.backward()

    optimizer.step()
    return acc

def evaluate(model, data):
   # model.eval()

    with torch.no_grad():
        logits = model(data)
        
    #begin1=time.time()
    outs = {}
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        pred = logits[mask].max(1)[1]
        
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        outs['{}_loss'.format(key)] = loss
        outs['{}_acc'.format(key)] = acc
    #end1=time.time()
    #print(end1-begin1)

    return logits,outs


def run(exp_name, data, model, runs, epochs, lr, weight_decay, early_stopping, device,num_run):
    val_losses, accs, durations = [], [], []
    data = data.to(device)
    model.to(device).reset_parameters()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()

    t_start = time.perf_counter()

    best_val_loss = float('inf')
    train_acc = 0
    val_acc = 0
    test_acc = 0
    val_loss_history = []
    epoch_count = -1
    max_10=0
    max_8=0
    max_5=0
    best_acc_test=0
    for i in range(1, epochs + 1):
            # print("epochs:",epoch)
        acc=train(model, optimizer, data)
        outs = [None]*num_run 
        #begin=time.time()
        outstmp,eval_info = evaluate(model, data)
        if eval_info['val_loss'] < best_val_loss:
                best_val_loss = eval_info['val_loss']
                train_acc = eval_info['train_acc']
        
        for j in range(num_run):
              
            outstmp,eval_info = evaluate(model, data)
            outs[j] = outstmp.cpu().data.numpy()
        #end=time.time()
        #print(end-begin)
        
        
       
        outs = np.stack(outs)
        c=torch.tensor(outs)
        labels=torch.tensor(labels_np)
        idx_test=torch.tensor(idx_test_np)
        d=uncertainty_mrun(c, labels, idx_test)
        acc_val_tr = accuracy_mrun_np(outs, labels_np, idx_val_np)
        acc_test_tr = accuracy_mrun_np(outs, labels_np, idx_test_np)
        if acc_test_tr > best_acc_test:
           best_acc_test=acc_test_tr
        if acc_test_tr >0.78:
          if d[8]>max_8:
            max_8=d[8]
          if d[10]>max_10:
            max_10=d[10]
          if d[5]>max_5:
            max_5=d[5]
       # print(best_acc_test)
            
    return best_acc_test,max_8
        
    
            
            
            
       


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default='0', type=str, help='gpu id')
    parser.add_argument('--exp_name', default='default_exp_name', type=str)
    parser.add_argument('--dataset', type=str, default='Cora')  # Cora/CiteSeer/PubMed
    parser.add_argument('--runs', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0005)  # 5e-4
    parser.add_argument('--early_stopping', type=int, default=0)  # 100
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.4)  # 0.5
    args = parser.parse_args()
    print(args)
    
    
    sample_num=[5,10,15]
    sample_baseline=20 
    device = torch.device('cuda')
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Cora')
    
    dataset = Planetoid(path,'Cora',transform=T.NormalizeFeatures())
    

    data = dataset[0]
    
    print("Size of train set:", data.train_mask.sum().item())
    print("Size of val set:", data.val_mask.sum().item())
    print("Size of test set:", data.test_mask.sum().item())
    print("Num classes:", dataset.num_classes)
    print("Num features:", dataset.num_features)

    model = BiGCN(dataset.num_features, args.hidden, dataset.num_classes, args.layers, args.dropout)
    acc_baseline, uncetain_baseline=run(args.exp_name, dataset[0], model, args.runs, args.epochs,           args.lr,args.weight_decay, args.early_stopping, device,sample_baseline)
    print(acc_baseline)
    for i in sample_num:
       acc, uncetain=run(args.exp_name, dataset[0], model, args.runs, args.epochs,           args.lr,args.weight_decay, args.early_stopping, device,i)
       if acc > acc_baseline-0.01:
          print(acc)
          print(i)
          break
       else :
          print(sample_baseline)
      


if __name__ == '__main__':
    main()
