from __future__ import division

import time
import argparse
import os.path as osp
import torch
from torch import tensor
from torch.optim import Adam
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from models import BiGCN_layerspar
import numpy as np
import torch_geometric.transforms as T
from utils import load_data, accuracy_mrun_np, normalize_torch,uncertainty_mrun


seed = 5
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    
num_run = 5   



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
idx_val_np = idx_val.cpu().numpy()
idx_test_np = idx_test.cpu().numpy()

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
    model.eval()
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
    
    
def evaluate1(model, data):
    model.eval()
    
    with torch.no_grad():
        logits = model.inference1(data,3)
        outs_opt = [None]*num_run
                     
        for j in range(num_run):
              
            logits2 = model.inference2(logits,3,data.edge_index)  
            outs_opt[j] = logits2.cpu().data.numpy()
        
              
        outs_opt= np.stack(outs_opt)
        c_opt=torch.tensor(outs_opt)
        labels_opt=torch.tensor(labels_np)
        idx_test=torch.tensor(idx_test_np)
        pavpu_opt=uncertainty_mrun(c_opt, labels_opt, idx_test)
        acc_val_opt = accuracy_mrun_np(outs_opt, labels_np, idx_val_np)
        acc_test_opt= accuracy_mrun_np(outs_opt, labels_np, idx_test_np)
           

    return acc_test_opt,pavpu_opt


def run(exp_name, data, model, runs, epochs, lr, weight_decay, early_stopping, device):
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
    best_acc_base=0
    best_acc_opt=0
    
    
       
        
        
    
    for i in range(1, epochs + 1):           
        acc=train(model, optimizer, data)
              
        acc_test_opt,pavpu_opt = evaluate1(model, data)
        if acc_test_opt > best_acc_opt:
           best_acc_opt=acc_test_opt
        print(acc_test_opt)
                     
        print(pavpu_opt)
    print(best_acc_opt)
        
    
    
            
            
            
       


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
    parser.add_argument('--layers', type=int, default=4)
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

    model = BiGCN_layerspar(dataset.num_features, args.hidden, dataset.num_classes, args.layers, args.dropout)
   
    
    run(args.exp_name, dataset[0], model, args.runs, args.epochs, args.lr, args.weight_decay,
        args.early_stopping, device)
       
      


if __name__ == '__main__':
    main()
