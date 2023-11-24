
import numpy as np
from sklearn import discriminant_analysis
from compute_eer_and_mdcf import *
from sre14_io import *
import torch
import os
import torch
from torch_geometric.data import Data, DataLoader
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import GCNConv,GATConv
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch_geometric.loader as Load
from torch_sparse import SparseTensor
from torch.nn import Sequential, Linear, BatchNorm1d
from compute_eer_and_mdcf import *
from sre14_baseline import *
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True
     
# 设置随机数种子

class SreData(Data):
    def __init__(self, edge_index,x, y, train_mask, test_mask,num_node_features, num_class):
        super(SreData, self).__init__()
        self.edge_index = SparseTensor(row = edge_index[0],col = edge_index[1])
        self.x = x
        self.y = y
        self.train_mask = train_mask
        self.test_mask  = test_mask
        self.num_edge_features = num_node_features
        self.num_class = num_class

        
def sre14_load_data(save_data_path,device, thresh = 0.3):
    
    
    [model_ivec_, model_key] = torch.load(save_data_path + "model_data.pth")
    # dev_ids, dev_durations, dev_ivec = load_ivectors(raw_data_path + 'dev_ivectors.csv')
    # model_ids, model_durations, model_ivec = load_ivectors(raw_data_path + 'model_ivectors.csv')
    # test_ids, test_durations, test_ivec = load_ivectors(raw_data_path + 'test_ivectors.csv')
    # model_key = load_model_key(raw_data_path + 'target_speaker_models.csv', model_ids)
    [dev_ivec, model_ivec, avg_model_ivec] = torch.load(save_data_path + "speaker_model.pth")
    [test_ivec, test_key] = torch.load(save_data_path + "test.pth")
    
    adj = []
    print("model_ivec size {}".format(model_ivec.shape[0]))
    print("test vec size is {}".format(test_ivec.shape[0]))
    train_len = model_ivec.shape[0]
    test_len = test_ivec.shape[0]

    ivec_all = np.vstack((model_ivec, test_ivec))
    
   
    import time
    st = time.time()
    norm = np.linalg.norm(ivec_all, axis = 1)
    mat = np.dot(ivec_all, ivec_all.T)
    edge_idx = []
    train_mask = np.zeros(train_len+test_len)
    train_mask[:train_len] = 1
    test_mask = np.zeros(train_len+test_len)
    test_mask[train_len:]=1
    
    train_mask.astype(np.bool)
    test_mask.astype(np.bool)
    test_mask = torch.BoolTensor(test_mask).to(device)
    train_mask = torch.BoolTensor(train_mask).to(device)
    data_path = save_data_path+f'gnn_data_{thresh}.pth'
    # data_path = save_data_path+f'gnn_data.pth'
    if os.path.exists(data_path):
        print("load data")
        edge_idx = torch.load(data_path)
        
    else:
        for j in range(ivec_all.shape[0]):
            if j%1000==0:
                print(j)
            sim = []
            for i in range(ivec_all.shape[0]):
                
                cos = mat[j,i]/norm[i]/norm[j]
                sim.append(cos)
                if cos>thresh:
                    edge_idx.append([i,j])
            adj.append(sim)
        et = time.time()
        edge_idx = np.array(edge_idx)
        print("cost time : {}".format(et-st))
        edge_idx = torch.tensor(edge_idx, dtype = torch.long).to(device).t()
        torch.save( edge_idx,data_path)
    label_dict = {}
    id = 0
    for i, key in enumerate(np.unique(model_key)):
        label_dict[key] = i

    y = np.zeros(train_len+test_len)
    for i in range(train_len):
        y[i] = label_dict[model_key[i]]
        
    
    x = torch.FloatTensor(ivec_all).to(device)
    y = torch.LongTensor(y).to(device)

    data = SreData( edge_idx,x, y, train_mask, test_mask,x.shape[1], len(label_dict))
    data =data.to(device)
    return data

class Net(nn.Module):
    def __init__(self,num_node_features,num_classes):
        super(Net, self).__init__()
        self.conv_list = torch.nn.ModuleList()
        self.bn_list = torch.nn.ModuleList()
        self.conv_list.append(GATConv(num_node_features, 256, heads=1)) #GCNConv(num_node_features, 256))
        for _ in range(2):
            # self.conv_list.append(GCNConv(256, 256))
            self.conv_list.append(GATConv(64*4, 64,heads = 4))
            bn = BatchNorm1d(256)
            self.bn_list.append(bn)
        bn = BatchNorm1d(256)
        self.bn_list.append(bn)
        self.ln = Linear(256, num_classes)
    def forward(self, data):
        x, edge_index = data.x, data.edge_index.t()
        

        for i in range(len(self.conv_list)):
            x = self.conv_list[i](x, edge_index)
            x = self.bn_list[i](x)
        
        x = self.ln(x)
        x = F.softmax(x, dim=1)

        return x
    def extract_feature(self,data):
        x, edge_index = data.x, data.edge_index.t()
        

        for i in range(len(self.conv_list)):
            x = self.conv_list[i](x, edge_index)
            x = self.bn_list[i](x)       
        return x
    
def main(thresh=0.3):
    setup_seed(20)
    save_data_path ='./exp/'
    raw_data_path = '../data/'
    os.makedirs(save_data_path, exist_ok=True)
    sre14_baseline_train_lambda(raw_data_path, save_data_path)
    sre14_baseline_train_speaker(save_data_path)
    sre14_baseline_test(save_data_path)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    dataset = sre14_load_data(save_data_path,device, thresh)
    
    
    gnn = Net(dataset.num_node_features, dataset.num_class).to(device)

    import torch.optim as optim
    # transform = T.ToSparseTensor('edge_index')
    loader = [dataset]
    # loader = Load.NeighborLoader(
    #     dataset,
    #     # Sample 30 neighbors for each node for 2 iterations
    #     num_neighbors=[30] * 2,
    #     # Use a batch size of 128 for sampling training nodes
    #     batch_size=128,
    #     input_nodes=dataset.train_mask,
    # )
    # loader = DataLoader(dataset, batch_size=1, shuffle=True)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(gnn.parameters(), lr=0.0005, weight_decay=5e-4)
        
    for epoch in range(500):
        for data in loader:
           
 
            out = gnn(data)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])#/data.train_mask.sum()
            if (epoch+1) % 500 ==0:
                torch.save(gnn.state_dict(), save_data_path+f'gat_{epoch}_{thresh}.ckpt')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, pred = torch.max(out[data.train_mask], dim=1)
            correct = (pred == data.y[data.train_mask]).sum().item()
            acc = correct/data.train_mask.sum().item()

            print('Epoch {:03d} train_loss: {:.4f} train_acc: {:.4f}'.format(
                epoch, loss.item(), acc*10))

def test(thresh=0.3):
    save_data_path ='./exp/'
    ckpt_path =save_data_path+f'gat_499_{thresh}.ckpt'
    [ _ , model_key] = torch.load(save_data_path + "model_data.pth")
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    [_, test_key, _] = torch.load(save_data_path + "test_data.pth")
    dataset = sre14_load_data(save_data_path,device)
    gnn = Net(dataset.num_node_features, dataset.num_class).to(device)
    state_dict = torch.load(ckpt_path)
    gnn.load_state_dict(state_dict)
    print("=== cosine scoring\n")
    gnn.eval()

    feature = gnn.extract_feature(dataset)
    model_ivec = feature[dataset.train_mask].detach().cpu().numpy()
    test_ivec = feature[dataset.test_mask].detach().cpu().numpy()
    avg_model_ivec = np.zeros((len(np.unique(model_key)), model_ivec.shape[1]))
    for i, key in enumerate(np.unique(model_key)):
        avg_model_ivec[i] = np.mean(model_ivec[model_key == key], axis=0)

    model_ivec /= np.sqrt(np.sum(model_ivec ** 2, axis=1))[:, np.newaxis]
    test_ivec /= np.sqrt(np.sum(test_ivec ** 2, axis=1))[:, np.newaxis]
    avg_model_ivec /= np.sqrt(np.sum(avg_model_ivec ** 2, axis=1))[:, np.newaxis]
    score = np.dot(avg_model_ivec, test_ivec.T)
    score_col = score.flatten()
    
    print("=== evaluate score\n")
    [eer, mindcf_sre08, mindcf_sre10, mindcf_sre12, mindcf_sre14, mindcf_sre16] = compute_eer_mdcf(score_col, test_key)

    # print("user  :", getpass.getuser())
    # print("time  :", get_current_time())
    print("eer = {0:.2f} %".format(100 * eer))
    # print("mindcf_sre08 = {0:.4f}".format(mindcf_sre08))
    # print("mindcf_sre10 = {0:.4f}".format(mindcf_sre10))
    # print("mindcf_sre12 = {0:.4f}".format(mindcf_sre12))
    print("mindcf_sre14 = {0:.4f}".format(mindcf_sre14))
    # print("mindcf_sre16 = {0:.4f}".format(mindcf_sre16))

if __name__ == '__main__':
    main(0.3)
    
    test(0.3)
    
    
    