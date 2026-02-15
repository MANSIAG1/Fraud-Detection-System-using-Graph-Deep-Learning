#This code is used to create, train and save the Graph Autoencoder

import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.transforms import RandomLinkSplit
import numpy as np
import time 

st = time.process_time()

class GCNEncoder(torch.nn.Module):
     def __init__(self, in_channels, out_channels):
         super(GCNEncoder, self).__init__()
         self.conv1 = GCNConv(in_channels, out_channels, cached=False)
         self.conv2 = GCNConv(out_channels, 2 * out_channels, cached=False)
         self.conv3 = GCNConv(2 * out_channels, out_channels, cached=False)

     def forward(self, x, edge_index, edge_weight):
         x = self.conv1(x, edge_index, edge_weight).relu()
         x = self.conv2(x, edge_index, edge_weight).relu()
         return self.conv3(x, edge_index, edge_weight)

out_channels = 20
num_features = 7
epochs = 150

# model
model = GAE(GCNEncoder(num_features, out_channels))
# move to GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu' #not enough RAM on the GPU
print('The used device is', device)
model = model.to(device)
# x = data.x.to(device)
# train_pos_edge_index = data.pos_edge_label_index.to(device)           
# edge_weight = data.edge_attr[:, :1].to(device)
# inizialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train(data):
    x, train_pos_edge_index, edge_weight = data.x.to(device), data.pos_edge_label_index.to(device), data.edge_attr[:, :1].to(device)
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index, edge_weight)  # x, train_pos_edge_index, edge_weight already stated
    loss = model.recon_loss(z, train_pos_edge_index)      
    loss.backward()
    optimizer.step()
    return float(loss)


def test(data, pos_edge_index, neg_edge_index):
    x = data.x.to(device)
    train_pos_edge_index = data.pos_edge_label_index.to(device)  
    edge_weight = data.edge_attr[:, :1].to(device)
    model.eval()
    with torch.no_grad():
        z = model(x, train_pos_edge_index, edge_weight)
    return model.test(z, pos_edge_index, neg_edge_index)


transform = RandomLinkSplit(num_val=0, num_test=0, is_undirected=False, split_labels=True, add_negative_train_samples=True)
# pyg_graph_test = torch.load('/mnt/redpro/home/aid23001/Pyg Graphs/pyg_graph_6')
# data_test = transform(pyg_graph_test)[0]

# training on the known graph, testing on the last
for i in range(1, 6):
    print(f'for graph, {i}')
    pyg_graph = torch.load(f"/mnt/redpro/home/aid23001/Pyg Graphs/pyg_graph_{i}")
    data = transform(pyg_graph)[0]
    for epoch in range(1, epochs+1):
        loss = train(data)
        # auc, ap = test(data_test, data_test.pos_edge_label_index, data_test.neg_edge_label_index)
        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Loss: {loss:.04f}') #, AUC: {auc:.04f}, AP:{ap:.04f}')

et = time.process_time()
res = et - st
print('CPU Execution time:', res, 'seconds')
#print('Saving the trained model'),torch.save(model.state_dict(), '/mnt/redpro/home/aid23001/Pyg Graphs/GAE_GINConv.pth')  # Saving the model