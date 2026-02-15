# This code is used to create the embeding space and save it in a numpy array 

import torch
from torch_geometric.nn import GCNConv, GAE
import numpy as np
import time

st = time.process_time()

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels, cached=False)
        self.conv2 = GCNConv(out_channels,2 * out_channels, cached=False)
        self.conv3 = GCNConv(2 * out_channels, out_channels,cached=False)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight).relu()
        return self.conv3(x, edge_index, edge_weight)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('The used device is', device)
model = GAE(GCNEncoder(7, 15))
model.load_state_dict(torch.load('/mnt/redpro/home/aid23001/GAE_model_3_15.pth'))

for i in range(1, 7):
    pyg_graph = torch.load(f"/mnt/redpro/home/aid23001/Pyg Graphs/pyg_graph_{i}")
    x = pyg_graph.x
    edge_index = pyg_graph.edge_index
    edge_weight = pyg_graph.edge_attr[:, :1]
    encode = model.encode(x, edge_index, edge_weight)
    np.save(f'/mnt/redpro/home/aid23001/Numpy Arrays 2/Z_{i}', encode.detach().numpy())
    np.save(f'/mnt/redpro/home/aid23001/Numpy Arrays 2/Y_{i}', pyg_graph.y.numpy())

et = time.process_time()
res = et - st
print('CPU Execution time:', res, 'seconds')

print('Numpy arrays and labels are created')