# This code is used to convert nx.graph to pyg.graph
# in order to be used in training the Graph Autoencoder 
# and produce the node embednings

import torch
import networkx as nx
from torch_geometric.utils import from_networkx
from datetime import datetime
import pandas as pd
import numpy as np


def datetime_to_numeric(graph):
    for u, v, attr in graph.edges(data=True):
        if 'datetime' in attr:  # 'datetime' is the name of the attribute
            datetime_str = attr['datetime']
            datetime_obj = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
            timestamp = datetime_obj.timestamp()
            attr['datetime'] = timestamp  # Changing the datetime attribute from str to numeric


def propagate_malicious(G):
    # Find all nodes that are initially malicious
    initial_malicious = [node for node, attr in G.nodes(data=True) if attr.get('malicious', False)]
    
    # Propagate maliciousness to neighbors
    for node in initial_malicious:
        for neighbor in G.neighbors(node):
            G.nodes[neighbor]['malicious'] = True  # Mark neighbor as malicious


print('Loading graphs...')
graph_1 = nx.read_graphml(
    '/mnt/redpro/home/aid23001/Nx graphs semesters/WCC_semester_2011_S1.graphml')
graph_2 = nx.read_graphml(
    '/mnt/redpro/home/aid23001/Nx graphs semesters/WCC_semester_2011_S2.graphml')
graph_3 = nx.read_graphml(
    '/mnt/redpro/home/aid23001/Nx graphs semesters/WCC_semester_2012_S1.graphml')
graph_4 = nx.read_graphml(
    '/mnt/redpro/home/aid23001/Nx graphs semesters/WCC_semester_2012_S2.graphml')
graph_5 = nx.read_graphml(
    '/mnt/redpro/home/aid23001/Nx graphs semesters/WCC_semester_2013_S1.graphml')
graph_6 = nx.read_graphml(
    '/mnt/redpro/home/aid23001/Nx graphs semesters/WCC_semester_2013_S2.graphml')
print('graphs are loaded')

graphs = [graph_1, graph_2, graph_3, graph_4, graph_5, graph_6]
for graph in graphs:
    datetime_to_numeric(graph)
    malicious_count = sum(1 for node, data in graph.nodes(data=True) if data.get("malicious"))
    nodes = graph.number_of_nodes()
    while malicious_count/nodes < 0.01:
        propagate_malicious(graph)
        malicious_count = sum(1 for node, data in graph.nodes(data=True) if data.get("malicious"))
    print(f'The new malicious rate is: {malicious_count/nodes}')
print('Converting nx graph to Torch Geometry')
pyg_graph_1 = from_networkx(graph_1, group_edge_attrs=['amount', 'datetime'])
pyg_graph_2 = from_networkx(graph_2, group_edge_attrs=['amount', 'datetime'])
pyg_graph_3 = from_networkx(graph_3, group_edge_attrs=['amount', 'datetime'])
pyg_graph_4 = from_networkx(graph_4, group_edge_attrs=['amount', 'datetime'])
pyg_graph_5 = from_networkx(graph_5, group_edge_attrs=['amount', 'datetime'])
pyg_graph_6 = from_networkx(graph_6, group_edge_attrs=['amount', 'datetime'])


del graph_1, graph_2, graph_3, graph_4, graph_5, graph_6, graphs
pyg_graphs = [pyg_graph_1, pyg_graph_2, pyg_graph_3, pyg_graph_4, pyg_graph_5, pyg_graph_6]
# True/False to ones and zeros
for pyg_graph in pyg_graphs:
    pyg_graph.y = pyg_graph.malicious.long()
print('Adding pre-calculated node features/attributes')
for i, pyg_graph in zip(range(1, 7), pyg_graphs):
    features = pd.read_csv(f'/mnt/redpro/home/aid23001/Features/features_{i}.csv')
    pyg_graph.x = torch.from_numpy(features.loc[:, 'In-Degree':'Egonet Weight'].to_numpy(dtype=np.float32))
    pyg_graph.x = torch.nn.functional.normalize(pyg_graph.x)
    torch.save(pyg_graph, f'/mnt/redpro/home/aid23001/Propagated_pyg_graphs/pyg_graph_propagated_{i}')

print('Done')