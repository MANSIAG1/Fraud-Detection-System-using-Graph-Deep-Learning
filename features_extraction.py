# This code is used to extracted node features from nx.graphs 
# and save them in csv files

import pandas as pd
import networkx as nx
import time

st = time.process_time()
graphs = ["/mnt/redpro/home/aid23001/Nx graphs semesters/WCC_semester_2011_S1.graphml","/mnt/redpro/home/aid23001/Nx graphs semesters/WCC_semester_2011_S2.graphml",
"/mnt/redpro/home/aid23001/Nx graphs semesters/WCC_semester_2012_S1.graphml","/mnt/redpro/home/aid23001/Nx graphs semesters/WCC_semester_2012_S2.graphml",
"/mnt/redpro/home/aid23001/Nx graphs semesters/WCC_semester_2013_S1.graphml"]
for graph in graphs:
	g = nx.read_graphml(graph)
	print('Reading Graph completed')
	features = pd.DataFrame()
	features['Node'] = [node for  node in g.nodes()]
	print('Nodes column completed')
	features['In-Degree'] = [g.in_degree(node) for node in g.nodes()]
	features['Out-Degree'] = [g.out_degree(node) for node in g.nodes()]
	print('In-Degree & Out-Degree column completed')
	features['Incoming Weight'] = [sum(d['amount'] for u, v, d in g.in_edges(node, data=True)) for node in g.nodes()]
	features['Outgoing Weight'] = [-sum(d['amount'] for u, v, d in g.out_edges(node, data=True)) for node in g.nodes()]
	print('Incoming and Outgoing Bitcoins column completed')
	pagerank_values = nx.pagerank(g,max_iter=150)
	features['PageRank'] = [pagerank_values[node] for node in features['Node']]

	egonet_edges = {}
	for node in g.nodes():
		if g.out_degree(node) == 0:
			egonet_edges[node] = 0
		else:
			egonet_edges[node] = nx.ego_graph(g,node, radius=3,undirected=False).number_of_edges()
	features['Egonet Edges'] = [egonet_edges[node] for node in features['Node']]
	print('Egonet edges completed')
	features['Egonet Weight'] = [sum(d['amount'] for u, v, d in nx.ego_graph(g, node, radius=3, undirected=False).edges(data=True)) for node in g.nodes()]
	print('Egonet weight column completed')
	malicious_dict = nx.get_node_attributes(g,'malicious')
	features['Malicious'] = [malicious_dict[node] for node in features['Node']]
	print('Feature dataframe completed')
	del g
et = time.process_time()
res = et - st
print('CPU Execution time:', res, 'seconds')
#features.to_csv('features_6.csv')
print('Saving features dataframe completed')