import os
import numpy as np
import networkx as nx
import pickle

class TemporalGraph:

	def __init__(self,path,name,build_type='edge',legend={'source':'sources_batch','destination':'destinations_batch','timestamp':'timestamps_batch','edge_idx':'edge_idxs_batch','n_neighbor':'n_neighbors'},attrs={'pos_prob':'pos_prob','neg_prob':'neg_prob','negative':'negative_samples'}):

		assert build_type == 'edge' or build_type == 'node'

		self.path = path
		self.name = name
		self.build_type = build_type
		self.legend = legend
		self.attrs = attrs

		self.graph = nx.Graph(name=name)

	def construct(self):
		for fname in os.listdir(self.path):
			if not '.pkl' in fname:
				continue
			print('Constructing: ', fname)

			d = pickle.load(open(os.path.join(self.path,fname),'rb'))
			for key in d.keys():
				try:
					d[key] = np.concatenate(d[key],axis=0).tolist()
				except ValueError:
					continue

			for i,tup in enumerate(zip(d[self.legend['source']],d[self.legend['destination']])):
				source,dest = tup
				ts = d[self.legend['timestamp']][i]
				edge_idx = d[self.legend['edge_idx']][i]

				self.graph.add_edge(source,dest,timestamp=ts,idx=edge_idx)

				for attr in self.attrs.keys():
					#print(i,len(d[self.attrs[attr]]))
					self.graph[source][dest][attr] = d[self.attrs[attr]][i]

				# if self.build_type == 'edge':
				# 	self.graph[source][dest]['pos_prob'] = d[self.attrs['pos_prob']][i]
				#
				# 	neg = d[self.attrs['negative']][i]
				# 	self.graph[source][neg]['neg_prob'] = d[self.attrs['neg_prob']][i]
				#
				# elif self.build_type == 'node':
				#
				# 	self.graph[source][dest]['dest_embed'] = d[self.attrs['dest_embed']][i]
				# 	self.graph.nodes[source]['source_embed'] = d[self.attrs['source_embed']][i]
				# 	self.graph.nodes[source]['pred_prob'] = d[self.attrs['pred_prob']][i]
				#
                    
	def at_time(self,timestamp):
		pass			
