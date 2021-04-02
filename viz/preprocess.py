import os
import json
import networkx as nx
from networkx.readwrite import json_graph
from TemporalGraph import TemporalGraph
import argparse


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='jsonify a temporal graph')
	parser.add_argument('--inpath','-i',type=str,default='static/edge/graph',help='path to pickle files containing edge prediction information')
	parser.add_argument('--outpath','-o',type=str,default='static/edge',help='path to output JSON graph')
	parser.add_argument('--name','-n',type=str,default='',help='name of graph')
	parser.add_argument('--build_type','-b',type=str,default='edge',help='graph build type, corresponds to task {edge prediction, node classification}')
	args = parser.parse_args()

	legend={'source':'sources_batch','destination':'destinations_batch','timestamp':'timestamps_batch','edge_idx':'edge_idxs_batch','n_neighbor':'n_neighbors'}
	if args.build_type == 'edge':
		attrs={'pos_prob':'pos_prob','neg_prob':'neg_prob','negative':'negative_samples'}	
	elif args.build_type == 'node':
		attrs={'dest_embed':'destination_embedding','source_embed':'source_embedding','pred_prob':'pred_prob_batch'}	
	else:
		raise argparse.ArgumentTypeError('Unknown build type. Supported types are \'edge\' and \'node\'.')

	graph = TemporalGraph(path=args.inpath,name=args.name,build_type=args.build_type,legend=legend,attrs=attrs)
	graph.construct()
	graph_json = json_graph.node_link_data(graph.graph)
	json.dump(graph_json,open(os.path.join(args.outpath,args.name+'.json'),'w'))
