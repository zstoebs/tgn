import os
import flask
import numpy as np
import argparse
import json
import csv
import pickle

import networkx as nx
from networkx.readwrite import json_graph

from flask import Flask
from flask import request
from flask_cors import CORS
import math

from sklearn.cluster import KMeans

# create Flask app
app = Flask(__name__)
CORS(app)

timestamps = None	
edge_graph = None
node_graph = None

def extract_timestamps_from_graph(G):
	ts = []
	for n1,n2,data in G.edges(data=True):
		ts += [int(data['timestamp'])]
	ts.sort()
	print('Number of timestamps: ', len(ts))
	print('Earliest timestamp: ', ts[0])
	print('Latest timestamp: ', ts[-1])
	return ts

def read_json_graph(fname):
	with open(fname,'r') as f:
		js_graph = json.load(f)
	g = json_graph.node_link_graph(js_graph)

	# remove negative edges --> not informative
	edges = list(g.edges)
	for s,t in edges:
		if 'neg_prob' in g[s][t].keys():
			g.remove_edge(s, t)
	return g


def get_graph_at_timestamp(G,timestamp):

	node_bunch = set()
	for n1,n2,data in G.edges(data=True):
		if int(data['timestamp']) <= timestamp:
			node_bunch.add(n1)
			node_bunch.add(n2)
	return nx.subgraph(G,node_bunch)

def parse_graph(subgraph):
	nodes = list(subgraph.nodes)
	node_info = [{'id': node} for node in nodes]
	edges = list(subgraph.edges)
	edge_info = []
	for s, t in edges:
		info = {}
		check = 'pos_prob' in subgraph[s][t].keys()
		prob = subgraph[s][t]['pos_prob'] if check else subgraph[s][t]['neg_prob']
		gt = 1 if check else 0

		info['source'] = s
		info['target'] = t
		# info['ground_truth'] = gt
		info['prob'] = prob[0]
		info['timestamp'] = int(subgraph[s][t]['timestamp'])
		info['source_embed'] = subgraph[s][t]['source_embed']
		info['dest_embed'] = subgraph[s][t]['dest_embed']
		edge_info += [info]

	return node_info, edge_info

def compose_full_graph():

	count = 0
	for s, t in list(edge_graph.edges):
		try:
			edge_graph[s][t]['source_embed'] = node_graph[s][t]['source_embed']
			edge_graph[s][t]['dest_embed'] = node_graph[s][t]['dest_embed']
		except BaseException:
			count += 1
			edge_graph[s][t]['source_embed'] = None
			edge_graph[s][t]['dest_embed'] = None

	print('Num edges without context: ', count)


@app.route('/get_timestamps',methods=['GET'])
def get_timestamps():
	return flask.jsonify(timestamps)

@app.route('/get_subgraph_by_node_id',methods=['GET','POST'])
def get_subgraph_by_node_id():
	ids = request.get_json()
	node_bunch = set()
	for i in ids:
		id = int(i)
		node_bunch.add(id)
		for n in edge_graph.neighbors(id):
			node_bunch.add(n)
			
	subgraph = nx.subgraph(edge_graph, node_bunch)

	node_info, edge_info = parse_graph(subgraph)

	return flask.jsonify({'nodes': node_info, 'edges': edge_info})

@app.route('/get_timeframe',methods=['GET','POST'])
def get_timeframe():
	end = request.get_json()

	tf = timestamps[:end+1]
	subgraph = get_graph_at_timestamp(edge_graph,tf[-1]) # HARDCODED	
	node_info, edge_info = parse_graph(subgraph)

	return flask.jsonify({'nodes':node_info,'edges':edge_info})

@app.route('/get_graph',methods=['GET'])
def get_graph():
	node_info, edge_info = parse_graph(edge_graph)
	return flask.jsonify({'nodes':node_info,'edges':edge_info}) 

if __name__=='__main__':
	edge_graph = read_json_graph('static/edge/edge_prediction.json')
	node_graph = read_json_graph('static/node/node_classification.json')

	compose_full_graph()
	timestamps = extract_timestamps_from_graph(edge_graph)	

	app.run(host='localhost')