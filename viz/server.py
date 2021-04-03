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
	return json_graph.node_link_graph(js_graph)

def get_graph_at_timestamp(G,timestamp):

	node_bunch = set()
	for n1,n2,data in G.edges(data=True):
		if int(data['timestamp']) <= timestamp:
			node_bunch.add(n1)
			node_bunch.add(n2)
	return nx.subgraph(G,node_bunch)

@app.route('/get_timestamps',methods=['GET'])
def get_timestamps():
	return flask.jsonify(timestamps)

@app.route('/get_timeframe',methods=['GET','POST'])
def get_timeframe():
	print('here')
	endtime = request.get_json()

	tf = timestamps[:endtime]
	subgraph = get_graph_at_timestamp(edge_graph,tf[-1]) # HARDCODED	
	nodes = list(subgraph.nodes)
	edges = list(subgraph.edges)
	edge_info = []
	for s,t in edges:
		info = {}
		check = 'pos_prob' in subgraph[s][t].keys()
		prob = subgraph[s][t]['pos_prob'] if check else subgraph[s][t]['neg_prob']
		gt = 1 if check else 0
		
		info['edge'] = [s,t]
		info['ground_truth'] = gt
		info['prob'] = prob
		edge_info += [info]

	return flask.jsonify({'timeframe':tf,'nodes':nodes,'edges':edge_info})

if __name__=='__main__':
	edge_graph = read_json_graph('static/edge/edge_prediction.json')
	timestamps = extract_timestamps_from_graph(edge_graph)	

	app.run(host='localhost')
	
