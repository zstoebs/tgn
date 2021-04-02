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
#node_graph = None

def get_timestamps():
	ts = []
	for n1,n2,data in edge_graph.edges(data=True):
		ts += [int(data['timestamp'])]
#	for n1,n2,data in node_graph.edges(data=True):
#		ts += [int(data['timestamp'])]
	ts.sort()
	print('Number of timestamps: ', len(ts))
	print('Earliest timestamp: ', ts[0])
	print('Latest timestamp: ', ts[-1])
	return ts

def read_json_graph(fname):
	with open(filename,'r') as f:
		js_graph = json.load(f)
	return json_graph.node_link_graph(js_graph)

def get_graph_at_timestamp(timestamp):

	node_bunch = []
	for n1,n2,data in edge_graph.edges(data=True):
		keep = 'pos_prob' in data.keys()
		node_bunch += [n1,n2] if keep and int(data['timestamp']) <= timestamp else []
	return nx.subgraph(edge_graph,node_bunch)

@app.route('/all_timestamps',methods=['GET'])
def all_timestamps():
	return flask.jsonify(timestamps)

@app.route('/get_timeframe',methods=['GET','POST'])
def get_timeframe():
	endtime = request.get_json()

	tf = timestamps[:endtime]
	subgraph = get_graph_at_timestamp(tf[-1])	
	nodes = list(subgraph.nodes)
	edges = list(subgraph.edges)
	probs = [G[s][t]['pos_prob'] for s,t in edges]

	return flask.jsonify({'timeframe':tf,'nodes':nodes,'edges':edges,'probs':probs})

if __name__=='__main__':
	edge_graph = read_json_graph('static/edge/edge_prediction.json')
#	node_graph = read_json_graph('static/node/node_classification.json')
	timestamps = get_timestamps()	

	app.run()
