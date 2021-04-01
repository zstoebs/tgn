import os
import flask
import numpy as np
import argparse
import json
import csv

from flask import Flask
from flask import request
from flask_cors import CORS
import math

from sklearn.cluster import KMeans

# create Flask app
app = Flask(__name__)
CORS(app)

# --- these will be populated in the main --- #

# list of attribute names of size m
attribute_names=None

# a 2D numpy array containing binary attributes - it is of size n x m, for n paintings and m attributes
painting_attributes=None

# a list of epsiode names of size n
episode_names=None

# a list of painting image URLs of size n
painting_image_urls=None

'''
This will return an array of strings containing the episode names -> these should be displayed upon hovering over circles.
'''
@app.route('/get_episode_names', methods=['GET'])
def get_episode_names():
    return flask.jsonify(episode_names)
#

'''
This will return an array of URLs containing the paths of images for the paintings
'''
@app.route('/get_painting_urls', methods=['GET'])
def get_painting_urls():
    return flask.jsonify(painting_image_urls)
#

'''
TODO: implement PCA, this should return data in the same format as you saw in the first part of the assignment:
    * the 2D projection
    * x loadings, consisting of pairs of attribute name and value
    * y loadings, consisting of pairs of attribute name and value
'''
@app.route('/initial_pca', methods=['GET'])
def initial_pca():
	E = painting_attributes
	E_centered = (E - np.mean(E,axis=0)).T
	
	# compute covariance matrix and its eigenvectors/values
	Ce = np.cov(E_centered)
	u,v = np.linalg.eigh(Ce)
	inds = u.argsort()[::-1]
	u = u[inds].real
	v = v[:,inds].real
	
	# project to 2 dims
	comps = v[:,:2]
	proj = E_centered.T @ comps
	x = (u[0]*comps[:,0]).tolist()
	y = (u[1]*comps[:,1]).tolist()

	# format and send 
	x_loading = []
	y_loading = []
	for i,tup in enumerate(zip(x,y)):
		ux,uy = tup
		attr = attribute_names[i]
		x_loading += [{'attribute': attr, 'loading': ux}]
		y_loading += [{'attribute': attr, 'loading': uy}]	
		
	return flask.jsonify({'projection': proj.tolist(), 'x_loading': x_loading, 'y_loading': y_loading})
#

'''
TODO: implement ccPCA here. This should return data in _the same format_ as initial_pca above.
It will take in a list of data items, corresponding to the set of items selected in the visualization. This can be acquired from `flask.request.json`. This should be a list of data item indices - the **target set**.
The alpha value, from the paper, should be set to 1.1 to start, though you are free to adjust this parameter.
'''
@app.route('/ccpca', methods=['GET','POST'])
def ccpca():
	inds = request.get_json()
	E = painting_attributes
	mean = np.mean(E,axis=0) # assume same centering effects
	E_centered = (E - mean).T

	# compute selected covariance [cPCA]
	K = np.array(E[inds,:])
	K_centered = (K - mean).T
	Ck = np.cov(K_centered)

	# compute entire covariance [ccPCA]
	Ce = np.cov(E_centered)

	# compute remainder covariance
	R = np.delete(E,inds,axis=0)
	R_centered = (R - mean).T
	Cr = np.cov(R_centered)

	# compute the contrast covariance
	C = Ce - 1.3*Cr

	# compute eigens and project to full dataset
	u,v = np.linalg.eigh(C)
	inds = u.argsort()[::-1]
	u = u[inds].real
	v = v[:,inds].real
	comps = v[:,:2]
	proj = E_centered.T @ comps
	x = (u[0]*comps[:,0]).tolist()
	y = (u[1]*comps[:,1]).tolist()

	# format output and send 
	x_loading = []
	y_loading = []
	for i,tup in enumerate(zip(x,y)):
		ux,uy = tup
		attr = attribute_names[i]
		x_loading += [{'attribute': attr, 'loading': ux}]
		y_loading += [{'attribute': attr, 'loading': uy}]	

	return flask.jsonify({'projection': proj.tolist(), 'x_loading': x_loading, 'y_loading': y_loading})

#
'''
TODO: run kmeans on painting_attributes, returning data in the same format as in the first part of the assignment. Namely, an array of objects containing the following properties:
    * label - the cluster label
    * id: the data item's id, simply its index
    * attribute: the attribute name
    * value: the binary attribute's value
'''
@app.route('/kmeans', methods=['GET'])
def kmeans():
	# get cluster labels
	kmeans = KMeans(n_clusters=6)
	kmeans.fit(painting_attributes)
	labels = kmeans.labels_
	
	# format output and send 
	data = []
	for ID,row in enumerate(painting_attributes):
		for i,value in enumerate(row):
			attr = attribute_names[i] 
			label = int(labels[ID]) 
			data += [{'attribute': attr,'id': ID, 'label': label, 'value': int(value)}] 
	
	return flask.jsonify(data)		
#

if __name__=='__main__':
    painting_image_urls = json.load(open('painting_image_urls.json','r'))
    attribute_names = json.load(open('attribute_names.json','r'))
    episode_names = json.load(open('episode_names.json','r'))
    painting_attributes = np.load('painting_attributes.npy')

    app.run()
#
