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

if __name__=='__main__':
    painting_image_urls = json.load(open('painting_image_urls.json','r'))
    attribute_names = json.load(open('attribute_names.json','r'))
    episode_names = json.load(open('episode_names.json','r'))
    painting_attributes = np.load('painting_attributes.npy')

    app.run()
#
