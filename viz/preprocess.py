import os
import numpy as np
import networkx as nx
import pickle


if __name__ == '__main__':
	predicted = nx.Graph(name='predicted')
	truth = nx.Graph(name='ground_truth')
	
