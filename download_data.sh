#!/bin/bash

wget http://snap.stanford.edu/jodie/wikipedia.csv -P data/
python utils/preprocess_data.py --data wikipedia --bipartite
