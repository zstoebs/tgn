#!/bin/bash

python train_self_supervised.py --use_memory --prefix tgn-attn --n_runs 10
python train_supervised.py --use_memory --prefix tgn-attn --n_runs 10 --use_validation
