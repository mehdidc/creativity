#!/bin/sh
python /home/gridcl/mehdicherti/work/code/pylearn2/pylearn2/scripts/tutorials/rbm_autoencoder/deep_autoencoder_train.py dataset.yaml auto.yaml layer.yaml sigmoid '!import pylearn2.expr.activations.relu' 256 50 20

