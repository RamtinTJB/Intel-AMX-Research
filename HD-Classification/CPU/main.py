
#python main.py --path isolet.pickle --d 1000 --alg rp --epoch 20
#python main.py --path isolet.pickle --d 1000 --alg rp-sign --epoch 20
#python main.py --path isolet.pickle --d 1000 --alg idlv --epoch 20 --L 64 
#python main.py --path isolet.pickle --d 1000 --alg perm --epoch 20 --L 64 

import HD
import numpy as np
import sys
import time
from copy import deepcopy
import argparse
import pickle
from sklearn import preprocessing

def adjust_data_size(X, y, new_size):
    current_size = len(X)
    if new_size == current_size:
        return X, y
    elif new_size < current_size:
        indices = np.random.choice(current_size, new_size, replace=False)
        return [X[i] for i in indices], [y[i] for i in indices]
    else:
        indices = np.random.choice(current_size, new_size, replace=True)
        return [X[i] for i in indices], [y[i] for i in indices]


parser = argparse.ArgumentParser()
parser.add_argument('--path', action='store', type=str, help='path to pickle dataset', required=True)
parser.add_argument('--d', action='store', dest='D', type=int, default=500, help='number of dimensions (default 500)')
parser.add_argument('--alg', action='store', type=str, default='rp', help='encoding technique (rp, rp-sign, idlv, perm')
parser.add_argument('--epoch', action='store', type=int, default=20, help='number of retraining iterations (default 20)')
parser.add_argument('--lr', '-lr', action='store', type=float, default=1.0, help='learning rate (default 1.0)')
parser.add_argument('--L', action='store', type=int, default=64, help='number of levels (default 64)')
parser.add_argument('--amx', action='store_true', help='Use AMX accelerators for the matrix multiplications')
parser.add_argument('--size', action='store', type=int, default=None, help='desired input data size')
# Add cmd argument for the dataset size
# dimensions 512 - 16k
# choose between int8 (similarity checking, result should not be int8) and bf16 

inputs = parser.parse_args()
path = inputs.path
D = inputs.D
alg = inputs.alg
epoch = inputs.epoch
lr = inputs.lr
L = inputs.L
amx = inputs.amx
data_size = inputs.size

assert alg in ['rp', 'rp-sign', 'idlv', 'perm']

with open(path, 'rb') as f:
	dataset = pickle.load(f, encoding='latin1')	

X_train, y_train, X_test, y_test = deepcopy(dataset)

if data_size is not None:
    X_train, y_train = adjust_data_size(X_train, y_train, data_size)
    X_test, y_test = adjust_data_size(X_test, y_test, data_size)

acc = HD.train(X_train, y_train, X_test, y_test, D, alg, epoch, lr, L, amx)
print('\n')
print(acc)
