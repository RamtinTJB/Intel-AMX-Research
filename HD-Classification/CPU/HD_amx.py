import numpy as np
import pickle
import sys
import os
import math
from utils import timer
from copy import deepcopy
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier as KNN

import torch


def binarize(base_matrix):
    return np.where(base_matrix < 0, -1, 1)

@timer
def encoding_rp_amx(X_data, base_matrix, signed=False):
    #A = torch.from_numpy(base_matrix).float()
    #X = torch.tensor(X_data, dtype=torch.float32)

    with torch.amp.autocast('cpu', dtype=torch.bfloat16):
        #Perform batch matrix multiplication: (n, d) x (d, m) = (n, m)
        hv = torch.matmul(X_data, base_matrix.T)

    hv_np = hv.cpu().float().numpy()
    if signed:
        hv_np = binarize(hv_np)

    return hv_np

@timer
def encoding_idlv_amx(X_data, lvl_hvs, id_hvs, D, bin_len, x_min, L=64):
#    X_data = torch.tensor(np.asarray(X_data), dtype=torch.bfloat16) # Doing this beforehand has significant performance boost
    #lvl_hvs = torch.tensor(lvl_hvs, dtype=torch.bfloat16)
    #id_hvs = torch.tensor(id_hvs, dtype=torch.bfloat16)

    bins = torch.floor((X_data - x_min) / bin_len)
    bins = torch.clamp(bins, 0, L-1).to(torch.int64)

    lvl_vectors = lvl_hvs[bins]

    with torch.amp.autocast('cpu', dtype=torch.bfloat16):
        #print(lvl_vectors.shape, id_hvs.shape)
        #bound =  lvl_vectors * id_hvs
        #print(bound.shape)
        #enc_hvs = bound.sum(dim=1)
        #print(enc_hvs.shape)
        #enc_hvs = lvl_vectors @ id_hvs.T
        enc_hvs = torch.einsum('ijk,jk->ik', lvl_vectors, id_hvs) # Figure out how this works

    return enc_hvs.cpu().float().numpy()

def encoding_perm(X_data, lvl_hvs, D, bin_len, x_min, L=64): # Not this one
    enc_hvs = []
    for i in range(len(X_data)):
        if i % int(len(X_data)/20) == 0:
            sys.stdout.write(str(int(i/len(X_data)*100)) + '% ')
            sys.stdout.flush()
        #sum_ = np.array([0] * D)
        sum_ = np.zeros((D))
        for j in range(len(X_data[i])):
            # bin_ = min( np.round((X_data[i][j] - x_min)/bin_len), L-1)
            bin_ = min( np.floor((X_data[i][j] - x_min)/bin_len), L-1)
            bin_ = int(bin_)
            sum_ += np.roll(lvl_hvs[bin_], j)
        enc_hvs.append(sum_)
    return enc_hvs

def max_match_amx(class_hvs, enc_hv, class_norms): # use qint8 to optimize this
    max_score = -np.inf
    max_index = -1
    for i in range(len(class_hvs)):
        score = np.matmul(class_hvs[i], enc_hv) / class_norms[i]
        #score = np.matmul(class_hvs[i], enc_hv)
        if score > max_score:
            max_score = score
            max_index = i
    return max_index

def max_match_amx_experimental(class_hvs, enc_hv, class_norms, scale=1.0, zero_point=0):
    """
    Computes the best matching class using a quantized dot product.

    Args:
      class_hvs (list or array): List of class hypervectors (each as a numpy array).
      enc_hv (numpy array): The encoded hypervector.
      class_norms (list): Pre-computed norms for the class hypervectors.
      scale (float): Quantization scale.
      zero_point (int): Quantization zero point.

    Returns:
      int: Index of the best matching class.
    """
    # Ensure enc_hv is a torch tensor in int8
    if not torch.is_tensor(enc_hv):
        enc_hv = torch.tensor(enc_hv, dtype=torch.int8)
    else:
        enc_hv = enc_hv.to(torch.int8)

    max_score = -float('inf')
    max_index = -1
    # Loop over class hypervectors (stored, for example, as a list of torch tensors)
    for i in range(len(class_hvs)):
        # Convert class_hvs[i] to a torch tensor if needed
        hv_i = class_hvs[i]
        if not torch.is_tensor(hv_i):
            hv_i = torch.tensor(hv_i, dtype=torch.int8)
        else:
            hv_i = hv_i.to(torch.int8)
        # Cast to int32 for the dot product (to avoid overflow)
        dot_product = torch.sum(hv_i.to(torch.int32) * enc_hv.to(torch.int32)).item()
        score = dot_product / class_norms[i]
        if score > max_score:
            max_score = score
            max_index = i
    return max_index

def train_amx(X_train, y_train, X_test, y_test, D=500, alg='rp', epoch=20, lr=1.0, L=64):

    #randomly select 20% of train data as validation
    permvar = np.arange(0, len(X_train))
    np.random.shuffle(permvar)
    X_train = [X_train[i] for i in permvar]
    y_train = [y_train[i] for i in permvar]
    cnt_vld = int(0.2 * len(X_train))
    X_validation = X_train[0:cnt_vld]
    y_validation = y_train[0:cnt_vld]
    X_train = X_train[cnt_vld:]
    y_train = y_train[cnt_vld:]

    # Convert inputs into bf16 
    X_train = torch.tensor(np.array(X_train), dtype=torch.bfloat16)
    X_validation = torch.tensor(np.array(X_validation), dtype=torch.bfloat16)
    X_test = torch.tensor(np.array(X_test), dtype=torch.bfloat16)

    #encodings
    if alg in ['rp', 'rp-sign']:
        #create base matrix
        base_matrix = ((torch.rand(D, len(X_train[0])) > 0.5) * 2 - 1).to(torch.bfloat16)
        print('\nEncoding ' + str(len(X_train)) + ' train data')
        train_enc_hvs = encoding_rp_amx(X_train, base_matrix, signed=(alg == 'rp-sign'))
        print('\n\nEncoding ' + str(len(X_validation)) + ' validation data')
        validation_enc_hvs = encoding_rp_amx(X_validation, base_matrix, signed=(alg == 'rp-sign'))

    elif alg in ['idlv', 'perm']:
        #create level matrix
        #lvl_hvs = []
        #temp = [-1]*int(D/2) + [1]*int(D/2)
        #np.random.shuffle(temp)
        #lvl_hvs.append(temp)
        #change_list = np.arange(0, D)
        #np.random.shuffle(change_list)
        #cnt_toChange = int(D/2 / (L-1))
        #for i in range(1, L):
            #temp = np.array(lvl_hvs[i-1])
            #temp[change_list[(i-1)*cnt_toChange : i*cnt_toChange]] = -temp[change_list[(i-1)*cnt_toChange : i*cnt_toChange]]
            #lvl_hvs.append(list(temp))
        #lvl_hvs = np.array(lvl_hvs, dtype=np.int8)
        #x_min = min( np.min(X_train), np.min(X_validation) )
        #x_max = max( np.max(X_train), np.max(X_validation) )
        #bin_len = (x_max - x_min)/float(L)
        D_half = D // 2

        # Create initial vector with half -1 and half 1 (as bfloat16 floats)
        neg_part = torch.full((D_half,), -1, dtype=torch.bfloat16)
        pos_part = torch.full((D - D_half,), 1, dtype=torch.bfloat16)
        temp = torch.cat((neg_part, pos_part))
#
        ## Shuffle the vector using torch.randperm
        perm_indices = torch.randperm(D)
        temp = temp[perm_indices]
        lvl_hvs = [temp]
#
        ## Create a shuffled list of indices to use for flipping
        change_list = torch.randperm(D)
        cnt_toChange = D_half // (L - 1)
#
        ## Generate additional levels by flipping the sign at selected indices
        for i in range(1, L):
            prev_vec = lvl_hvs[i - 1].clone()
            indices = change_list[(i - 1) * cnt_toChange : i * cnt_toChange]
            prev_vec[indices] = -prev_vec[indices]
            lvl_hvs.append(prev_vec)
###
        # Stack the level hypervectors into a tensor of shape (L, D)
        lvl_hvs = torch.stack(lvl_hvs)  # dtype remains torch.bfloat16

        # Compute x_min and x_max entirely in torch
        x_min = torch.min(torch.cat((X_train.flatten(), X_validation.flatten())))
        x_max = torch.max(torch.cat((X_train.flatten(), X_validation.flatten())))

        # Compute the bin length
        bin_len = (x_max - x_min) / L


        #need to create id hypervectors if encoding is level-id
        if alg == 'idlv':
            cnt_id = len(X_train[0])
            id_hvs = []
            for i in range(cnt_id):
                temp = [-1]*int(D/2) + [1]*int(D/2)
                np.random.shuffle(temp)
                id_hvs.append(temp)
            #id_hvs = np.array(id_hvs, dtype=np.int8)
            id_hvs = torch.tensor(id_hvs, dtype=torch.bfloat16)
            print('\nEncoding ' + str(len(X_train)) + ' train data')
            train_enc_hvs = encoding_idlv_amx(X_train, lvl_hvs, id_hvs, D, bin_len, x_min, L)
            print('\n\nEncoding ' + str(len(X_validation)) + ' validation data')
            validation_enc_hvs = encoding_idlv_amx(X_validation, lvl_hvs, id_hvs, D, bin_len, x_min, L)
        elif alg == 'perm':
            print('\nEncoding ' + str(len(X_train)) + ' train data')
            train_enc_hvs = encoding_perm(X_train, lvl_hvs, D, bin_len, x_min, L)
            print('\n\nEncoding ' + str(len(X_validation)) + ' validation data')
            validation_enc_hvs = encoding_perm(X_validation, lvl_hvs, D, bin_len, x_min, L)

    #training, initial model
    class_hvs = [[0.] * D] * (max(y_train) + 1)
    for i in range(len(train_enc_hvs)):
        class_hvs[y_train[i]] += train_enc_hvs[i]
    class_norms = [np.linalg.norm(hv) for hv in class_hvs]
    class_hvs_best = deepcopy(class_hvs)
    class_norms_best = deepcopy(class_norms)
    #retraining
    if epoch > 0:
        acc_max = -np.inf
        print('\n\n' + str(epoch) + ' retraining epochs')
        for i in range(epoch):
            sys.stdout.write('epoch ' + str(i) + ': ')
            sys.stdout.flush()
            #shuffle data during retraining
            pickList = np.arange(0, len(train_enc_hvs))
            np.random.shuffle(pickList)
            for j in pickList:
                predict = max_match_amx(class_hvs, train_enc_hvs[j], class_norms)
                if predict != y_train[j]:
                    class_hvs[predict] -= np.multiply(lr, train_enc_hvs[j])
                    class_hvs[y_train[j]] += np.multiply(lr, train_enc_hvs[j])
            class_norms = [np.linalg.norm(hv) for hv in class_hvs]
            correct = 0
            for j in range(len(validation_enc_hvs)):
                predict = max_match_amx(class_hvs, validation_enc_hvs[j], class_norms)
                if predict == y_validation[j]:
                    correct += 1
            acc = float(correct)/len(validation_enc_hvs)
            sys.stdout.write("%.4f " %acc)
            sys.stdout.flush()
            if i > 0 and i%5 == 0:
                print('')
            if acc > acc_max:
                acc_max = acc
                class_hvs_best = deepcopy(class_hvs)
                class_norms_best = deepcopy(class_norms)

    del X_train
    del X_validation
    del train_enc_hvs
    del validation_enc_hvs

    print('\n\nEncoding ' + str(len(X_test)) + ' test data')
    if alg == 'rp' or alg == 'rp-sign':
        test_enc_hvs = encoding_rp_amx(X_test, base_matrix, signed=(alg == 'rp-sign'))
    elif alg == 'idlv':
        test_enc_hvs = encoding_idlv_amx(X_test, lvl_hvs, id_hvs, D, bin_len, x_min, L)
    elif alg == 'perm':
        test_enc_hvs = encoding_perm(X_test, lvl_hvs, D, bin_len, x_min, L)
    correct = 0
    for i in range(len(test_enc_hvs)):
        predict = max_match_amx(class_hvs_best, test_enc_hvs[i], class_norms_best)
        if predict == y_test[i]:
            correct += 1
    acc = float(correct)/len(test_enc_hvs)
    return acc
