#*- coding:UTF-8 -*-
"""
##  ==========================================================================
##
##       author : Liang He, heliang@mail.tsinghua.edu.cn
##                Xianhong Chen, chenxianhong@mail.tsinghua.edu.cn
##   descrption : sre14 preprocess for SO/MO sGPLDA
##                This script is based on
##                NIST SRE14 offical released script (cosine scoring).
##      created : 20180206
## last revised : 20180511
##
##    Liang He, +86-13426228839, heliang@mail.tsinghua.edu.cn
##    Aurora Lab, Department of Electronic Engineering, Tsinghua University
##  ==========================================================================
"""

import numpy as np
import random
from compute_eer_and_mdcf import *

def find_neighbors_same_count(vecs_mean, vecs, vecs_not):
    """find nearest neighbors, same count

    Parameters
    ----------
    filename : 
        mean of vectors: target class, within class mean
        vectors: within class vectors
        between vectors: between class vectors

    Returns
    -------
    nearest neighbor vectors
    """    
    
    w_count = len(vecs)
    vecs_not_sim = [np.dot(vecs_mean, vecs_not[i]) for i in range(0,len(vecs_not))]
    vecs_not_label = np.argsort(vecs_not_sim)
    vecs_not_label = vecs_not_label[::-1]
    neighbors_vecs = [vecs_not[vecs_not_label[i]] for i in range(0,w_count)]
    
    return np.array(neighbors_vecs)

def find_random_select(vecs_not, num):
    """find readom vectors

    Parameters
    ----------
    filename : 
        between vectors
        number

    Returns
    -------
    random selection
    """    

    random_index = random.sample(range(0,len(vecs_not)), num)
    random_select = [vecs_not[random_index[i]] for i in range(0,num)]
    return np.array(random_select)

def compute_neighbor(vectors, labels):
    """compute between vectors

    Parameters
    ----------
    filename : 
        development and train vectors
        labels

    Returns
    -------
    between vectors
    """    

    if len(vectors) != len(labels):
        print ('len(vectors) != len(labels)')
        exit(-1)

    unique_labels = np.unique(labels)
    print (len(labels), len(unique_labels))        
        
    b_vectors = []
    b_labels = []
    
    for label in unique_labels:
        
        vecs = [vectors[i] for i in range(len(vectors)) if labels[i] == label]
        vecs_not = [vectors[i] for i in range(len(vectors)) if labels[i] != label]
        
        ## nearest selection
        vecs_mean = np.mean(vecs, axis=0)
        vecs_neighbors = find_neighbors_same_count(vecs_mean, vecs, vecs_not)
        
#        ## random selection
#        vecs_neighbors = find_random_select(vecs_not, len(vecs))
                
        if len(b_vectors) == 0:
            b_vectors = np.vstack((vecs, vecs_neighbors))
            b_labels = label*np.ones((len(vecs) + len(vecs_neighbors),1))
        else:
            b_vectors = np.vstack((b_vectors, np.vstack((vecs, vecs_neighbors))))
            b_labels = np.vstack((b_labels, label * np.ones((len(vecs) + len(vecs_neighbors),1))))
    
        print (len(vecs), len(vecs_neighbors), len(b_vectors), len(vecs_not))
        
    return b_vectors, b_labels

