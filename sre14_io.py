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

import csv
import numpy as np

def load_model_key(filename, model_ids):
    """Loads a model key to match order of model_ids
    from
    Parameters
    ----------
    filename : stringb
        Path to target_speaker_models.csv
    model_ids : list
        List of model ivectorids from model_ivectors.csv

    Returns
    -------
    y : array, shaped('n_model_ivecs',)
        Array with each entry the target_speaker_modelid, in
        same order as model_ids
    """
    
    # load a reverse lookup from ivectorid->target speaker model
    id_speaker_lookup = {}
    for row in csv.DictReader(open(filename, 'r')):
        for e in row['model_ivectorids[5]'].split():
            id_speaker_lookup[e] = row['target_speaker_modelid']

    # convert to a simple key array for model ivectors
    y = []
    for model_id in model_ids:
        y.append( id_speaker_lookup[ model_id ] )

    return np.array(y)
	
	
def load_dev_key(dev_lable, dev_ids):
    """Loads development data label to match order of dev_ivectors' ivector ids

    Parameters
    ----------
    dev_lable : string
        Path to development_data_labels.csv
     dev_ids : list
        List of dev_ivectors from dev_ivectors.csv

    Returns
    -------
    y : array, shaped('n_dev_ivectors',)
        Array with each entry the development data speaker_id, in
        same order as dev_ids
    """
    
    # load a reverse lookup from ivectorid->target speaker model
    id_speaker_lookup = {}
    id_speaker_dict = {}

    for row in csv.DictReader(open(dev_lable, 'r')):
        id_speaker_lookup[row['ivectorid']] = row['speaker_id']

        if row['speaker_id'] in id_speaker_dict:
            id_speaker_dict[row['speaker_id']].append(row['ivectorid'])
        else:
            id_speaker_dict[row['speaker_id']] = []
            id_speaker_dict[row['speaker_id']].append(row['ivectorid'])

    # convert to a simple key array for dev ivectors
    y = []
    for dev_id in dev_ids:
        y.append( id_speaker_lookup[ dev_id ] )

    return [np.array(y), id_speaker_lookup, id_speaker_dict]


def load_ivectors(filename):
    """Loads ivectors

    Parameters
    ----------
    filename : string
        Path to ivector files (e.g. dev_ivectors.csv)

    Returns
    -------
    ids : list
        List of ivectorids
    durations : array, shaped('n_ivectors')
        Array of durations for each ivectorid
    ivectors : array, shaped('n_ivectors', 600)
        Array of ivectors for each ivectorid
    """
    
    ids = []
    durations = []
    ivectors = []

    for row in csv.DictReader(open(filename, 'r')):
        ids.append( row['ivectorid'] )
        durations.append( float(row['duration_secs']) )
        ivectors.append( np.fromstring(row['values[600]'], count=600, sep=' ', dtype=np.float32) )

    return ids, np.array(durations, dtype=np.float32), np.vstack( ivectors )
    
    
def load_test_key(trial_key_file):
    """Loads test data keys

    Parameters
    ----------
    trial_key_file : string
        Path to trial_key files (e.g. ivec14_sre_trial_key_release.tsv)

    Returns
    -------
    key : array
        a vector of 1(target) or 0(nontarget)

    """

    file_tsv = open(trial_key_file)
    key = []
    mask = []

    for line in file_tsv.readlines():
        if line.split('\t')[2] == 'target':
            key.append(1)
        else:
            key.append(0)
            
        if line.strip().split('\t')[3] == 'prog':
            mask.append(1)
        elif line.strip().split('\t')[3] == 'eval':
            mask.append(2)
        else:
            mask.append(3)

    file_tsv.close()
    
    # the first line is not label, remove it
    del key[0]
    del mask[0]

    #convert list  np.array, a vector
    key = np.array(key, dtype=int).flatten()
    mask = np.array(mask, dtype=int).flatten()
    
    return [key, mask]


def filter_dev_key(dev_ids, id_speaker_lookup, id_speaker_dict, dev_durations):
    """filter development ivectors

    Parameters
    ----------
    filename : dev_ids, id_speaker_lookup, id_speaker_lookup, dev_durations

    Returns
    -------
    filtered ivectors
    """

    if len(dev_ids) != len(dev_durations):
        exit(-1)
    
    y = []

    dev_durations_list = dev_durations
    dev_durations_list.tolist()
    
    dur_reverse_dict = {}
    for idx in range(len(dev_ids)):
        dev_id = dev_ids[idx]
        dur_reverse_dict[dev_id] = dev_durations[idx]
    
    for idx in range(len(dev_ids)):
        
        dev_id = dev_ids[idx]
        key_id = id_speaker_lookup[dev_id]
        seg_count = len(id_speaker_dict[key_id])
        if seg_count < 3:
            y.append(0)
        else:
            acc_count = 0
            for item in id_speaker_dict[key_id]:
                dur = dur_reverse_dict[item]        
                if dur > 30:
                    acc_count = acc_count + 1
            
            if acc_count > 3:
                y.append(1)
            else:
                y.append(0)
                
    return np.array(y)


def label_str_to_int(label_str):
    """label, string to int

    Parameters
    ----------
    filename : string label

    Returns
    -------
    int label
    """
    
    label_dict = {}
    label_int = []
    for item in label_str:
        if item not in label_dict.keys():
            label_dict[item] = len(label_dict) + 1
        label_int.append(label_dict[item])
    
    return np.array(label_int)

