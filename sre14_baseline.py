# *- coding:UTF-8 -*-


import numpy as np
from sklearn import discriminant_analysis
from compute_eer_and_mdcf import *
from sre14_io import *
import torch
import os

def sre14_baseline_train_lambda(raw_data_path,save_data_path):
    lda_dim = 250

    print("=== begin load vectors and labels\n")
    # load ivector ids, durations and ivectors (as row vectors)
    dev_ids, dev_durations, dev_ivec = load_ivectors(raw_data_path + 'dev_ivectors.csv')
    model_ids, model_durations, model_ivec = load_ivectors(raw_data_path + 'model_ivectors.csv')
    test_ids, test_durations, test_ivec = load_ivectors(raw_data_path + 'test_ivectors.csv')

    # load model key corresponding to the same ordering as model_ids
    model_key = load_model_key(raw_data_path + 'target_speaker_models.csv', model_ids)

    # load development data key corresponding to the same ordering as dev_ids
    [dev_key, dev_id_speaker_lookup, dev_id_speaker_dict] = load_dev_key(raw_data_path + 'development_data_labels.csv',
                                                                         dev_ids)
    dev_mask = filter_dev_key(dev_ids, dev_id_speaker_lookup, dev_id_speaker_dict, dev_durations)

    # load development data key corresponding to the same ordering as dev_ids
    [test_key, test_mask] = load_test_key(raw_data_path + 'ivec14_sre_trial_key_release.tsv')
    
    print("=== whiten\n")
    # compute the mean and whitening transformation over dev set only
    m = np.mean(dev_ivec, axis=0)
    S = np.cov(dev_ivec, rowvar=0)
    D, V = np.linalg.eig(S)
    # W = (1/np.sqrt(D) * V).transpose().astype('float32')

    #    # center and whiten all i-vectors
    #    dev_ivec = np.dot(dev_ivec - m, W.transpose())
    #    model_ivec = np.dot(model_ivec - m, W.transpose())
    #    test_ivec = np.dot(test_ivec - m, W.transpose())

    # center and whiten all i-vectors
    dev_ivec = dev_ivec - m
    model_ivec = model_ivec - m

    # dev filter
    dev_ivec = [dev_ivec[i] for i in range(0, len(dev_ivec)) if dev_mask[i] == 1]
    dev_key = [dev_key[i] for i in range(0, len(dev_key)) if dev_mask[i] == 1]

    adj = []
    # import pdb;pdb.set_trace()
    norm = np.linalg.norm(model_ivec, axis = 1)
    mat = np.dot(model_ivec, model_ivec.T)
    for j in range(model_ivec.shape[0]):
        if j%500==0:
            print(j)
        sim = []
        for i in range(model_ivec.shape[0]):
            cur_vec = model_ivec[i]
            cos = mat[j,i]/norm[i]/norm[j]
            sim.append(cos)
        adj.append(sim)
    
    # import pdb;pdb.set_trace()
    print("=== LDA\n")
    # train LDA using model_ivec and dev_ivec
    model_dev_ivec = np.vstack((model_ivec, dev_ivec))
    model_dev_key = np.hstack((model_key, dev_key))
    #    model_dev_ivec = np.vstack(dev_ivec)
    #    model_dev_key = np.hstack(dev_key)
    lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=lda_dim)
    lda.fit(model_dev_ivec, model_dev_key)

    # save data
    print("=== save lambda data")
    dev_data = [dev_ivec, dev_key]
    torch.save(dev_data, save_data_path + "dev_data.pth")

    model_data = [model_ivec, model_key]
    torch.save(model_data, save_data_path + "model_data.pth")

    test_data = [test_ivec, test_key, test_mask]
    torch.save(test_data, save_data_path + "test_data.pth")

    m_lda=[m,lda]
    torch.save(m_lda,save_data_path+"mean_lda.pth")


def sre14_baseline_train_speaker(save_data_path):
    [m,lda] = torch.load(save_data_path + "mean_lda.pth")
    [dev_ivec, dev_key] = torch.load(save_data_path + "dev_data.pth")
    [model_ivec, model_key] = torch.load(save_data_path + "model_data.pth")

    print("=== tranining speaker model")
    # use lda to convert ivector to low dimension ivector
    dev_ivec = lda.transform(dev_ivec)
    model_ivec = lda.transform(model_ivec)

    avg_model_ivec = np.zeros((len(np.unique(model_key)), model_ivec.shape[1]))
    avg_model_names = []
    for i, key in enumerate(np.unique(model_key)):
        avg_model_ivec[i] = np.mean(model_ivec[model_key == key], axis=0)
        avg_model_names.append(key)

    speaker_model = [dev_ivec, model_ivec, avg_model_ivec]
    torch.save(speaker_model, save_data_path + "speaker_model.pth")


def sre14_baseline_test(save_data_path, ):
    # load test data
    print("=== test...\n")
    [test_ivec, test_key, test_mask] = torch.load(save_data_path + "test_data.pth")
    [m, lda] = torch.load(save_data_path + "mean_lda.pth")
    test_ivec = test_ivec - m
    test_ivec = lda.transform(test_ivec)
    test_data = [test_ivec, test_key]
    torch.save(test_data, save_data_path + "test.pth")


def sre14_baseline_eval_score(save_data_path, score_file):
    [dev_ivec, model_ivec, avg_model_ivec] = torch.load(save_data_path + "speaker_model.pth")
    [test_ivec, test_key] = torch.load(save_data_path + "test.pth")
    # cosine score
    # project the converted develepment i-vectors into unit sphere
    print("=== cosine scoring\n")
    dev_ivec /= np.sqrt(np.sum(dev_ivec ** 2, axis=1))[:, np.newaxis]
    model_ivec /= np.sqrt(np.sum(model_ivec ** 2, axis=1))[:, np.newaxis]
    test_ivec /= np.sqrt(np.sum(test_ivec ** 2, axis=1))[:, np.newaxis]
    avg_model_ivec /= np.sqrt(np.sum(avg_model_ivec ** 2, axis=1))[:, np.newaxis]
    score = np.dot(avg_model_ivec, test_ivec.T)
    score_col = score.flatten()

    print("=== evaluate score\n")
    [eer, mindcf_sre08, mindcf_sre10, mindcf_sre12, mindcf_sre14, mindcf_sre16] = compute_eer_mdcf(score_col, test_key)

    # print("user  :", getpass.getuser())
    # print("time  :", get_current_time())
    print("eer = {0:.2f} %".format(100 * eer))
    # print("mindcf_sre08 = {0:.4f}".format(mindcf_sre08))
    # print("mindcf_sre10 = {0:.4f}".format(mindcf_sre10))
    # print("mindcf_sre12 = {0:.4f}".format(mindcf_sre12))
    print("mindcf_sre14 = {0:.4f}".format(mindcf_sre14))
    # print("mindcf_sre16 = {0:.4f}".format(mindcf_sre16))

def main():
    print("=== begin run.\n")

    raw_data_path = '../data/'
    save_data_path = "./exp/"
    os.makedirs(save_data_path, exist_ok=True)
    # raw_data_path = 'D:/database/sre14_ivec_challenge/'
    score_file = './results/sre14_cosine_score.txt'

    sre14_baseline_train_lambda(raw_data_path, save_data_path)
    sre14_baseline_train_speaker(save_data_path)
    sre14_baseline_test(save_data_path)
    # sre14_baseline_eval_score(save_data_path, score_file)

    print("=== finish.\n")


if __name__ == '__main__':
    main()
