import numpy as np
import torch
import os
import sys
import pickle as pkl
import time
feat_dir = "/home/lhy/datasets/Places2/feat/"
#exampler_dir = "/home/lhy/datasets/Places2/examplers"
exampler_flist_path = "/home/lhy/datasets/Places2/train_exampler_flist.txt"
img_flist_path = "/home/lhy/datasets/Places2/train_flist.txt"

def distance(x, y):
    return np.sum((x-y)**2)

def compute_retrieval_matrix(feats,k=50):
    N = len(feats)
    dist_matrix = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i == j:
                dist_matrix[i][j] = np.inf
            else:
                dist_matrix[i][j] = distance(feats[i], feats[j])
    retrieval_results = []
    for i in range(N):
        topk = np.argsort(dist_matrix[i])[:k]
        retrieval_results.append(topk)
    return retrieval_results



cls_record = {}
feat_records = []
feat_path_records = []
img_path_records = []
start = time.time()
with open(exampler_flist_path, 'w') as wf:
    with open(img_flist_path, "r") as rf:
        for line in rf:
            img_path = line.strip()

            cls_label = img_path[:img_path.rfind("/")]

            feat_path = img_path.replace("data_256", "feat/data_256")
            feat_path = feat_path.replace("jpg", 'pkl')
            feat = pkl.load(open(feat_path, "rb"))
            fc7_feat = feat["out"]

            if cls_label in cls_record:
                img_path_records.append(img_path)
                feat_path_records.append(feat_path)
                feat_records.append(fc7_feat)
            else:
                exampler_dir = cls_label.replace("data_256", "examplers/data_256")
                print(exampler_dir)
                if not os.path.exists(exampler_dir):
                    os.makedirs(exampler_dir)
                if len(cls_record) > 0:
                    print(cls_record, cls_label)

                    retri_results = compute_retrieval_matrix(feat_records)
                    for i in range(len(retri_results)):
                        exampler_path = feat_path_records[i].replace("feat", "examplers")
                        pkl.dump({"retrieval":retri_results[i]}, open(exampler_path, 'wb'))
                        wf.write("{} {}\n".format(img_path_records[i], img_path_records[retri_results[i][0]]))
                end = time.time()
                print("A Class Retrieval Time:{}".format(end-start))
                start = time.time()
                feat_records = []
                feat_path_records = []
                cls_record[cls_label] = 1
                feat_records.append(fc7_feat)
                img_path_records.append(img_path)
                feat_path_records.append(feat_path)
