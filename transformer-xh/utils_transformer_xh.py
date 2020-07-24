import json
import pickle
import numpy as np
import random
# from fairseq.data import Dictionary
# from fairseq import (
# 	checkpoint_utils, distributed_utils, metrics, options, progress_bar, tasks, utils
# )
import torch
import argparse
import os
# from fairseq.models.roberta import RobertaModel
import random
import dgl
from dgl import DGLGraph

random.seed(1)
np.random.seed(1) 
torch.manual_seed(1) 
torch.cuda.manual_seed(1)
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    mean_squared_error,
    accuracy_score,
    f1_score,
)

#from reco_utils.recommender.deeprec.io.iterator import BaseIterator


def mrr_score(y_true, y_score):
    """Computing mrr score metric.
    Args:
        y_true (numpy.ndarray): ground-truth labels.
        y_score (numpy.ndarray): predicted labels.
    
    Returns:
        numpy.ndarray: mrr scores.
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def ndcg_score(y_true, y_score, k=10):
    """Computing ndcg score metric at k.
    Args:
        y_true (numpy.ndarray): ground-truth labels.
        y_score (numpy.ndarray): predicted labels.
    Returns:
        numpy.ndarray: ndcg scores.
    """
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def hit_score(y_true, y_score, k=10):
    """Computing hit score metric at k.
    Args:
        y_true (numpy.ndarray): ground-truth labels.
        y_score (numpy.ndarray): predicted labels.
    Returns:
        numpy.ndarray: hit score.
    """
    ground_truth = np.where(y_true == 1)[0]
    argsort = np.argsort(y_score)[::-1][:k]
    for idx in argsort:
        if idx in ground_truth:
            return 1
    return 0


def dcg_score(y_true, y_score, k=10):
    """Computing dcg score metric at k.
    Args:
        y_true (numpy.ndarray): ground-truth labels.
        y_score (numpy.ndarray): predicted labels.
    Returns:
        numpy.ndarray: dcg scores.
    """
    k = min(np.shape(y_true)[-1], k)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def cal_metric(labels, preds, metrics):
    """Calculate metrics,such as auc, logloss.
    
    FIXME: 
        refactor this with the reco metrics and make it explicit.
    """
    res = {}
    for metric in metrics:
        if metric == "auc":
            auc = roc_auc_score(np.asarray(labels), np.asarray(preds))
            res["auc"] = round(auc, 4)
        elif metric == "rmse":
            rmse = mean_squared_error(np.asarray(labels), np.asarray(preds))
            res["rmse"] = np.sqrt(round(rmse, 4))
        elif metric == "logloss":
            # avoid logloss nan
            preds = [max(min(p, 1.0 - 10e-12), 10e-12) for p in preds]
            logloss = log_loss(np.asarray(labels), np.asarray(preds))
            res["logloss"] = round(logloss, 4)
        elif metric == "acc":
            pred = np.asarray(preds)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            acc = accuracy_score(np.asarray(labels), pred)
            res["acc"] = round(acc, 4)
        elif metric == "f1":
            pred = np.asarray(preds)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            f1 = f1_score(np.asarray(labels), pred)
            res["f1"] = round(f1, 4)
        elif metric == "mean_mrr":
            mean_mrr = np.mean(
                [
                    mrr_score(each_labels, each_preds)
                    for each_labels, each_preds in zip(labels, preds)
                ]
            )
            res["mean_mrr"] = round(mean_mrr, 4)
        elif metric.startswith("ndcg"):  # format like:  ndcg@2;4;6;8
            ndcg_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                ndcg_list = [int(token) for token in ks[1].split(";")]
            for k in ndcg_list:
                ndcg_temp = np.mean(
                    [
                        ndcg_score(each_labels, each_preds, k)
                        for each_labels, each_preds in zip(labels, preds)
                    ]
                )
                res["ndcg@{0}".format(k)] = round(ndcg_temp, 4)
        elif metric.startswith("hit"):  # format like:  hit@2;4;6;8
            hit_list = [1, 2]
            ks = metric.split("@")
            if len(ks) > 1:
                hit_list = [int(token) for token in ks[1].split(";")]
            for k in hit_list:
                hit_temp = np.mean(
                    [
                        hit_score(each_labels, each_preds, k)
                        for each_labels, each_preds in zip(labels, preds)
                    ]
                )
                res["hit@{0}".format(k)] = round(hit_temp, 4)
        elif metric == "group_auc":
            #print([(each_labels, each_preds) for each_labels, each_preds in zip(labels, preds) ])
            group_auc = np.mean(
                [
                    roc_auc_score(each_labels, each_preds)
                    for each_labels, each_preds in zip(labels, preds)
                ]
            )
            res["group_auc"] = round(group_auc, 4)
        else:
            raise ValueError("not define this metric {0}".format(metric))
    return res



def read_bert_features(filename):
    news_id={}
    max_length=20
    f=open(filename,'r')
    for line in f:
        line=line.strip().split('\t')
        features=line[1:]
        # if len(features)>max_length:
        #     assert 1==0
        news_id[line[0]]=features
    return news_id


def encode_sequence_hotpot(context,  max_seq_len):
    
    input_ids = context
    sequence_ids = [0]*len(context)
    input_mask = [1]*len(input_ids)

    while len(input_ids) < max_seq_len:
        input_ids.append(0)
        sequence_ids.append(0)
        input_mask.append(0)

    return (torch.LongTensor(input_ids), torch.LongTensor(input_mask), torch.LongTensor(sequence_ids))

def get_batch(filename,batch_size):
    
    doc_dict=read_bert_features('../../Recommenders/data/MINDlarge_train/news_token_bert_features.txt')
    f=open(filename,'r').readlines()
    batch_data=[]
    all_nodes=0
    i=0
    batch_graph=[]
    bert_max_len=20
    imp_index=[]
    batch_t=0
    while i<len(f):
        data=json.loads(f[i])
        neg_list=data['neg_list']
        neg_sample=np.random.choice(len(neg_list),1,replace=False)[0]
        #ex['neg_list']=doc_dict[neg_list[neg_sample]]
        data['node'][-1]['context']=doc_dict[neg_list[neg_sample]]
        all_nodes+=len(data['node'])
        g = DGLGraph()
        g.add_nodes(len(data['node']))
        # edge_dict = dict()
        # for edge in data['edges']:
        #     e_start = edge['start']
        #     e_end = edge['end']

        #     idx = (e_start, e_end)
        #     if idx not in edge_dict:
        #         edge_dict[idx] =list()
        #     # if edge['sent'] not in edge_dict[idx]:
        #     #     edge_dict[idx].append(edge['sent'])
        
        # for idx, context in edge_dict.items():
        #     start, end = idx
        #     g.add_edge(start, end)

        for k in range(len(data['node'])-2):
            for j in range(k+1,len(data['node'])):
                #if i!=len(node)-2 and j!=len(node)_1:
                # data['edges'].append({'start': i, 'end': j})
                # data['edges'].append({'start': j, 'end': i})
                g.add_edge(k, j)
                g.add_edge(j, k)

        for k, node in enumerate(data['node']):

            context = node['context']
            context=[int(x) for x in context]
            #evidence = list(set(node['evidence']))
            encoding_inputs, encoding_masks, encoding_ids = encode_sequence_hotpot(context, bert_max_len)
            g.nodes[k].data['encoding'] = encoding_inputs.unsqueeze(0)
            g.nodes[k].data['encoding_mask'] = encoding_masks.unsqueeze(0)
            g.nodes[k].data['segment_id'] = encoding_ids.unsqueeze(0)

            #print('????',node)

            if node['label'] == 1:
                g.nodes[k].data['label'] = torch.tensor(1).unsqueeze(0).type(torch.FloatTensor)
            elif node['label'] ==0:
                g.nodes[k].data['label'] = torch.tensor(0).unsqueeze(0).type(torch.FloatTensor)
            else:
                g.nodes[k].data['label'] = torch.tensor(-1).unsqueeze(0).type(torch.FloatTensor)

        if batch_t>=batch_size:
            batch_t=0
            g = dgl.batch(batch_graph)
            batch_graph=[]
            all_nodes=0
            yield  g,imp_index,torch.tensor([0]*len(imp_index))
            imp_index=[]
        else:
            batch_t+=1
            batch_graph.append(g)
            imp_index.append(data['imp_id'])
            #print('???',data['imp_id'],imp_index)
            i+=1




def get_batch_test(filename):
    
    #doc_dict=read_bert_features('../../fairseq/news_token_bert_features_test.txt')
    f=open(filename,'r').readlines()
    batch_data=[]
    all_nodes=0
    i=0
    batch_graph=[]
    bert_max_len=20
    imp_index=[]
    batch_t=0
    while i<len(f):
        data=json.loads(f[i])
        # neg_list=data['neg_list']
        # neg_sample=np.random.choice(len(neg_list),1,replace=False)[0]
        # #ex['neg_list']=doc_dict[neg_list[neg_sample]]
        # data['node'][-1]['context']=doc_dict[neg_list[neg_sample]]
        all_nodes+=len(data['node'])
        g = DGLGraph()
        g.add_nodes(len(data['node']))
        # edge_dict = dict()
        # for edge in data['edges']:
        #     e_start = edge['start']
        #     e_end = edge['end']

        #     idx = (e_start, e_end)
        #     if idx not in edge_dict:
        #         edge_dict[idx] =list()
        #     # if edge['sent'] not in edge_dict[idx]:
        #     #     edge_dict[idx].append(edge['sent'])
        # for idx, context in edge_dict.items():
        #     start, end = idx
        #     g.add_edge(start, end)
        for k in range(len(data['node'])):
            for j in range(k+1,len(data['node'])):
                #if i!=len(node)-2 and j!=len(node)_1:
                # data['edges'].append({'start': i, 'end': j})
                # data['edges'].append({'start': j, 'end': i})
                g.add_edge(k, j)
                g.add_edge(j, k)

        for k, node in enumerate(data['node']):

            context = node['context']
            context=[int(x) for x in context]
            #evidence = list(set(node['evidence']))
            encoding_inputs, encoding_masks, encoding_ids = encode_sequence_hotpot(context, bert_max_len)
            g.nodes[k].data['encoding'] = encoding_inputs.unsqueeze(0)
            g.nodes[k].data['encoding_mask'] = encoding_masks.unsqueeze(0)
            g.nodes[k].data['segment_id'] = encoding_ids.unsqueeze(0)

            #print('????',node)

            if node['label'] == 1:
                g.nodes[k].data['label'] = torch.tensor(1).unsqueeze(0).type(torch.FloatTensor)
            elif node['label'] ==0:
                g.nodes[k].data['label'] = torch.tensor(0).unsqueeze(0).type(torch.FloatTensor)
            else:
                g.nodes[k].data['label'] = torch.tensor(-1).unsqueeze(0).type(torch.FloatTensor)

        if all_nodes>=2500 or (i==len(f)-1):
            if i==(len(f)-1):
                batch_graph.append(g)
                imp_index.append(data['imp_id'])
                i+=1
            batch_t=0
            g = dgl.batch(batch_graph)
            batch_graph=[]
            all_nodes=0
            yield  g,imp_index,torch.tensor([0]*len(imp_index))
            imp_index=[]
        else:
            batch_t+=1
            batch_graph.append(g)
            imp_index.append(data['imp_id'])
            #print('???',data['imp_id'],imp_index)
            i+=1



def get_batch_dist(filename,batch_size,gpu_rank):
    
    doc_dict=read_bert_features('../../Recommenders/data/MINDlarge_train/news_token_bert_features.txt')
    f=open(filename,'r').readlines()
    length=int(len(f)/4)
    f=f[gpu_rank*length:(gpu_rank+1)*length]
    batch_data=[]
    all_nodes=0
    i=0
    batch_graph=[]
    bert_max_len=20
    imp_index=[]
    batch_t=0
    while i<len(f):
        data=json.loads(f[i])
        neg_list=data['neg_list']
        neg_sample=np.random.choice(len(neg_list),1,replace=False)[0]
        #ex['neg_list']=doc_dict[neg_list[neg_sample]]
        data['node'][-1]['context']=doc_dict[neg_list[neg_sample]]
        # data['node'][-3]['label']=-2

        all_nodes+=len(data['node'])
        g = DGLGraph()
        g.add_nodes(len(data['node']))
        # edge_dict = dict()
        # for edge in data['edges']:
        #     e_start = edge['start']
        #     e_end = edge['end']

        #     idx = (e_start, e_end)
        #     if idx not in edge_dict:
        #         edge_dict[idx] =list()
        #     # if edge['sent'] not in edge_dict[idx]:
        #     #     edge_dict[idx].append(edge['sent'])
        
        # for idx, context in edge_dict.items():
        #     start, end = idx
        #     g.add_edge(start, end)



        for k in range(len(data['node'])-2):
            for j in range(k+1,len(data['node'])):
                #if i!=len(node)-2 and j!=len(node)_1:
                # data['edges'].append({'start': i, 'end': j})
                # data['edges'].append({'start': j, 'end': i})
                g.add_edge(k, j)
                g.add_edge(j, k)

        for k, node in enumerate(data['node']):

            context = node['context']
            context=[int(x) for x in context]
            #evidence = list(set(node['evidence']))
            encoding_inputs, encoding_masks, encoding_ids = encode_sequence_hotpot(context, bert_max_len)
            g.nodes[k].data['encoding'] = encoding_inputs.unsqueeze(0)
            g.nodes[k].data['encoding_mask'] = encoding_masks.unsqueeze(0)
            g.nodes[k].data['segment_id'] = encoding_ids.unsqueeze(0)

            #print('????',node)

            if node['label'] == 1:
                g.nodes[k].data['label'] = torch.tensor(1).unsqueeze(0).type(torch.FloatTensor)
            elif node['label'] ==0:
                g.nodes[k].data['label'] = torch.tensor(0).unsqueeze(0).type(torch.FloatTensor)
            elif k==len(data['node'])-3:
                g.nodes[k].data['label'] = torch.tensor(-2).unsqueeze(0).type(torch.FloatTensor)
            else:
                g.nodes[k].data['label'] = torch.tensor(-1).unsqueeze(0).type(torch.FloatTensor)

        if batch_t>=batch_size:
            batch_t=0
            g = dgl.batch(batch_graph)
            batch_graph=[]
            all_nodes=0
            yield  g,imp_index,torch.tensor([0]*len(imp_index))
            imp_index=[]
        else:
            batch_t+=1
            batch_graph.append(g)
            imp_index.append(data['imp_id'])
            #print('???',data['imp_id'],imp_index)
            i+=1


