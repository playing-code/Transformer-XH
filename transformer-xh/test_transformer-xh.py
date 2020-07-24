import json
import pickle
import numpy as np
import random
# from fairseq.data import Dictionary
import sys
import torch
import argparse
import os
from model_plain_bert2 import  Plain_bert
from fairseq.models.roberta import RobertaModel
from utils_transformer_xh import cal_metric
import utils_transformer_xh as utils
import dgl
from dgl import DGLGraph
# import dgl
# import dgl.function as fn
#from gpu_mem_track import  MemTracker
#import inspect
#from multiprocessing import Pool
import math


import argparse
import json

from model import Model_Hotpot
#import data
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm

from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers import AdamW, WarmupLinearSchedule
from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
import torch.nn.functional as F

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
random.seed(1)
np.random.seed(1) 
torch.manual_seed(1) 
torch.cuda.manual_seed(1)

#cudaid=2
metrics=['group_auc','mean_mrr','ndcg@5;10']

def parse_args():
    parser = argparse.ArgumentParser("Transformer-XH")
    parser.add_argument("--config-file", "--cf",
                    help="pointer to the configuration file of the experiment", type=str, required=True)
    parser.add_argument("--max_seq_length", default=128, type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                    "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--no_cuda",
                    default=False,
                    action='store_true',
                    help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_accumulation_steps',
                    type=int,
                    default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                    default=False,
                    action='store_true',
                    help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--seed',
                    type=int,
                    default=42,
                    help="random seed for initialization")
    parser.add_argument('--checkpoint',
                    type=int,
                    default=2500)
    parser.add_argument('--loss_scale',
                    type=float, default=0,
                    help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--test',
                    default=False,
                    action='store_true',
                    help="Whether on test mode")
    parser.add_argument("--test-file", "--tf",
                    help="pointer to the configuration file of the experiment", type=str, required=True)
    parser.add_argument("--cudaid", "--cid",
                    help="pointer to the configuration file of the experiment", type=int, required=True)


    return parser.parse_args()



def group_labels_func(labels, preds, group_keys):
    """Devide labels and preds into several group according to values in group keys.

    Args:
        labels (list): ground truth label list.
        preds (list): prediction score list.
        group_keys (list): group key list.

    Returns:
        all_labels: labels after group.
        all_preds: preds after group.

    """

    all_keys = list(set(group_keys))
    group_labels = {k: [] for k in all_keys}
    group_preds = {k: [] for k in all_keys}

    for l, p, k in zip(labels, preds, group_keys):
        group_labels[k].append(l)
        group_preds[k].append(p)

    all_labels = []
    all_preds = []
    for k in all_keys:
        all_labels.append(group_labels[k])
        all_preds.append(group_preds[k])

    return all_labels, all_preds




def test(model,config,args,cuda_num,test_file):

	
	#train_file='train_ms_roberta_plain_pair_sample_shuffle.txt'
	#test_file='test_ms_transformer_xh.txt'
	preds = []
	labels = []
	imp_indexes = []
	#train_file='train_ms_roberta.txt'
	#iterator=NewsIterator(batch_size=24, npratio=4)
	#for epoch in range(0,100):
	batch_t=0
	iteration=0
	#print('train...',cuda_list)
	pre_batch_t=177796
	
	epoch=0
	model.eval()

	with torch.no_grad():
		data_batch=utils.get_batch_test(test_file)
		for  batch  in data_batch:
			#g,imp_index,label
			#batch=imp_index , g , label
			#print('batch: ',batch)
			batch_t+=len(batch[1])
			#batch[0] = batch[0].to(torch.device('cuda:2'))
			g=batch[0].to(torch.device('cuda:'+str(cudaid)))
			logit=model.network((g,batch[-1]), cudaid)
			#label = batch[-1].cuda(cudaid)
			#print('logit: ',logit.shape)
			#pos_node_idx = [i for i in range(batch[2].size(0)) if batch[1][i].item() != -1]
			#print('????label:',batch[0].ndata['label'])

			pos_node_idx=[i for i in range(batch[0].ndata['label'].size(0)) if batch[0].ndata['label'][i].item()!=-1 ]
			#print('pos_node_idx: ',pos_node_idx)
			logit=logit[pos_node_idx].reshape(-1)
			logit=F.sigmoid(logit)
			
			label=batch[0].ndata['label'][pos_node_idx]
			# print('logit: ',logit)
			# print('label: ',label)
			logit=np.reshape(np.array(logit.cpu()), -1)
			label=np.reshape(np.array(label.cpu()), -1)
			imp_index=batch[1]
			print('batch_t:',batch_t)
			for i in range(len(imp_index)):
				print('imp_index:',imp_index[i],'logit:',logit[i],'label:',label[i])
			

			# preds.extend(logit)
			# labels.extend(label)
			# imp_indexes.extend(np.reshape(np.array(imp_index), -1))
			#batch_t+=len(candidate_id)
			#print('logit: ',logit.shape)
			#break
		# group_labels, group_preds = group_labels_func(labels, preds, imp_indexes)
		# res = cal_metric(group_labels, group_preds, metrics)

	#return res

def exact_result():
	preds = []
	labels = []
	imp_indexes = []
	f1=open('log_transformer_test6_0.txt','r').readlines()
	i=0
	for line in f1:

		if line[:9]=='imp_index':
			if 'Using' in line:
				# print('???',line,f1[i+1],f1[i+2],f1[i+3])

				# assert 1==0
				logit=0.5320914
				imp_index=23601
				label=1
				preds.append(logit)
				labels.append(label)
				imp_indexes.append(imp_index)
			else:
				line=line.strip().split(' ')
				#print('???',line)
				logit=float(line[3])
				imp_index=int(line[1])
				label=int(float(line[5]))
				preds.append(logit)
				labels.append(label)
				imp_indexes.append(imp_index)
		i+=1

	f1=open('log_transformer_test6_1.txt','r').readlines()
	i=0
	for line in f1:
		if line[:9]=='imp_index':
			if 'Using' in line:
				# print('???',line,f1[i+1],f1[i+2],f1[i+3])
				# assert 1==0
				logit=0.75541914
				imp_index=47433
				label=0
				preds.append(logit)
				labels.append(label)
				imp_indexes.append(imp_index)
			else:
				line=line.strip().split(' ')
				logit=float(line[3])
				imp_index=int(line[1])
				label=int(float(line[5]))
				preds.append(logit)
				labels.append(label)
				imp_indexes.append(imp_index)
		i+=1

	f1=open('log_transformer_test6_2.txt','r').readlines()
	i=0
	for line in f1:
		if line[:9]=='imp_index':
			if 'Using' in line:
				# print('???',line,f1[i+1],f1[i+2],f1[i+3])
				# assert 1==0
				logit=0.34159982
				imp_index=70936
				label=0
				preds.append(logit)
				labels.append(label)
				imp_indexes.append(imp_index)
			else:
				line=line.strip().split(' ')
				logit=float(line[3])
				imp_index=int(line[1])
				label=int(float(line[5]))
				preds.append(logit)
				labels.append(label)
				imp_indexes.append(imp_index)
		i+=1

	group_labels, group_preds = group_labels_func(labels, preds, imp_indexes)
	res = cal_metric(group_labels, group_preds, metrics)
	print(res)



def splice_file():
	test_file='test_ms_transformer_xh.txt'
	f=open(test_file,'r').readlines()
	#f=shuffle_data(test_file)
	w1=open('test_ms_transformer_xh_splice0.txt','w')
	w2=open('test_ms_transformer_xh_splice1.txt','w')
	w3=open('test_ms_transformer_xh_splice2.txt','w')
	for line in f[:int(len(f)/3)]:
		w1.write(line)

	for line in f[int(len(f)/3):int(2*len(f)/3)]:
		w2.write(line)

	for line in f[int(2*len(f)/3):]:
		w3.write(line)

	w1.close()
	w2.close()
	w3.close()


if __name__ == '__main__':

	# cuda_num=int(sys.argv[1])

	# splice_file()
	# assert 1==0
	exact_result()
	assert 1==0

	random.seed(1)
	np.random.seed(1) 
	torch.manual_seed(1) 
	torch.cuda.manual_seed(1)
	
	#main()
	#mydict=utils.load_dict('/home/shuqilu/fairseq/model/roberta.base/')
	#model=Plain_bert(padding_idx=mydict['<pad>'],vocab_size=len(mydict))

	#optimizer = torch.optim.Adam(model.parameters(), lr=lr,betas=(0.9,0.98),eps=1e-6,weight_decay=0.0)
	args = parse_args()
	args.max_seq_length=20
	test_file=args.test_file
	cudaid=args.cudaid

	args.device = cudaid

	config = json.load(open(args.config_file, 'r', encoding="utf-8"))
	model=Model_Hotpot(args, config)
	model.network.load_state_dict(torch.load('./models/transformer_xh6.pkl', map_location=lambda storage, loc: storage))
	model.network.to(cudaid)

	res=test(model,config,args,cudaid,test_file)
	print(res)
	
	
			

























