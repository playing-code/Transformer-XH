import json
import pickle
import numpy as np
import random
# from fairseq.data import Dictionary
import sys
import torch
import argparse
import os
# from model_plain_bert2 import  Plain_bert
# from fairseq.models.roberta import RobertaModel
from utils_transformer_xh import cal_metric
import utils_transformer_xh as utils
import dgl
from dgl import DGLGraph
from datetime import timedelta
# import dgl
# import dgl.function as fn
#from gpu_mem_track import  MemTracker
#import inspect
#from multiprocessing import Pool
import math

import json

from model import Model_Hotpot2
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


# from multiprocessing import Pool
# from multiprocessing import Process

random.seed(1)
np.random.seed(1) 
torch.manual_seed(1) 
torch.cuda.manual_seed(1)

#cudaid=0

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


	return parser.parse_args()

def init_process(rank,args,fn,minutes=720):
	""" Initialize the distributed environment. """
	# os.environ['MASTER_ADDR'] = '127.0.0.1'
	# os.environ['MASTER_PORT'] = '1234'
	# torch.distributed.init_process_group(backend, rank=rank, world_size=size)
	print('???',rank)
	# fn(rank, size, model,optimizer,scheduler,config,args)

	# dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
	# 	master_ip='127.0.0.1', master_port='12345')
	dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
		master_ip='localhost', master_port='12345')
	# print('???',rank)
	dist.init_process_group(backend='gloo',
				init_method=dist_init_method,
				# If you have a larger dataset, you will need to increase it.
				timeout=timedelta(minutes=minutes),
				world_size=4,
				rank=rank)
	#print('???',rank)
	assert torch.distributed.get_rank() == rank
	fn(rank, args)


def average_gradients(model):
	size = float(dist.get_world_size())
	for param in model.network.parameters():
		dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
		param.grad.data /= size



def main(rank,args):

	#print('params: '," T_warm: ",T_warm," all_iteration: ",all_iteration," lr: ",lr)
	#writer = SummaryWriter('./model_snapshot_error')
	# cuda_list=range(cuda_num)
	#cuda_list=[2,1,0]
	#print('run')
	random.seed(1)
	np.random.seed(1) 
	torch.manual_seed(1) 
	torch.cuda.manual_seed(1)
	torch.set_num_threads(1)

	config = json.load(open(args.config_file, 'r', encoding="utf-8"))
	model=Model_Hotpot2(args, config)
	#model.network.to(cudaid)

	param_optimizer = list(model.network.named_parameters())
	param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
	no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in param_optimizer if not any(
			nd in n for nd in no_decay)], 'weight_decay': 0.01},
		{'params': [p for n, p in param_optimizer if any(
			nd in n for nd in no_decay)], 'weight_decay': 0.0}
	]
	optimizer = AdamW(optimizer_grouped_parameters, lr=config["training"]["learning_rate"])
	scheduler = WarmupLinearSchedule(optimizer, warmup_steps=config["training"]["warmup_proportion"], t_total=config["training"]["total_training_steps"])

	#cudaid=dist.get_rank()
	cudaid=rank
	model.network.to(cudaid)
	args.device = cudaid
	#model.cuda(cudaid)
	accumulation_steps=16
	#model = nn.DataParallel(model, device_ids=cuda_list)
	print('rank: ',rank)
	accum_batch_loss=1
	#train_file='train_ms_roberta_plain_pair_sample_shuffle.txt'
	train_file='train_ms_transformer_xh_shuffle.txt'
	#train_file='train_ms_roberta.txt'
	#iterator=NewsIterator(batch_size=24, npratio=4)
	#for epoch in range(0,100):
	batch_t=0
	iteration=0
	#print('train...',cuda_list)
	pre_batch_t=177796
	
	epoch=0
	model.train()
	for epoch in range(0,10):
	#while True:
		all_loss=0
		all_batch=0
		data_batch=utils.get_batch_dist(train_file,8,rank)
		for  batch  in data_batch:
			#g,imp_index,label
			#batch=imp_index , g , label
			#print('batch: ',batch)
			batch_t+=1
			#batch[0] = batch[0].to(torch.device('cuda:0'))
			# g=batch[0].to(torch.device('cuda:'+str(cudaid)))
			g=batch[0].to(torch.device('cuda:'+str(cudaid)))
			logit=model.network((g,batch[-1]), cudaid)
			label = batch[-1].cuda(cudaid)
			#print('logit: ',logit.shape)
			#pos_node_idx = [i for i in range(batch[2].size(0)) if batch[1][i].item() != -1]
			#print('????label:',batch[0].ndata['label'])
			#pos_node_idx=[i for i in range(batch[0].ndata['label'].size(0)) if batch[0].ndata['label'][i].item()!=-1 ]
			#print('pos_node_idx: ',pos_node_idx)
			# logit=logit[pos_node_idx].reshape(-1,2)
			logit=logit.reshape(-1,2)
			#print('logit: ',logit.shape)

			loss=F.nll_loss(
					F.log_softmax(
						logit.view(-1, logit.size(-1)),
						dim=-1,
						dtype=torch.float32,
					),
					label.view(-1),
					reduction='sum',
					#ignore_index=self.padding_idx,
				)

			#sample_size=float(sample_size.sum())
			#loss=loss.sum()/sample_size/math.log(2)
			loss=loss/len(batch[1])/math.log(2)
			# sample_size=float(sample_size)
			# loss=loss/sample_size/math.log(2)
			#print(' batch_t: ',batch_t, '  epoch: ',epoch,' loss: ',float(loss))
			
			accum_batch_loss+=float(loss)

			all_loss+=float(loss)
			all_batch+=1

			loss = loss/accumulation_steps
			loss.backward()

			if (batch_t)%accumulation_steps==0:
				#print('candidate_id: ',candidate_id)
				# total_norm=0
				# for p in model.network.parameters():
				# 	if p.grad==None:
				# 		print('error: ',index,p.size(),p.grad)
				# 	param_norm = p.grad.data.norm(2)
				# 	total_norm += param_norm.item() ** 2
				# total_norm = total_norm ** (1. / 2)
				

				# total_clip_norm=0
				# for p in model.network.parameters():
				# 	if p.grad==None:
				# 		print('error: ',index,p.size(),p.grad)
				# 	param_norm = p.grad.data.norm(2)
				# 	total_clip_norm += param_norm.item() ** 2
				# total_clip_norm = total_clip_norm ** (1. / 2)

				iteration+=1
				#adjust_learning_rate(optimizer,iteration)
				average_gradients(model)
				torch.nn.utils.clip_grad_norm_(model.network.parameters(), 1.0)
				optimizer.step()
				scheduler.step()
				optimizer.zero_grad()
				#print(' batch_t: ',batch_t, ' iteration: ', iteration, ' epoch: ',epoch,' accum_batch_loss: ',accum_batch_loss/accumulation_steps, " total_norm: ", total_norm,' clip_norm: ',total_clip_norm)
				

				if rank==0:
					print(' batch_t: ',batch_t, ' iteration: ', iteration, ' epoch: ',epoch,' accum_batch_loss: ',accum_batch_loss/accumulation_steps)
					torch.save(model.network.state_dict(),'./models/transformer_xh'+str(epoch)+'.pkl')
				accum_batch_loss=0

		dist.barrier()
				
			

if __name__ == '__main__':

	# cuda_num=int(sys.argv[1])
	random.seed(1)
	np.random.seed(1) 
	torch.manual_seed(1) 
	torch.cuda.manual_seed(1)
	cuda_num=3
	#main()
	#mydict=utils.load_dict('/home/shuqilu/fairseq/model/roberta.base/')
	#model=Plain_bert(padding_idx=mydict['<pad>'],vocab_size=len(mydict))

	#optimizer = torch.optim.Adam(model.parameters(), lr=lr,betas=(0.9,0.98),eps=1e-6,weight_decay=0.0)
	args = parse_args()
	args.max_seq_length=20
	
	

		#model.cuda(cudaid)
	#model.network.to(cudaid)
	#train(model,optimizer,scheduler,config,args)
	size = 4
	# processes = []
	# for rank in [1,2]:
	# 	p = Process(target=init_process, args=(rank, size, train, model,optimizer,scheduler,config,args))
	# 	p.start()
	# 	processes.append(p)
	# print('start....')

	# for p in processes:
	# 	p.join()
	mp = torch.multiprocessing.get_context('spawn')
	procs = []
	for rank in range(size):
		procs.append(mp.Process(target=init_process, args=(rank,args,main), daemon=True))
		procs[-1].start()
	for p in procs:
		p.join()

	
			

























