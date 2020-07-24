# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .model import Model, ModelHelper
import torch
import torch.nn as nn


class ModelHelper_Hotpot2(ModelHelper):
    def __init__(self, node_encoder, args, bert_config, config_model):
        super(ModelHelper_Hotpot2, self).__init__(node_encoder, args, bert_config, config_model)
        #self.mrc_final_layer = nn.Linear(self.config.hidden_size, 2)
        #self.mrc_final_layer.apply(self.init_weights)
        #self._init_weights(self.mrc_final_layer)
    
    
    def forward(self, batch, device):    
        ### Transformer-XH for node representations
        g = batch[0]
        # g.ndata['encoding'] = g.ndata['encoding'].to(device)
        # g.ndata['encoding_mask'] = g.ndata['encoding_mask'].to(device)
        # g.ndata['segment_id'] = g.ndata['segment_id'].to(device)
        outputs = self.node_encoder(g, g.ndata['encoding'], g.ndata['segment_id'], g.ndata['encoding_mask'], gnn_layer=self.config_model['gnn_layer'])
        node_sequence_output = outputs[0]
        node_pooled_output = outputs[1]
        node_pooled_output = self.node_dropout(node_pooled_output)
        
        #### Task specific layer (last layer)
        #mrc_logits = self.mrc_final_layer(node_sequence_output)
        node_idx=[i for i in range(batch[0].ndata['label'].size(0)) if batch[0].ndata['label'][i].item()!=-1 and batch[0].ndata['label'][i].item()!=-2  ]
        his_idx=[i for i in range(batch[0].ndata['label'].size(0)) if batch[0].ndata['label'][i].item()==-2 ]
        # print(len(node_idx))
        # print(len(his_idx))

        can_features=node_pooled_output[node_idx]
        his_features=node_pooled_output[his_idx]
        #print(his_features.shape,can_features.shape)
        assert 2*len(his_idx)==len(node_idx)

        his_features=his_features.unsqueeze(1).transpose(1,2).repeat(1,1,2).transpose(1,2).reshape(-1,his_features.shape[-1])
        #print(his_features.shape,can_features.shape)
        features=torch.cat( (his_features,can_features ) ,1)

        # print('features: ',features)

        # print('dense: ',self.score2(features))

        res=self.final_layer1(features)
        logits=self.final_layer2(res)



        # logits = self.final_layer(node_pooled_output).squeeze(-1)

        return logits#, mrc_logits




class Model_Hotpot2(Model):
    def __init__(self, args, config):
        super(Model_Hotpot2, self).__init__(args, config)
        self.network= ModelHelper_Hotpot2(self.bert_node_encoder, self.args, self.bert_config, self.config_model)




