import re
import torch
import numpy as np
import math
import torch.nn as nn
import time
import subprocess
import torch.nn.functional as F
from model.generator import Beam
from data.data import DocField, DocDataset, DocIter
from utils.config import is_cqa_task
from model.graph_model import * 
from utils.train_helper import * 

class CqaNet(PointerNet):
    def __init__(self, args):
        super().__init__(args)
        d_mlp = args.d_mlp
        mlp_output_d=2
        self.linears = nn.ModuleList([nn.Linear(args.d_rnn, d_mlp),
                                      nn.Linear(args.d_rnn * 2, d_mlp),
                                      nn.Linear(d_mlp, mlp_output_d),
                                      nn.Linear(args.ehid, d_mlp)])
        answer_type_class_num=4
        self.answer_type_paragraph_linear=nn.Linear(args.d_rnn, d_mlp)
        self.answer_type_question_linear=nn.Linear(args.d_rnn, d_mlp)
        self.answer_type_linear=nn.Linear(d_mlp, answer_type_class_num)
        self.multi_task_loss_weight = nn.Parameter(torch.ones(2))
         
    def forward(self, src_and_len, tgt_and_len, doc_num, ewords_and_len, elocs,label_of_one_batch_and_len,answer_types):
        entity_logics, answer_type_logics=self.gen_logics(src_and_len,doc_num,ewords_and_len,elocs)
        
        entity_class_weight =get_weights_inverse_num_of_samples(2,np.array([6770678,103586]))
        answer_type_class_weight =get_weights_inverse_num_of_samples(4,np.array([10987,8448,87836,1376]))
        # entity_class_weight = torch.tensor([1,50], dtype=torch.float32,device=torch.device("cuda"))
        # answer_type_class_weight = torch.tensor([10,10,1,10], dtype=torch.float32,device=torch.device("cuda"))
        entity_loss=self.gen_loss(label_of_one_batch_and_len[0],entity_logics,True,ewords_and_len,entity_class_weight)
        answer_type_loss=self.gen_loss(answer_types,answer_type_logics,False,None,answer_type_class_weight)
        
         
        loss=self.gen_multi_task_loss( entity_loss,answer_type_loss,False)
        return loss
        # return  entity_loss 


    def gen_multi_task_loss(self,entity_loss,answer_type_loss,is_fixed_weight):
        if is_fixed_weight:
            fixed_weight=get_weights_inverse_num_of_samples(2,np.array([6874264,108647]))
            loss=  entity_loss*fixed_weight[0]+answer_type_loss*fixed_weight[1]
        else:
            loss=0.5 * torch.stack((entity_loss,answer_type_loss)) / self.multi_task_loss_weight**2
            loss=loss.sum() + torch.log(self.multi_task_loss_weight.prod())
        return loss

    

    def gen_loss(self,label_of_one_batch,logics,need_mask,ewords_and_len,class_weight):
        if need_mask:
            entity_mask=self.gen_entity_mask( ewords_and_len[1],logics)
            logics.masked_fill_(entity_mask , -1e9)
        
        logp = F.log_softmax(logics, dim=-1)
        logp = logp.view(-1, logp.size(-1))

        loss_function = nn.NLLLoss(weight= class_weight )
        loss = loss_function(logp, label_of_one_batch.contiguous().view(-1))
        loss = loss.sum()/label_of_one_batch.size(0)
        return loss

    #some entities are padded fake entity
    def gen_entity_mask(self, entity_num,e):
        entity_mask=torch.ones_like(e ,dtype=torch.bool)
        # entity_mask  = e.new_zeros(label_of_one_batch.size(0), label_of_one_batch.size(1))
        # [e.new_zeros(e.size(0), 1, e.size(2)).byte()]
        for i in range(len(entity_num)):
            entity_mask[i,:entity_num[i],:]=False
        return entity_mask

    def gen_logics(self, src_and_len,   doc_num, ewords_and_len, elocs ):
        document_matrix, _, hcn, key,entity_h,paragraph_h = self.encode(src_and_len, doc_num, ewords_and_len, elocs)
        cur_question_h=self.gen_question_h(key,doc_num)
 
        # cur_question_h=cur_question_h.unsqueeze(1)
        entity_logics=self.gen_entity_relation(cur_question_h,entity_h )
        answer_type_logics=self.gen_answer_type_relation(cur_question_h,paragraph_h )
        return entity_logics, answer_type_logics

    def gen_entity_relation(self,question_hidden, compared_hidden ):
        compared_hidden = self.linears[3](compared_hidden)  
        question_hidden = self.linears[0](question_hidden) 
        relations=question_hidden+compared_hidden
        e = torch.tanh(relations)
        e = self.linears[2](e) 
        return e

    def gen_answer_type_relation(self,question_hidden, compared_hidden ):
        compared_hidden = self.answer_type_paragraph_linear(compared_hidden)  
        question_hidden = self.answer_type_question_linear(question_hidden) 
        relations=question_hidden+compared_hidden
        e = torch.tanh(relations)
        e = self.answer_type_linear(e) 
        return e

    def gen_question_h(self,key,doc_num):
        question_h_list=[]
        for i in range(len(doc_num)):
            one_question_h=key[i,doc_num[i]-1,:]
            question_h_list.append(one_question_h)
        batch_question_h=torch.stack(question_h_list,dim=0)
        batch_question_h=batch_question_h.unsqueeze(1)
        return batch_question_h

    def predict(self, src_and_len, doc_num, ewords_and_len, elocs):
        entity_logics, answer_type_logics=self.gen_logics(src_and_len,doc_num,ewords_and_len,elocs)
        m = nn.Softmax(dim=-1)
        predicted_entity_labels=self.gen_predicted_labels(entity_logics,m)
        predicted_answer_type_labels=self.gen_predicted_labels(answer_type_logics,m)
    
        return predicted_entity_labels   ,predicted_answer_type_labels   
    
    def gen_predicted_labels(self,logics,m):
        classes_probability = m(logics) 
        predicted_labels = torch.argmax(classes_probability, axis=-1)   
        return predicted_labels


    def predict_coqa_answer(self, src_and_len, doc_num, ewords_and_len, elocs,ewords_str,paragraph_id_list,turn_id_list):
        entity_logics, answer_type_logics=self.gen_logics(src_and_len,doc_num,ewords_and_len,elocs)
        m = nn.Softmax(dim=-1)
        predicted_entity_labels=self.gen_predicted_labels(entity_logics,m)
        predicted_answer_type_labels=self.gen_predicted_labels(answer_type_logics,m)

        answer,answer_json_array=self.gen_coqa_answer(predicted_entity_labels,predicted_answer_type_labels,ewords_str,paragraph_id_list,turn_id_list)
        
        
        return predicted_entity_labels   ,predicted_answer_type_labels    ,answer,answer_json_array
        
    def gen_coqa_answer(self,predicted_entity_labels,predicted_answer_type_labels,ewords_str,paragraph_id_list,turn_id_list):
        coqa_predictions = []
        answer_json_array=[]
        for i, (answer_type,entity,entity_name,paragraph_id,turn_id) in enumerate(zip(predicted_answer_type_labels, predicted_entity_labels,ewords_str,paragraph_id_list,turn_id_list)):
            answer_type=int(answer_type)
            if answer_type==0:
                prediction="YES"
            elif answer_type==1:
                prediction="NO"
            elif answer_type==3:
                prediction="UNKNOWN"
            else:
                prediction=predict_by_entity(entity.cpu().numpy(),entity_name)
            answer_json={"id":paragraph_id ,"turn_id":int(turn_id) ,"answer":prediction}
            answer_json_array.append(answer_json)
            coqa_predictions.append(prediction)
 
        return coqa_predictions,answer_json_array

def predict_by_entity(entity,ewords_str):
    entity_list=np.where(entity==1,ewords_str,"")
    entity_str=" ".join(entity_list).strip()
    if len(entity_str)>0:
        return entity_str 
    else:
        return "YES"

     