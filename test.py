import torch 
import os 
from torch import nn as nn
import numpy as np
from utils.train_helper import * 

def keep_K_lines(file_path,end,start=0):
    with open(file_path, "r") as f:
        lines = f.readlines()
    with open(file_path , "w") as f:
        for i  in range(start,end):
            f.write(lines[i])
                
         
 
         

        
# test_load()


#line_num ==
#entity_num==label_num




import unittest

class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

    
    def test_entity_equal_label(self):

        parent_directory="data/coqa"
        with open(parent_directory+'/train_eg.txt', 'r') as f,open(parent_directory+'/train_label.txt', 'r') as label_f:
            i=1
            for entity_line,  labels   in zip(f, label_f ):
                entity_num=entity_line.count(":")-entity_line.count("::")
                labels_num=labels.count("0")+labels.count("1")
                if labels_num!= entity_num:
                    # if entity_line.count(" ") != labels.count(" "):
                    print(f"other wrong in {i}, entity_num:{entity_num} ,labels_num:{labels_num}" )
                    print(f"entity:{entity_line} ,label:{labels}" )
             
                        
                # self.assertEqual(labels_num,entity_num)
                i+=1
                

def generate_part_data():
    start=1766
    K=2069
    keep_K_lines("data/coqa/31070/train_eg.txt",K,start)
    keep_K_lines("data/coqa/31070/train_label.txt",K,start)
    keep_K_lines("data/coqa/31070/train_lower.txt",K,start)
    keep_K_lines("data/coqa/31070/train_type.txt",K,start)

def weighted_loss():
    m = nn.LogSoftmax(dim=1)
    class_weight = torch.tensor([1,4], dtype=torch.float32)
    loss = nn.NLLLoss(weight= class_weight,reduction='none'   )
    # loss = nn.NLLLoss(reduction='none'   )
    # input is of size N x C = 3 x 5
    input = torch.tensor([[0.75, 0.25], [0.9, 0.1],[0.6, 0.4],[0.25, 0.75]], dtype=torch.float32,requires_grad=True) 
    # each element in target has to have 0 <= value < C
    target = torch.tensor([0,0,1,1])
    log_input=m(input)
    output = loss(log_input, target)
    # output.backward()
    print(output)

def cross_loss():
    target = torch.ones([1, 2], dtype=torch.float32)  # 2 classes, batch size = 3
    output = torch.full([1, 2], 0.5)  # A prediction (logit)
    pos_weight = torch.ones([2])  # All weights are equal to 1
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss=criterion(output, target)  # -log(sigmoid(1.5))
    print(loss)



def softmax():
    m = nn.Softmax(dim=0)
    input = torch.randn(2, 3)
    output = m(input)
    print(output)

 
if __name__ == '__main__':
    # unittest.main()
    # weighted_loss()
    samples_per_class=np.array([25,400])
    b_labels=torch.tensor([[0,1],[1,0],[0,1]])
    get_weights_inverse_num_of_samples(2,samples_per_class)