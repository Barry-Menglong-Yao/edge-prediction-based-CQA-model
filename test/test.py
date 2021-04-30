import torch 
import os 



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

if __name__ == '__main__':
    # unittest.main()
    generate_part_data()