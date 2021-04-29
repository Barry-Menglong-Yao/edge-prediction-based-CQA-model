import torch 
import os




         
 
         

        
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
            for entity_line,  labels   in zip(f, label_f ):
                entity_num=entity_line.count(":")-entity_line.count("::")
                labels_num=labels.count("0")+labels.count("1")
                self.assertEqual(labels_num, entity_num)
                break

if __name__ == '__main__':
    unittest.main()