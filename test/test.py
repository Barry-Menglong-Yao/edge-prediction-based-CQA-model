import torch 
import os



def test_load():
    with open('data/coqa/train_eg.txt', 'r') as f:
        content = f.readlines()
        print(content.count("\n"))
        print(len(content))
        # print(content[len(content)-1])


    with open('data/coqa/train_lower.txt', 'r') as f:
        content = f.readlines()
        print(content.count("\n"))
        print(len(content))
        # print(content[len(content)-1])

    with open('data/coqa/train_label.txt', 'r') as f:
        content = f.readlines()
        print(content.count("\n"))
        print(len(content))
        # print(content[len(content)-1])

    with open('data/coqa/train_type.txt', 'r') as f:
        content = f.readlines()
        print(content.count("\n"))
        print(len(content))

        # print(content[len(content)-1])
test_load()


#line_num ==
#entity_num==label_num