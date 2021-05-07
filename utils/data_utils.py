import json
import io
from transformers import *
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from collections import Counter, defaultdict
import numpy as np
from random import shuffle
import math
import textacy.preprocessing.replace as rep
from tqdm import tqdm
import spacy
import csv
import pickle
import os.path
import torch
import re
from spacy.symbols import nsubj, VERB, dobj

# Destination name of each type of trainable file
WORD_EMBEDDING_FILE_NAME = "data/bert_word_embedding.pt"
WORD_ID_FILE_NAME = "data/bert_word_id.pt"

TRAIN_LOWER_FILE_NAME = "data/train_lower.txt"
TRAIN_EG_FILE_NAME = "data/train_eg.txt"
TRAIN_LABEL_FILE_NAME = "data/train_label.txt"
TRAIN_TYPE_FILE_NAME = "data/train_type.txt"
TRAIN_EVAL_FILE_NAME = "data/train/train_eval.txt"

NEW_TRAIN_EG_FILE_NAME = "data/train_eg_new.txt"
NEW_TRAIN_LABEL_FILE_NAME = "data/train_label_new.txt"

DEV_LOWER_FILE_NAME = "data/dev/dev_lower.txt"
DEV_EG_FILE_NAME = "data/dev/dev_eg.txt"
DEV_LABEL_FILE_NAME = "data/dev/dev_label.txt"
DEV_TYPE_FILE_NAME = "data/dev/dev_type.txt"
DEV_EVAL_FILE_NAME = "data/dev/dev_eval.txt"


NEW_DEV_EG_FILE_NAME = "data/dev/dev_eg_new.txt"
NEW_DEV_LABEL_FILE_NAME = "data/dev/dev_label_new.txt"

# store each word to corresponding word id and embedding
word_id = {}
word_embedding = {}

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

def findWholeWord(w):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

# split each context into multiple sentence
def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

# Prepare and save all type of trainable dataset
def prepare_datasets(config, tokenizer_model):
    print("Prepare dataset begin")
    tokenizer = []
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)       
    tokenizer = tokenizer_model[1].from_pretrained(tokenizer_model[2])

    trainset = CoQADataset(model, tokenizer, config['trainset'])
    devset = CoQADataset(model, tokenizer, config['devset'])

    return trainset, devset, tokenizer

def get_file_contents(filename, encoding='utf-8'):
    with io.open(filename, encoding=encoding) as f:
        content = f.read()
    f.close()
    return content


def read_json(filename, encoding='utf-8'):
    contents = get_file_contents(filename, encoding=encoding)
    return json.loads(contents)

def preprocess(text):
    text = ' '.join(text)
    temp_text = rep.replace_currency_symbols(text, replace_with = '_CUR_')
    temp_text = rep.replace_emails(temp_text, replace_with = '_EMAIL_')
    temp_text = rep.replace_emojis(temp_text, replace_with='_EMOJI_')
    temp_text = rep.replace_hashtags(temp_text, replace_with='_TAG_')
    temp_text = rep.replace_numbers(temp_text, replace_with='_NUMBER_')
    temp_text = rep.replace_phone_numbers(temp_text, replace_with = '_PHONE_')
    temp_text = rep.replace_urls(temp_text, replace_with = '_URL_')
    temp_text = rep.replace_user_handles(temp_text, replace_with = '_USER_')

    doc = nlp(temp_text)
    tokens = []
    for t in doc:
        tokens.append(t.text)
    return tokens

# Get embedding of a sentence using BERT    
def get_embedding(model, tokenizer, sentence):
    tokenized_text = tokenizer.tokenize(sentence)
    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    tokens_tensor = torch.tensor([indexed_tokens])
    with torch.no_grad():
        outputs = model(tokens_tensor)
    # can use last hidden state as word embeddings
    last_hidden_state = outputs[0]
    word_embed = last_hidden_state
    #print("embed 1. ", np.shape(word_embed_1), len(sentence))
  
    for tup in zip(tokenized_text, indexed_tokens, word_embed[0]):
      word_id[tup[0]] = tup[1]
      word_embedding[tup[0]] = tup[2]
    return word_embed

# Generate and save file 
def bert_embedding(dataset, model, tokenizer):

    paragraphs = []
    unique_word = {"a"}
    global word_embedding, word_id
    paragraphs_sentences = [[] for i in range(60000)]
    paragraphs_questions = [[] for i in range(60000)]
    paragraphs_answers = [[] for i in range(60000)]
    cnt = 0
    paragraph_id = -1
    # Put the model in "evaluation" mode,meaning feed-forward operation.
    model.eval()
    for paragraph in tqdm(dataset['data']):
      context = paragraph['context']
      #break

      print(paragraph['annotated_context'])

      paragraph_id += 1
      paragraphs.append(context)
      words = context.split()
      for word in words:
        unique_word.add(word)
      #continue

      sentence_list = split_into_sentences(context)
      for sentence in sentence_list:
        sentence = "[CLS] " + sentence + " [SEP]"
        paragraphs_sentences[paragraph_id].append(sentence)
        sentence_embedding = get_embedding(model, tokenizer, sentence)

      #question_answer = paragraph['qas']
      for qas in paragraph['qas']:
        question = qas['question']
        answer = qas['answer']
        paragraphs_questions[paragraph_id].append(question)
        paragraphs_answers[paragraph_id].append(answer)

        words = question.split()
        for word in words:
          unique_word.add(word)
        words = answer.split()
        for word in words:
          unique_word.add(word)  

        question_embedding = get_embedding(model, tokenizer, question)
        answer_embedding = get_embedding(model, tokenizer, answer)

        #print('qas.  ', question, answer)  
    print("Embedding done.  && LEN of unique word  ", len(unique_word))

    if os.path.isfile(WORD_EMBEDDING_FILE_NAME):
      word_id = torch.load(WORD_ID_FILE_NAME)
      word_embedding = torch.load(WORD_EMBEDDING_FILE_NAME)
      print("Number of total word  >>>>>>>>>. ", len(word_embedding))

    count = 0
    sentence = "[CLS]"
    for word in iter(unique_word):
      count = count + 1
      if count % 4000 == 0:
        print("Total and percentage. ", count, (count*100)/len(unique_word))
        #break
      if count % 150 == 0:
        sentence += " [SEP]"
        get_embedding(model, tokenizer, sentence)
        sentence = "[CLS]"

      sentence += " " + word  

    torch.save(word_embedding, WORD_EMBEDDING_FILE_NAME)   
    torch.save(word_id, WORD_ID_FILE_NAME)   

    # example to get embedding. Comment out next couple of line to get the word embedding of a sentence(test_sentence)
    '''
    test_sentence = "This is a test sentence."
    test_sentence = "CLS" + test_sentence + "SEP"
    tokenized_text = tokenizer.tokenize(sentence)
    for word in tokenized_text:
      word_embed = word_embedding[word]
      print("Word and embedding", word, word_embed)
    '''      

# Prepare eg content. Example of eg file is given in /data/coqa/example/train.eg
def get_eg(train_low, nlp):
    
    SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
    OBJECTS = ["dobj", "dative", "attr", "oprd"]
    ADJECTIVES = ["acomp", "advclmod", "xcomp", "rcmod", "poss"," possessive"]
    COMPOUNDS = ["compound"]
    PREPOSITIONS = ["prep"]
    text = train_low
    #print("Text ====.  ", text)
    sentence_list = split_into_sentences(text)
    doc = nlp(text)
    SOV = {}
    SOV [""] = ""

    cnt = 0
    unique_word ={""}
    for token in doc:
      unique_word.add(token.text)
      if token.text == 'eos':
        cnt+=1
      if token.text not in SOV:
        SOV[token.text] = ""
      noun = 0  
      if (token.tag_ == 'NN' or token.tag_ == 'NNS' or token.tag_ == 'NNP' or token.tag_ == 'NNPS') and token.text != 'eos':
         noun = 1
      if noun == 0:
        continue   
      sov = "3"
      if (token.dep_ =="nsubj" or token.dep_ == "nsubjpass" or token.dep_ ==  "csubj"
       or token.dep_ ==  "csubjpass" or token.dep_ ==  "agent" or token.dep_ ==  "expl"):
        sov = "1"
      if (token.dep_ == "dobj" or token.dep_ ==  "dative" or token.dep_ == "attr" or token.dep_ == "oprd"):
        sov = "2" 
      if len(SOV[token.text]) >0:
        SOV[token.text] += '|'
      SOV[token.text] += str(cnt) + '-' + sov

    #print("LEN of unique word  ", len(unique_word), len(sentence_list), cnt)
    str1 = ""
    for a, value in SOV.items():
      if(len(value)) == 0:
        continue
      str1 += a + ':'
      str1 += value +" "
    #print(str1)
    return str1

#Prepare lower file. Example of lower file is given in /data/coqa/example/train.lower
def prepare_lower(LOWER_FILE_NAME, EG_FILE_NAME, dataset):
    #dataset = read_json(filename)
    nlp = spacy.load('en_core_web_sm')
    train_lower_list = []
    train_lower_str = ""
    train_eg_str = ""
    for paragraph in tqdm(dataset['data']):
          context = paragraph['context']
          sentence_list = split_into_sentences(context)

          train_lower = ""
          
          for sentence in sentence_list:
            if len(train_lower)>0:
              train_lower += ' <eos> '
            train_lower += sentence
            #self.paragraphs_sentences[paragraph_id].append(sentence)

          prev_ans = ""  
          for qas in paragraph['qas']:
            if len(prev_ans)>0:
              train_lower += ' <eos> ' + prev_ans 
            question = qas['question']
            train_lower += ' <eos> ' + question
            prev_ans = qas['answer']

            train_lower_list.append(train_lower)
            train_lower_str += train_lower + '\n'

            #print("############################.      ", train_lower)
          #print("Lower------------>>>> ", train_lower)
          #break

    #torch.save(train_lower_str, TRAIN_LOWER_FILE_NAME)
    f = open(LOWER_FILE_NAME, "w")
    f.write(train_lower_str)
    f.close()       

    total_len = train_lower_str.count('\n')
    print("Train lower done.  ", total_len)
    cnt = 1
    for line in train_lower_str.split('\n'):
      train_eg = get_eg(line, nlp)
      train_eg_str += train_eg + '\n'
      #'''
      if cnt % 100 == 0:
        print("Cnt percentage =====.    ", (cnt*100)/total_len, train_eg.count(" "))
        #break
      cnt +=1  
      #print(line)
      #'''
    print("Train eg done.  ", train_eg_str.count('\n'), cnt)

    f = open(EG_FILE_NAME, "w")
    f.write(train_eg_str)
    f.close() 
    #torch.save(train_eg_str, TRAIN_EG_FILE_NAME)  

#Prepare label file. Example of label file is given in /data/coqa/example/train.label
def prepare_label(LABEL_FILE_NAME, dataset):
    nlp = spacy.load('en_core_web_sm')
    train_label_str = ""
    for paragraph in tqdm(dataset['data']):
          context = paragraph['context']
          sentence_list = split_into_sentences(context)

          train_lower = ""
          
          for sentence in sentence_list:
            if len(train_lower)>0:
              train_lower += ' <eos> '
            train_lower += sentence
            #self.paragraphs_sentences[paragraph_id].append(sentence)

          prev_ans = ""
            
          for qas in paragraph['qas']:
            if len(prev_ans)>0:
              train_lower += ' <eos> ' + prev_ans 
            question = qas['question']
            train_lower += ' <eos> ' + question
            cur_ans = qas['answer']
            
            doc = nlp(train_lower)
            entity_label = ""
            st = {""}

            ENTITY = {}
            ENTITY[""] = ""
            for token in doc:
              noun = 0 
              if (token.tag_ == 'NN' or token.tag_ == 'NNS' or token.tag_ == 'NNP' or token.tag_ == 'NNPS') and token.text != 'eos':
                noun = 1
              if noun == 0:
                continue 
              st.add(token.text)
              if (' ' + token.text + ' ') in (' ' + cur_ans + ' '):
                ENTITY[token.text] = '1'
                #print("token text. ",  token.text, cur_ans)
              else:
                ENTITY[token.text] = '0'
            prev_ans = cur_ans

            for key, value in ENTITY.items():
              entity_label += value + " "
            entity_label = entity_label[:-1]  
            #print("len ====.   ", entity_label.count(" "), entity_label)  
            train_label_str += entity_label + '\n'
          
    
          #break          
    f = open(LABEL_FILE_NAME, "w")
    f.write(train_label_str)
    f.close() 

#Prepare type file. Example of type file is given in /data/coqa/example/train.type
def prepare_type(TYPE_FILE_NAME, dataset):
    nlp = spacy.load('en_core_web_sm')
    train_type_str = ""
    for paragraph in tqdm(dataset['data']):
          context = paragraph['context']
          sentence_list = split_into_sentences(context)

          train_lower = ""
          
          for sentence in sentence_list:
            if len(train_lower)>0:
              train_lower += ' <eos> '
            train_lower += sentence
            #self.paragraphs_sentences[paragraph_id].append(sentence)

          prev_ans = ""
            
          for qas in paragraph['qas']:
            if len(prev_ans)>0:
              train_lower += ' <eos> ' + prev_ans 
            question = qas['question']
            train_lower += ' <eos> ' + question
            cur_ans = qas['answer']
            if cur_ans.lower() == 'yes':
              train_type_str += '0\n'
            elif cur_ans.lower() == 'no':
              train_type_str += '1\n'
            elif cur_ans.lower() == 'unknown':
              train_type_str += '3\n'
            else: 
              train_type_str += '2\n'
          #break
      

    f = open(TYPE_FILE_NAME, "w")
    f.write(train_type_str)
    f.close()

def verify(EG_FILE_NAME, LABEL_FILE_NAME):
  f = open(EG_FILE_NAME, "r")
  f1 = open(LABEL_FILE_NAME, "r")

  cnt = 0
  #'''
  for a,b in zip(f, f1):
    cnt += 1
    if a.count(":") != b.count(" "):
      print(cnt, a.count(":"), b.count(" "), a)
  #'''
  print("Checking done.  ")

def generate_new(EG_FILE_NAME, LABEL_FILE_NAME, NEW_EG_FILE_NAME, NEW_LABEL_FILE_NAME):
  f = open(EG_FILE_NAME, "r")
  f1 = open(LABEL_FILE_NAME, "r")

  cnt = 0
  train_eg_str = "" 
  train_label_str = ""
  for a,b in zip(f, f1):
      entity_list = a.split()
      label_list = b.split()
      entity_sentence = ""
      label_sentence = ""
      for entity, label in zip(entity_list, label_list):
        entity_str = ""
        label_str = ""
        #print(entity)
        if entity.count(":") == 1:
          entity_str += entity + " "
          label_str+= label + " "
        entity_sentence += entity_str 
        label_sentence += label_str 
      cnt +=1
      train_eg_str += entity_sentence + "\n"
      train_label_str += label_sentence + "\n"

      #if a.count(" ") != b.count(" "):
      #print(cnt, a.count(" "), b.count(" "), len(entity_list), entity_sentence.count(" "), a.count(":"), entity_sentence.count(":"))
  sen = train_eg_str.split("\n")
  f = open(NEW_EG_FILE_NAME, "w")
  f.write(train_eg_str)
  f.close()
  f = open(NEW_LABEL_FILE_NAME, "w")
  f.write(train_label_str)
  f.close()
  print(sen[1768])

def gen_eval_file(dataset, EVAL_FILE_NAME):
  eval_string = ""
  cnt = 0
  total_len = len(dataset['data'])
  for paragraph in tqdm(dataset['data']):
    cnt += 1
    if cnt % 100 == 0:
      print("Cnt percentage =====.    ", (cnt*100)/total_len)

    context = paragraph['context']
    paragraph_id = paragraph['id']

    for qas in paragraph['qas']:
      turn_id = qas['turn_id']
      e_str = paragraph_id + ":" + str(turn_id)
      eval_string += e_str + "\n"
    #print(paragraph['id'], paragraph)
    #break
  f = open(EVAL_FILE_NAME, "w")
  f.write(eval_string)
  f.close()  

class CoQADataset(Dataset):
    """CoQA dataset."""

    def __init__(self, model, tokenizer, filename):
        #timer = Timer('Load %s' % filename)
        if filename == "data/coqa.dev.json":
          EG_FILE_NAME = DEV_EG_FILE_NAME
          LABEL_FILE_NAME = DEV_LABEL_FILE_NAME
          NEW_EG_FILE_NAME = NEW_DEV_EG_FILE_NAME
          NEW_LABEL_FILE_NAME = NEW_DEV_LABEL_FILE_NAME
          LOWER_FILE_NAME = DEV_LOWER_FILE_NAME
          TYPE_FILE_NAME = DEV_TYPE_FILE_NAME
          EVAL_FILE_NAME = DEV_EVAL_FILE_NAME
        else:
          EG_FILE_NAME = TRAIN_EG_FILE_NAME
          LABEL_FILE_NAME = TRAIN_LABEL_FILE_NAME
          NEW_EG_FILE_NAME = NEW_TRAIN_EG_FILE_NAME
          NEW_LABEL_FILE_NAME = NEW_TRAIN_LABEL_FILE_NAME
          LOWER_FILE_NAME = TRAIN_LOWER_FILE_NAME
          TYPE_FILE_NAME = TRAIN_TYPE_FILE_NAME
          EVAL_FILE_NAME = TRAIN_EVAL_FILE_NAME

        dataset = read_json(filename)

        #Comment out next line to use bert embedding 
        #bert_embedding(dataset, model, tokenizer)

        #Prepare all of the 4 types(lower, label, type, eg) of input data for our model
        prepare_lower(TRAIN_LOWER_FILE_NAME, TRAIN_EG_FILE_NAME, dataset)
        prepare_label(LABEL_FILE_NAME, dataset) 
        prepare_type(TYPE_FILE_NAME, dataset)  
        generate_new(EG_FILE_NAME, LABEL_FILE_NAME, NEW_EG_FILE_NAME, NEW_LABEL_FILE_NAME)
        
        # Generate the file for evaluation
        gen_eval_file(dataset, EVAL_FILE_NAME)
        # Verify if the preprocessd data are well enough to train
        verify(NEW_EG_FILE_NAME, NEW_LABEL_FILE_NAME)
        

if __name__=='__main__':
    from transformers import *
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = CoQADataset('coqa.train.json')

    

