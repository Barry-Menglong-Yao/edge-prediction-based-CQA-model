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


word_id = {}
word_embedding = {}

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

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

class CoQADataset(Dataset):
    """CoQA dataset."""

    def __init__(self, model, tokenizer, filename):
        #timer = Timer('Load %s' % filename)
        self.filename = filename
        self.paragraphs = []
        self.paragraphs_sentences = [[] for i in range(60000)]
        self.paragraphs_questions = [[] for i in range(60000)]
        self.paragraphs_answers = [[] for i in range(60000)]

        dataset = read_json(filename)

        print("File name in init. --------------------     ", self.filename, len(dataset['data']))

        cnt = 0
        paragraph_id = -1
        # Put the model in "evaluation" mode,meaning feed-forward operation.
        model.eval()
        for paragraph in tqdm(dataset['data']):
          context = paragraph['context']
          paragraph_id += 1
          self.paragraphs.append(context)

          sentence_list = split_into_sentences(context)
          for sentence in sentence_list:
            sentence = "[CLS] " + sentence + " [SEP]"
            self.paragraphs_sentences[paragraph_id].append(sentence)
            sentence_embedding = get_embedding(model, tokenizer, sentence)

          #question_answer = paragraph['qas']
          for qas in paragraph['qas']:
            question = qas['question']
            answer = qas['answer']
            self.paragraphs_questions[paragraph_id].append(question)
            self.paragraphs_answers[paragraph_id].append(answer)

            question_embedding = get_embedding(model, tokenizer, question)
            answer_embedding = get_embedding(model, tokenizer, answer)

            #print('qas.  ', question, answer)  
        print("Embedding done")

        # example to get embedding. Comment out next couple of line to get the word embedding of a sentence(test_sentence)
        '''
        test_sentence = "This is a test sentence."
        test_sentence = "CLS" + test_sentence + "SEP"
        tokenized_text = tokenizer.tokenize(sentence)
        for word in tokenized_text:
          word_embed = word_embedding[word]
          print("Word and embedding", word, word_embed)

        '''  

if __name__=='__main__':
    from transformers import *
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = CoQADataset('coqa.train.json')

    

