import torch
import torch.nn as nn
from transformers import *
from utils.data_utils import prepare_datasets
import os

MODELS = {'BERT':(BertModel,       BertTokenizer,       'bert-base-uncased')}

class ModelHandler():
	def __init__(self, config):
		self.config = config
		tokenizer_model = MODELS[config['model_name']]
		self.train_loader, self.dev_loader, tokenizer = prepare_datasets(config, tokenizer_model)

		self._n_dev_batches = len(self.dev_loader.dataset) // config['batch_size']
		self._n_train_batches = len(self.train_loader.dataset) // config['batch_size']
		if config['cuda']:
			self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	