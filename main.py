import argparse
from model_handler import ModelHandler

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--trainset', type = str, default = 'data/coqa.train.json', help = 'training dataset file')
parser.add_argument('--devset', type = str, default = 'data/coqa.dev.json', help = 'development dataset file')
parser.add_argument('--model_name', type = str, default = 'BERT', help = '[BERT|RoBERTa|DistilBERT|SpanBERT]')
parser.add_argument('--cuda', type = str2bool, default = True, help = 'use gpu or not')

args = vars(parser.parse_args())

# TODO: cuda check

handler = ModelHandler(args)
handler.train()