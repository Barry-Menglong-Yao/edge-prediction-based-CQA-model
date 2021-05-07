# edge-prediction-based-CQA-model

## Directory structure:

    main.py: 
        read args from cmd and start the process 
        call the seq2seq.py
    model/seq2seq.py:
        do the training, evaluation, testing in the minibatch.
        call model/cqa_model.py
    model: 
        the models class we use like BERT and our proposed model cqa_model.py  
    utils:
        the common utils used by multi layers or the whole model
    config.py:
        mutable configuration like epochs=5, output_dir
    data:
        data
    log:
        log 
    models:
        trained model parameters. save them here.
    output:
        other output like prediction.json
    requirement.txt:
        required packages
## Requirements
```bash
python == 3.6
pytorch == 1.5.0 
```     
## Dataset:
The dataset can be downloaded from [here](https://stanfordnlp.github.io/coqa/). Or you can download using the below command,
 ```bash
  mkdir data
  wget -P data https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json
  wget -P data https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json
```
The dataset needs to be preprocessed to make them usable for our model   
## How to run code:
        0, install the required packages in requirement.txt
        1, do the preprocessing. Check the following section 3) for details.
        2, put the obtained input file in data directory. Their path and name should be the following ones.
            data/coqa/train_eg.txt
            data/coqa/train_label.txt
            data/coqa/train_lower.txt
            data/coqa/train_type.txt
            data/coqa/dev/dev_eg_new.txt
            data/coqa/dev/dev_label_new.txt
            data/coqa/dev/lower.txt
            data/coqa/dev/dev_type.txt
            data/coqa/dev/dev_eval.txt
        3, put the coqa-dev-v1.0.json in data/coqa/dev directory. It is from CoQA website.
        4, run "python main.py". It will train. After training, it will test and give the final test result.

## How to do the preprocessing:
        1, Put the coqa-train-v1.0.json and coqa-dev-v1.0.json in data directory
        2, Set is_preprocessing = True and run main.py for preprocessing
        3, To use Bert embedding instead of GloVe call bert_embedding(params) in CoQADataset init.



   
    
    

    
