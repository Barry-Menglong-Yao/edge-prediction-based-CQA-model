# edge-prediction-based-CQA-model




   
    
    Directory structure:
    layers:
        encoder: the encoding layer. 
            Input: paragraph, conversation history, current question.
            output: paragraph, conversation history, current question with embeddings of each words
        reasoner: the reasoning layer.
            Input: the output of encoder layer
            output: evidence with embeddings of each words
        predicter: the prediction layer.
            Input: the output of reasoner layer
            output: answer (answer type, start position, end position)
        interface: define the output of each layer
    models: 
        the models class we use like BERT and our proposed model. 
        proposed_model.py will call functions of layer.py
    trainer.py:
        do the training, evaluation, testing in the minibatch.
        call proposed_model.py
    main.py: 
        read args from cmd and start the process 
        call the trainer.py
    utils:
        the common utils used by multi layers or the whole model
    config.py:
        mutable configuration like epochs=5, output_dir
    data:
        data
    log:
        log 
    model:
        trained model parameters. save them here.
    requirement.txt:
        required packages

    
