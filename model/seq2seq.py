import re
import torch
import numpy as np
import math
import torch.nn as nn
import time
import subprocess
import torch.nn.functional as F
from model.generator import Beam
from data.data import DocField, DocDataset, DocIter
from utils.config import is_cqa_task
from model.graph_model import * 
from model.cqa_model import * 
import itertools
from sklearn.metrics import accuracy_score
import logging
from datetime import datetime
from sklearn.metrics import f1_score
from utils.timer import Timer
from utils.evaluate import *
timestr=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
logging.basicConfig(level=logging.DEBUG,filename="log/training_"+timestr)



def beam_search_pointer(args, model, src_and_len, doc_num, ewords_and_len, elocs):
    sentences, _, dec_init, keys,entity_h,_ = model.encode(src_and_len, doc_num,  ewords_and_len, elocs)

    document = sentences.squeeze(0)
    T, H = document.size()

    W = args.beam_size

    prev_beam = Beam(W)
    prev_beam.candidates = [[]]
    prev_beam.scores = [0]

    target_t = T - 1

    f_done = (lambda x: len(x) == target_t)

    valid_size = W
    hyp_list = []

    for t in range(target_t):
        candidates = prev_beam.candidates
        if t == 0:
            # start
            dec_input = sentences.new_zeros(1, 1, H)
            pointed_mask = sentences.new_zeros(1, T).byte()
        else:
            index = sentences.new_tensor(list(map(lambda cand: cand[-1], candidates))).long()
            # beam 1 H
            dec_input = document[index].unsqueeze(1)

            pointed_mask[torch.arange(index.size(0)), index] = 1

        dec_h, dec_c, log_prob = model.step(dec_input, dec_init, keys, pointed_mask)

        next_beam = Beam(valid_size)
        done_list, remain_list = next_beam.step(-log_prob, prev_beam, f_done)
        hyp_list.extend(done_list)
        valid_size -= len(done_list)

        if valid_size == 0:
            break

        beam_remain_ix = src_and_len[0].new_tensor(remain_list)
        dec_h = dec_h.index_select(1, beam_remain_ix)
        dec_c = dec_c.index_select(1, beam_remain_ix)
        dec_init = (dec_h, dec_c)

        pointed_mask = pointed_mask.index_select(0, beam_remain_ix)

        prev_beam = next_beam

    score = dec_h.new_tensor([hyp[1] for hyp in hyp_list])
    sort_score, sort_ix = torch.sort(score)
    output = []
    for ix in sort_ix.tolist():
        output.append((hyp_list[ix][0], score[ix].item()))
    best_output = output[0][0]

    the_last = list(set(list(range(T))).difference(set(best_output)))
    best_output.append(the_last[0])

    return best_output


def print_params(model):
    print('total parameters:', sum([np.prod(list(p.size())) for p in model.parameters()]))


def train(args, train_iter, dev, fields, checkpoint):
    if is_cqa_task():
        model = CqaNet(args)
    else:
        model = PointerNet(args)
    # 
    model.cuda()

    DOC, ORDER, GRAPH = fields
    print('1:', DOC.vocab.itos[1])
    model.load_pretrained_emb(DOC.vocab.vectors)

    print_params(model)
    print(model)

    wd = 1e-5
    opt = torch.optim.Adadelta(model.parameters(), lr=args.lr, rho=0.95, weight_decay=wd)

    best_answer_type_score = -np.inf
    best_entity_score = -np.inf
    best_iter = 0
    offset = 0

    criterion = nn.NLLLoss(reduction='none')
    model.equip(criterion)

    start = time.time()

    early_stop = args.early_stop

    test_data = DocDataset(path=args.test, text_field=DOC, order_field=ORDER, graph_field=GRAPH)
    test_real = DocIter(test_data, 1, device='cuda', batch_size_fn=None,
                        train=False, repeat=False, shuffle=False, sort=False)

    fake_epc=-1
    is_validate_before_train=False
    timer=Timer()
    if is_validate_before_train:
        validate(args,   dev,  checkpoint,model,DOC,fake_epc,best_answer_type_score,best_iter,is_validate_before_train,best_entity_score,timer)
    
    
    for epc in range(args.maximum_steps):
        for iters, batch in enumerate(train_iter):
            model.train()

            model.zero_grad()

            t1 = time.time()

            loss = model(batch.doc, batch.order, batch.doc_len, batch.e_words, batch.elocs,batch.labels,batch.answer_types)

            loss.backward()
            opt.step()

            t2 = time.time()
            # print('epc:{} iter:{} loss:{:.2f} t:{:.2f} lr:{:.1e}'.format(epc, iters + 1, loss, t2 - t1,
            #                                                              opt.param_groups[0]['lr']))

        if epc < 5:
            print(f"finish epoch {epc}, lastest loss{loss}")
            continue

        best_answer_type_score,best_iter,best_entity_score,is_early_stop=validate(args,   dev,  checkpoint,model,DOC,epc,best_answer_type_score,best_iter,False,best_entity_score,timer)
        if is_early_stop:
            break

    print('\n*******Train Done********{}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    minutes = (time.time() - start) // 60
    if minutes < 60:
        print('best:{:.2f}, iter:{}, time:{} mins, lr:{:.1e}, '.format(best_answer_type_score, best_iter, minutes,
                                                                       opt.param_groups[0]['lr']))
    else:
        hours = minutes / 60
        print('best:{:.2f}, iter:{}, time:{:.1f} hours, lr:{:.1e}, '.format(best_answer_type_score, best_iter, hours,
                                                                            opt.param_groups[0]['lr']))

    checkpoint = torch.load('{}/{}.best.pt'.format(args.model_path, args.model), map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    with torch.no_grad():
        if args.loss:
            entity_score = valid_model(args, model, dev, DOC, 'loss')
            print('epc:{}, loss:{:.2f} best:{:.2f}\n'.format(epc, entity_score, best_answer_type_score))
        else:
            entity_acc, answer_type_acc, ktau, pm  = valid_model(args, model, dev, DOC)
            print('test entity_acc:{:.4%} answer_type_acc:{:.2%} ktau:{:.4f} pm:{:.2%}'.format(entity_acc, answer_type_acc, ktau, pm))
        


def remaining_time(timer,epc,best_epc, args):
    
    remaining_time_str=f'{timer.remains(args.early_stop+best_epc,epc)} or {timer.remains(args.maximum_steps,epc)}'
    return remaining_time_str

def validate(args,   dev,  checkpoint,model,DOC,epc,best_score,best_iter,
is_validate_before_train,best_entity_score,timer):
    early_stop = args.early_stop
    is_early_stop=False
    is_validate_by_coqa_f1=True
    with torch.no_grad():
        print('valid..............')
        if args.loss:
            entity_score = valid_model(args, model, dev, DOC, 'loss')
            print('epc:{}, loss:{:.2f} best:{:.2f}\n'.format(epc, entity_score, best_score))
        else:
            entity_acc ,answer_type_acc, overall_coqa_f1, _ = valid_model(args, model, dev, DOC)
            print( f'epc:{epc}, val f1: overall_coqa_f1:{overall_coqa_f1},  answer_type :{answer_type_acc}, answer_type_best:{best_score}, entity:{entity_acc}, entity_best:{best_entity_score}, {remaining_time(timer,epc,best_iter, args)} '   )
            logging.debug(f'epc:{epc}, val f1: overall_coqa_f1:{overall_coqa_f1},  answer_type :{answer_type_acc}, answer_type_best:{best_score}, entity:{entity_acc}, entity_best:{best_entity_score}, {remaining_time(timer,epc,best_iter, args)} ' )
        if is_validate_by_coqa_f1:
            best_score,best_iter=check_coqa_f1(overall_coqa_f1, best_score,best_iter,epc,model,args,is_validate_before_train)
             
        else:
            best_score,best_entity_score,best_iter=check_f1(answer_type_acc,entity_acc,best_score,best_entity_score,is_validate_before_train,epc,model,args)
        if early_stop and (epc - best_iter) >= early_stop:
            print('early stop at epc {}'.format(epc))
            is_early_stop=True
    return best_score,best_iter,best_entity_score,is_early_stop


def check_f1(answer_type_acc,entity_acc,best_score,best_entity_score,is_validate_before_train,epc,model,args):
    if  answer_type_acc > best_score and entity_acc>best_entity_score :
        best_score = answer_type_acc
        best_entity_score=entity_acc
        best_iter = epc

        if is_validate_before_train!=True:
            print('save best model best.pt at epc={}'.format(epc))
            checkpoint = {'model': model.state_dict(),
                            'args': args,
                            'loss': best_score}
            torch.save(checkpoint, '{}/{}.best.pt'.format(args.model_path, args.model))
    elif answer_type_acc > best_score  :
        best_score = answer_type_acc
        best_iter = epc

        if is_validate_before_train!=True:
            print('save best model answer_type_best.pt at epc={}'.format(epc))
            checkpoint = {'model': model.state_dict(),
                            'args': args,
                            'loss': best_score}
            torch.save(checkpoint, '{}/{}.answer_type_best.pt'.format(args.model_path, args.model))
    elif   entity_acc>best_entity_score:
        best_entity_score = entity_acc
        best_iter = epc

        if is_validate_before_train!=True:
            print('save best model entity_best.pt at epc={}'.format(epc))
            checkpoint = {'model': model.state_dict(),
                            'args': args,
                            'loss': best_entity_score}
            torch.save(checkpoint, '{}/{}.entity_best.pt'.format(args.model_path, args.model))
    return best_score,best_entity_score,best_iter



def check_coqa_f1(coqa_f1, best_coqa_f1,best_iter,epc,model,args,is_validate_before_train):
    if   coqa_f1>best_coqa_f1:
        best_coqa_f1 = coqa_f1
        best_iter = epc

        if is_validate_before_train!=True:
            print('save best model coqa_best.pt at epc={}'.format(epc))
            checkpoint = {'model': model.state_dict(),
                            'args': args,
                            'loss': best_coqa_f1}
            torch.save(checkpoint, '{}/{}.coqa_best.pt'.format(args.model_path, args.model))
    return  best_coqa_f1,best_iter


def valid_model(args, model, dev, field, dev_metrics=None, shuflle_times=1):
    model.eval()

    if dev_metrics == 'loss':
        total_score = []
        number = 0

        for iters, dev_batch in enumerate(dev):
            loss = model(dev_batch.doc, dev_batch.order, dev_batch.doc_len, dev_batch.e_words, dev_batch.elocs,dev_batch.labels)
            n = dev_batch.order[0].size(0)
            batch_loss = -loss.item() * n
            total_score.append(batch_loss)
            number += n

        return sum(total_score) / number
    else:
        if is_cqa_task():
            return valid_cqa_model_by_coqa_acc(args, model, dev, field, dev_metrics , shuflle_times)
        else:
            return valid_sentence_ordering_model(args, model, dev, field, dev_metrics , shuflle_times)
        


def valid_cqa_model(args, model, dev, field, dev_metrics , shuflle_times):
    entity_truth = []
    entity_predicted = []
    answer_type_truth = []
    answer_type_predicted = []
    for j, dev_batch in enumerate(dev):
        entity_truth.append( dev_batch.labels[0].view(-1).tolist() )   
        answer_type_truth.append(dev_batch.answer_types.view(-1).tolist())
        predicted_entity_labels,predicted_answer_type_labels   = model.predict(dev_batch.doc, dev_batch.doc_len, dev_batch.e_words, dev_batch.elocs )
        entity_predicted.append(predicted_entity_labels.view(-1).tolist())
        answer_type_predicted.append(predicted_answer_type_labels.view(-1).tolist())
    entity_acc = accuracy_score( list(itertools.chain.from_iterable(entity_truth)),
                            list(itertools.chain.from_iterable(entity_predicted))   )
    answer_type_acc = accuracy_score( list(itertools.chain.from_iterable(answer_type_truth)),
                            list(itertools.chain.from_iterable(answer_type_predicted))   )
    prediction_distribution_str=gen_prediction_distribution(answer_type_predicted,entity_predicted)
    entity_f1=f1_score(list(itertools.chain.from_iterable(entity_truth)),
                                list(itertools.chain.from_iterable(entity_predicted))  , 
                                average='macro')
    answer_type_f1=f1_score(list(itertools.chain.from_iterable(answer_type_truth)),
                                list(itertools.chain.from_iterable(answer_type_predicted))   , 
                            average='macro')      
    logging.debug(f'entity_acc:{entity_acc},  answer_type_acc:{    answer_type_acc},entity_f1:{ entity_f1},answer_type_f1:{answer_type_f1},prediction_distribution_str:{prediction_distribution_str}' )
    return entity_f1 ,answer_type_f1,0, 0



def valid_cqa_model_by_coqa_acc(args, model, dev, field, dev_metrics , shuflle_times):
    entity_truth = []
    entity_predicted = []
    answer_type_truth = []
    answer_type_predicted = []
    coqa_prediction_json_array=[]
    for j, dev_batch in enumerate(dev):
        entity_truth.append( dev_batch.labels[0].view(-1).tolist() )   
        answer_type_truth.append(dev_batch.answer_types.view(-1).tolist())
        predicted_entity_labels,predicted_answer_type_labels,coqa_predictions,answer_json_array   = model.predict_coqa_answer(dev_batch.doc, dev_batch.doc_len, dev_batch.e_words, dev_batch.elocs,dev_batch.e_words_str,dev_batch.paragraph_ids,dev_batch.turn_ids )
        entity_predicted.append(predicted_entity_labels.view(-1).tolist())
        answer_type_predicted.append(predicted_answer_type_labels.view(-1).tolist())
        coqa_prediction_json_array.append(answer_json_array[0])
    entity_acc = accuracy_score( list(itertools.chain.from_iterable(entity_truth)),
                            list(itertools.chain.from_iterable(entity_predicted))   )
    answer_type_acc = accuracy_score( list(itertools.chain.from_iterable(answer_type_truth)),
                            list(itertools.chain.from_iterable(answer_type_predicted))   )
    prediction_distribution_str=gen_prediction_distribution(answer_type_predicted,entity_predicted)
    entity_f1=f1_score(list(itertools.chain.from_iterable(entity_truth)),
                                list(itertools.chain.from_iterable(entity_predicted))  , 
                                average='macro')
    answer_type_f1=f1_score(list(itertools.chain.from_iterable(answer_type_truth)),
                                list(itertools.chain.from_iterable(answer_type_predicted))   , 
                            average='macro')      
    scores=test_coqa_acc(coqa_prediction_json_array)
    overall_coqa_f1=scores["overall"]['f1']
 
    logging.debug(f'entity_acc:{entity_acc},  answer_type_acc:{    answer_type_acc},entity_f1:{ entity_f1},answer_type_f1:{answer_type_f1},prediction_distribution_str:{prediction_distribution_str},overall_coqa_f1:{overall_coqa_f1}' )
    return entity_f1 ,answer_type_f1,overall_coqa_f1, 0

 

def gen_prediction_distribution( answer_type_truth,entity_predicted):
    hist,_=np.histogram(list(itertools.chain.from_iterable(answer_type_truth)), bins=[0, 1, 2, 3,4])
    hist2,_=np.histogram(list(itertools.chain.from_iterable(entity_predicted)), bins=[0, 1, 2])
    return f'answer type prediction distribution: {hist}, entity prediction distribution:{hist2}'
        

def check_prediction(j,predicted_answer_type_labels,predicted_entity_labels):
    if int(predicted_answer_type_labels)!=2:
        print(f'in {j}, predict answer_type{int(predicted_answer_type_labels)}')
        logging.debug(f'in {j}, predict answer_type{int(predicted_answer_type_labels)}')
    if predicted_entity_labels.squeeze().sum()>0:
        print(f'in {j}, predict predicted_entity_labels{predicted_entity_labels.squeeze().tolist()}')
        logging.debug(f'in {j}, predict predicted_entity_labels{predicted_entity_labels.squeeze().tolist()}')
    

def valid_sentence_ordering_model(args, model, dev, field, dev_metrics , shuflle_times):
    f = open(args.writetrans, 'w')

    if args.beam_size != 1:
        print('beam search with beam', args.beam_size)

    best_acc = []
    for epc in range(shuflle_times):
        truth = []
        predicted = []

        for j, dev_batch in enumerate(dev):
            tru = dev_batch.order[0].view(-1).tolist()
            truth.append(tru)

            if len(tru) == 1:
                pred = tru
            else:
                pred = beam_search_pointer(args, model, dev_batch.doc, dev_batch.doc_len, dev_batch.e_words, dev_batch.elocs)

            predicted.append(pred)
            print('{}|||{}'.format(' '.join(map(str, pred)), ' '.join(map(str, truth[-1]))),
                    file=f)

        right, total = 0, 0
        pmr_right = 0
        taus = []
        # pm
        pm_p, pm_r = [], []
        import itertools

        from sklearn.metrics import accuracy_score

        for t, p in zip(truth, predicted):
            if len(p) == 1:
                right += 1
                total += 1
                pmr_right += 1
                taus.append(1)
                continue

            eq = np.equal(t, p)
            right += eq.sum()
            total += len(t)

            pmr_right += eq.all()

            # pm
            s_t = set([i for i in itertools.combinations(t, 2)])
            s_p = set([i for i in itertools.combinations(p, 2)])
            pm_p.append(len(s_t.intersection(s_p)) / len(s_p))
            pm_r.append(len(s_t.intersection(s_p)) / len(s_t))

            cn_2 = len(p) * (len(p) - 1) / 2
            pairs = len(s_p) - len(s_p.intersection(s_t))
            tau = 1 - 2 * pairs / cn_2
            taus.append(tau)

        # acc = right / total

        acc = accuracy_score(list(itertools.chain.from_iterable(truth)),
                                list(itertools.chain.from_iterable(predicted)))

        best_acc.append(acc)

        pmr = pmr_right / len(truth)
        taus = np.mean(taus)

        pm_p = np.mean(pm_p)
        pm_r = np.mean(pm_r)
        pm = 2 * pm_p * pm_r / (pm_p + pm_r)

        print('acc:', acc)

    f.close()
    acc = max(best_acc)
    return acc, pmr, taus, pm

def decode(args, test_real, fields, checkpoint):
    with torch.no_grad():
        if is_cqa_task():
            model = CqaNet(args)
        else:
            model = PointerNet(args)
        model.cuda()

        print('load parameters')
        model.load_state_dict(checkpoint['model'])
        DOC, ORDER = fields
        entity_acc ,answer_type_acc, overall_coqa_f1, _ =  valid_model(args, model, test_real, DOC)
        print('test entity_acc:{:.2%} answer_type_acc:{:.2%} overall_coqa_f1:{:.2f}'.format(entity_acc, answer_type_acc, overall_coqa_f1))


