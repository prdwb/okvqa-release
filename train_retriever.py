#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function

# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="3"

import argparse
import logging
import os
import random
import glob
import timeit
import json
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F 

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import pytrec_eval
import faiss

from transformers import WEIGHTS_NAME, BertConfig, BertTokenizer
from transformers import LxmertConfig, LxmertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from retriever_utils import RetrieverDataset, GenPassageRepDataset
from retriever_utils import save_model, load_model, rank_fusion, DynamicEval
from modeling import Pipeline, BertForRetriever, LxmertForRetriever


# In[2]:


logger = logging.getLogger(__name__)


# In[3]:


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# In[4]:


def train(args, train_dataset, model, query_tokenizer, passage_tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(
        train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=args.num_workers)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    args.warmup_steps = int(t_total * args.warmup_portion)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        # model.to(f'cuda:{model.device_ids[0]}')

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    tr_loss, logging_loss = 0.0, 0.0
    all_preds = []
    all_labels = []
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs),
                            desc="Epoch", disable=args.local_rank not in [-1, 0])
    # Added here for reproductibility (even between python 2 and 3)
    set_seed(args)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration",
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):           
            model.train()
            batch = {k: v.to(args.device) for k, v in batch.items() if k != 'question_id'}
            loss, preds, labels = model(**batch)[0:3]
            
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            all_preds.extend(to_list(preds))
            all_labels.extend(to_list(labels))
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    # Only evaluate when single GPU otherwise metrics may not average well
                    # if args.local_rank == -1 and args.evaluate_during_training:
                    #     results = evaluate(args, model, tokenizer)
                    #     for key, value in results.items():
                    #         tb_writer.add_scalar(
                    #             'eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar(
                        'Train/LR', scheduler.get_last_lr()[0], global_step)
                    tb_writer.add_scalar(
                        'Train/Loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    tb_writer.add_scalar(
                        'Train/Acc', np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_preds), global_step)
                    
                    all_preds = []
                    all_labels = []
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(
                        args.output_dir, 'checkpoint-{}'.format(global_step))
                    save_model(args, model, query_tokenizer, passage_tokenizer, output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


# In[5]:


def evaluate(args, model, query_tokenizer, passage_tokenizer, prefix=""):
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    predict_dir = os.path.join(args.output_dir, 'predictions')
    if not os.path.exists(predict_dir) and args.local_rank in [-1, 0]:
        os.makedirs(predict_dir)

    if args.collection_reps_path:
        logger.info(f'Reading collection reps from cache: {args.collection_reps_path}')
        
        passage_ids = []
        # passage_reps = []
        if os.path.isdir(args.collection_reps_path):
            logger.info('Reading a directory of cache files.')
            collection_reps_files = [os.path.join(
                args.collection_reps_path, f) for f in os.listdir(args.collection_reps_path)]
        else:
            logger.info('Reading a sigle cache file.')
            collection_reps_files = [args.collection_reps_path]
        
        index = faiss.IndexFlatIP(args.proj_size)
        for collection_reps_file in collection_reps_files:
            with open(collection_reps_file) as fin:
                for line in tqdm(fin):
                    dic = json.loads(line.strip())
                    passage_ids.append(dic['id'])
                    # passage_reps.append(dic['rep'])
                    index.add(np.expand_dims(np.asarray(dic['rep'], dtype='float32'), axis=0))
        # passage_reps = np.asarray(passage_reps, dtype='float32')
        
    else:
        logger.info('Generating val collection reps.')
        passage_ids, passage_reps = gen_passage_rep(
            args, model, passage_tokenizer, write_to_file=False)
        
        passage_reps = np.asarray(passage_reps, dtype='float32')
        index = faiss.IndexFlatIP(args.proj_size)
        index.add(passage_reps)

    logger.info('len index: {}'.format(index.ntotal))
    
    qids, query_reps = gen_query_rep(args, model, query_tokenizer)
    query_reps = np.asarray(query_reps, dtype='float32')
    logger.info('query_reps.shape: {}'.format(
        ' '.join([str(d) for d in query_reps.shape])))
    
    query_reps_orig_shape = query_reps.shape
    if len(query_reps_orig_shape) == 3:
        # query_reps shape w/ fusion: (batch_size, 36, proj_size).
        query_batch_size, query_obj_num, _ = query_reps_orig_shape
        query_reps = query_reps.reshape(-1, args.proj_size)
        logger.info('Reshaped query_reps.shape: {}'.format(
            ' '.join([str(d) for d in query_reps.shape])))
    
    # Shape: (num_queries, retrieve_top_k).
    D, I = index.search(query_reps, args.retrieve_top_k)
    logger.info('D.shape: {}'.format(
        ' '.join([str(d) for d in D.shape])))
    logger.info('I.shape: {}'.format(
        ' '.join([str(d) for d in I.shape])))
    index = None
    
    if len(query_reps_orig_shape) == 3:
        D, I = rank_fusion(
            D=D.reshape((query_batch_size, query_obj_num, args.retrieve_top_k)), 
            I=I.reshape((query_batch_size, query_obj_num, args.retrieve_top_k)), 
            fusion=args.lxmert_rep_type['fusion'])
        logger.info('fusion D.shape: {}'.format(
            ' '.join([str(d) for d in D.shape])))
        logger.info('fusion I.shape: {}'.format(
            ' '.join([str(d) for d in I.shape])))

    fout = open(os.path.join(predict_dir, f'results_{prefix}.txt'), 'w')
    run = {}
    for qid, retrieved_ids, scores in zip(qids, I, D):
        run[str(qid)] = {passage_ids[retrieved_id]: float(
            score) for retrieved_id, score in zip(retrieved_ids, scores)}
        for i, (retrieved_id, score) in enumerate(zip(retrieved_ids, scores)):
            fout.write(f'{qid} Q0 {passage_ids[retrieved_id]} {i + 1} {score} DENSE\n')
    fout.close()
    
    # qrels = defaultdict(dict)
    # with open(args.qrels) as fin:
    #     for line in fin:
    #         qid, _, pid, relevance = line.strip().split()
    #         qrels[qid][pid] = int(relevance)
    
    logger.info('Generating dynamic qrels.')
    qrels = dynamic_eval.gen_qrels(qids, I, passage_ids)
    
    with open(os.path.join(predict_dir, f'qrels_{prefix}.txt'), 'w') as qrels_fout:
        for qid, pids in qrels.items():
            for pid, relevance in pids.items():
                qrels_fout.write(f'{qid} 0 {pid} {relevance}\n')
    logger.info('Dynamic qrels generated.')
    
    assert len(
        qrels) == len(qids) == len(
        run), f'lengths of qrels, qids, and run do not match {len(qrels)}, {len(qids)}, {len(run)}'
    num_passages_in_qrels = len([pid for l in qrels.values() for pid in l])
    num_pos_passages = len([pid for l in qrels.values() for pid in l if pid != 'placeholder'])
    num_placeholder_passages = len([pid for l in qrels.values() for pid in l if pid == 'placeholder'])
    num_questions_with_pos_passages = len([ps for ps in qrels.values() if ps != {'placeholder': 0}])
    assert num_pos_passages + num_placeholder_passages == num_passages_in_qrels
    assert num_placeholder_passages == len(qrels)
    logger.info(f'len(qrels): {len(qrels)}')
    logger.info(f'num_passages_in_qrels: {num_passages_in_qrels}')
    logger.info(f'num_pos_passages: {num_pos_passages}')
    logger.info(f'num_placeholder_passages: {num_placeholder_passages}')
    logger.info(f'num_questions_with_pos_passages: {num_questions_with_pos_passages}')
    
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {'recip_rank', 'P_5'})
    metrics = evaluator.evaluate(run)

    mrr_list = [v['recip_rank'] for v in metrics.values()]
    p_list = [v['P_5'] for v in metrics.values()]
    eval_metrics = {'MRR': np.average(
        mrr_list), 'Precision': np.average(p_list)}
    
    for k, v in eval_metrics.items():
        logger.info(f'{k}: {v}')
        
    return eval_metrics


# In[6]:


def evaluate_pairs(args, model, query_tokenizer, passage_tokenizer, prefix=""):
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    predict_dir = os.path.join(args.output_dir, 'predictions')
    if not os.path.exists(predict_dir) and args.local_rank in [-1, 0]:
        os.makedirs(predict_dir)

    if prefix == 'test':
        eval_file = args.input_file.format(args.test_data_sub_type)
        eval_data_sub_type = args.test_data_sub_type
    else:
        eval_file = args.input_file.format(args.val_data_sub_type)
        eval_data_sub_type = args.val_data_sub_type

    dataset = RetrieverDataset(filename=eval_file, 
                               image_features_path=args.image_features_path,
                               data_sub_type=eval_data_sub_type,
                               query_tokenizer=query_tokenizer,
                               passage_tokenizer=passage_tokenizer,
                               load_small=args.load_small,
                               question_max_seq_length=args.question_max_seq_length,
                               passage_max_seq_length=args.passage_max_seq_length,
                               query_only=False)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(
        dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        # model.to(f'cuda:{model.device_ids[0]}')
        
    # Eval!
    logger.info("***** Eval pairs {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    all_preds = []
    all_labels = []
    eval_loss = 0.0
    eval_steps = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        eval_steps += 1
        model.eval()
        batch = {k: v.to(args.device)
                 for k, v in batch.items() if k != 'question_id'}
        with torch.no_grad():
            outputs = model(**batch)
            loss, preds, labels = outputs[0:3]
            
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
        
            eval_loss += loss.item()
            all_preds.extend(to_list(preds))
            all_labels.extend(to_list(labels))
    
    logger.info('len preds: {}'.format(len(all_preds)))
    logger.info('len labels: {}'.format(len(all_labels)))
    
    eval_metrics = {'Acc': np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_preds), 
                    'Loss': eval_loss / eval_steps}
    
    for k, v in eval_metrics.items():
        logger.info(f'{k}: {v}')
        
    return eval_metrics


# In[7]:


def gen_passage_rep(args, model, tokenizer, write_to_file=False):
    dataset = GenPassageRepDataset(args.gen_passage_rep_input, tokenizer, args.load_small, 
                                   passage_max_seq_length=args.passage_max_seq_length)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(
        dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        # model.to(f'cuda:{model.device_ids[0]}')

    # Eval!
    logger.info("***** Gen passage rep *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    run_dict = {}
    start_time = timeit.default_timer()
    
    if write_to_file:
        fout = open(args.gen_passage_rep_output, 'a')
        
    passage_ids = []
    passage_reps_list = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        example_ids = np.asarray(
            batch['example_id']).reshape(-1).tolist()
        passage_ids.extend(example_ids)
        batch = {k: v.to(args.device)
                 for k, v in batch.items() if k != 'example_id'}
        with torch.no_grad():
            passage_reps = model(**batch)[0]          
            passage_reps_list.extend(to_list(passage_reps))
        
        if write_to_file:
            for example_id, passage_rep in zip(example_ids, to_list(passage_reps)):
                fout.write(json.dumps({'id': example_id, 'rep': passage_rep}) + '\n')
    
    if write_to_file:
        fout.close()
    
    return passage_ids, passage_reps_list


# In[8]:


def gen_query_rep(args, model, tokenizer, prefix=''):
    if prefix == 'test':
        eval_file = args.input_file.format(args.test_data_sub_type)
        eval_data_sub_type = args.test_data_sub_type
    else:
        eval_file = args.input_file.format(args.val_data_sub_type)
        eval_data_sub_type = args.val_data_sub_type

    dataset = RetrieverDataset(filename=eval_file, 
                               image_features_path=args.image_features_path,
                               data_sub_type=eval_data_sub_type,
                               query_tokenizer=tokenizer,
                               load_small=args.load_small,
                               question_max_seq_length=args.question_max_seq_length, 
                               query_only=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(
        dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        # model.to(f'cuda:{model.device_ids[0]}')
        
    # Eval!
    logger.info("***** Gen query rep {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_qids = []
    all_query_reps = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        qids = np.asarray(
            batch['question_id']).reshape(-1).tolist()
        batch = {k: v.to(args.device)
                 for k, v in batch.items() if k != 'question_id'}
        with torch.no_grad():
            outputs = model(**batch)
            query_reps = outputs[0]

        all_qids.extend(qids)
        all_query_reps.extend(to_list(query_reps))

    return all_qids, all_query_reps


# In[9]:


parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument("--input_file", default='/mnt/scratch/chenqu/okvqa/data_v2/{}_pairs_cap_combine_sum.txt',
                    type=str, required=False,
                    help="okvqa files.")
parser.add_argument("--image_features_path", default='/mnt/scratch/chenqu/okvqa/okvqa_image_features/okvqa.datasets',
                    type=str, required=False,
                    help="Path to image features.")
parser.add_argument("--train_data_sub_type", default='train2014',
                   type=str, required=False,
                   help="Train data sub type.")
parser.add_argument("--val_data_sub_type", default='test2014',
                   type=str, required=False,
                   help="Val data sub type.")
parser.add_argument("--test_data_sub_type", default='',
                   type=str, required=False,
                   help="Test data sub type.")

parser.add_argument("--ann_file", default='/mnt/scratch/chenqu/okvqa/mscoco_val2014_annotations.json',
                    type=str, required=False,
                    help="Path to val okvqa annotations. For dynamic evaluation.")
parser.add_argument("--ques_file", default='/mnt/scratch/chenqu/okvqa/OpenEnded_mscoco_val2014_questions.json',
                    type=str, required=False,
                    help="Path to val okvqa questions. For dynamic evaluation.")
parser.add_argument("--passage_id_to_line_id_file", default='/mnt/scratch/chenqu/okvqa/passage_id_to_line_id.json',
                    type=str, required=False,
                    help="Path to passage_id_to_line_id_file. For dynamic evaluation.")
parser.add_argument("--all_blocks_file", default='/mnt/scratch/chenqu/okvqa/all_blocks.txt',
                    type=str, required=False,
                    help="Path to all blocks file. For dynamic evaluation.")


parser.add_argument("--query_model_name_or_path", default='unc-nlp/lxmert-base-uncased', type=str, required=False,
                    help="unc-nlp/lxmert-base-uncased or bert-base-uncased")
parser.add_argument("--passage_model_name_or_path", default='bert-base-uncased', type=str, required=False,
                    help="Path to pre-trained model or shortcut name")
parser.add_argument("--output_dir", default='/mnt/scratch/chenqu/okvqa_output/test_10', type=str, required=False,
                    help="The output directory where the model checkpoints and predictions will be written.")
# parser.add_argument("--qrels", default='/mnt/scratch/chenqu/okvqa/bm25/val2014_qrels.txt', type=str, required=False,
#                     help="qrels to evaluate open retrieval")    

parser.add_argument("--query_encoder_type", default='lxmert', type=str, required=False,
                    help="whether to use bert or lxmert as the query encoder") 
parser.add_argument("--lxmert_rep_type", 
                    # default='{"pooled_output": "none", "vision_output": "none", "fusion": "combine_sum"}', 
                    # default='{"pooled_output": "none", "language_output": "max", "vision_output": "mean"}', 
                    default='{"pooled_output": "none"}', 
                    type=str, required=False,
                    help="The type of hte outputs and their pooling method for the query rep") 
parser.add_argument("--neg_type", 
                    default='other_pos+all_neg', 
                    type=str, required=False,
                    help="What type of negatives we use to compute the loss. These options only "
                    "work with `lxmert_rep_type`=`'pooled_output': 'none'}` except for `neg`. "
                    "neg: use the retrieved negative passage. "
                    "all_neg: use all negative passages in the batch. "
                    "other_pos+neg: use all other positive passages in the batch and the retrieved negative passage. "
                    "other_pos+all_neg: use all other positive passages in the batch and all negative passages in the batch.") 
                    

parser.add_argument("--retrieve_top_k", default=5, type=int,
                    help="Retrieve top k passages.")
parser.add_argument("--gen_passage_rep", default=False, type=str2bool,
                    help="generate passage representations for all passages.")
parser.add_argument("--retrieve_checkpoint",
                    # default='/mnt/scratch/chenqu/okvqa_output/test_10/checkpoint-10', type=str,
                    default='', type=str,
                    help="generate query/passage representations with this checkpoint")
parser.add_argument("--gen_passage_rep_input",
                    default='/mnt/scratch/chenqu/okvqa/data_v2/val2014_blocks_cap_combine_sum.txt', type=str,
                    help="generate passage representations for this file that contains passages")
parser.add_argument("--gen_passage_rep_output",
                    # default='/mnt/scratch/chenqu/okvqa_output/test_10/checkpoint-10/reps.json', type=str,
                    default='', type=str,
                    help="Val passage representations")
parser.add_argument("--collection_reps_path",
                    default='', type=str,
                    # default='/mnt/scratch/chenqu/okvqa_output/test_10/checkpoint-10/reps.json', type=str,
                    help="All passage representations")

# Other parameters
# parser.add_argument("--config_name", default="", type=str,
#                     help="Pretrained config name or path if not the same as model_name")
# parser.add_argument("--tokenizer_name", default="", type=str,
#                     help="Pretrained tokenizer name or path if not the same as model_name")
parser.add_argument("--cache_dir", default="/mnt/scratch/chenqu/okvqa_huggingface_cache", type=str,
                    help="Where do you want to store the pre-trained models downloaded from s3")

parser.add_argument("--question_max_seq_length", default=20, type=int,
                    help="The maximum input sequence length of query."
                         "125 is the max question length in the reader.")
parser.add_argument("--passage_max_seq_length", default=384, type=int,
                    help="The maximum input sequence length of passage.")
parser.add_argument("--proj_size", default=128, type=int,
                    help="The size of the query/passage rep after projection of [CLS] rep.")
parser.add_argument("--do_train", default=False, type=str2bool,
                    help="Whether to run training.")
parser.add_argument("--do_eval", default=True, type=str2bool,
                    help="Whether to run retrieval on the dev set.")
parser.add_argument("--do_test", default=False, type=str2bool,
                    help="Whether to run eval on the test set.")
parser.add_argument("--do_eval_pairs", default=False, type=str2bool,
                    help="Whether to do classification in pairs on the dev set.")
# parser.add_argument("--evaluate_during_training", default=False, type=str2bool,
#                     help="Rul evaluation during training at each logging step.")
parser.add_argument("--do_lower_case", default=True, type=str2bool,
                    help="Set this flag if you are using an uncased model.")

parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--per_gpu_eval_batch_size", default=50, type=int,
                    help="Batch size per GPU/CPU for evaluation.")
parser.add_argument("--learning_rate", default=5e-5, type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--weight_decay", default=0.0, type=float,
                    help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")
parser.add_argument("--num_train_epochs", default=1.0, type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--max_steps", default=0, type=int,
                    help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
parser.add_argument("--warmup_steps", default=0, type=int,
                    help="Linear warmup over warmup_steps.")
parser.add_argument("--warmup_portion", default=0.1, type=float,
                    help="Linear warmup over warmup_steps (=t_total * warmup_portion). override warmup_steps ")
parser.add_argument("--verbose_logging", action='store_true',
                    help="If true, all of the warnings related to data processing will be printed. "
                         "A number of warnings are expected for a normal SQuAD evaluation.")

parser.add_argument('--logging_steps', type=int, default=2,
                    help="Log every X updates steps.")
parser.add_argument('--save_steps', type=int, default=5,
                    help="Save checkpoint every X updates steps.")
parser.add_argument("--eval_all_checkpoints", default=True, type=str2bool,
                    help="Evaluate all checkpoints starting with the same prefix "
                    "as model_name ending and ending with step number")
parser.add_argument("--no_cuda", default=False, type=str2bool,
                    help="Whether not to use CUDA when available")
parser.add_argument('--overwrite_output_dir', default=True, type=str2bool,
                    help="Overwrite the content of the output directory")
parser.add_argument('--overwrite_cache', action='store_true',
                    help="Overwrite the cached training and evaluation sets")
parser.add_argument('--seed', type=int, default=42,
                    help="random seed for initialization")

parser.add_argument("--local_rank", type=int, default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument('--fp16', default=False, type=str2bool,
                    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
parser.add_argument('--fp16_opt_level', type=str, default='O1',
                    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                         "See details at https://nvidia.github.io/apex/amp.html")
parser.add_argument('--server_ip', type=str, default='',
                    help="Can be used for distant debugging.")
parser.add_argument('--server_port', type=str, default='',
                    help="Can be used for distant debugging.")

parser.add_argument("--load_small", default=True, type=str2bool, required=False,
                    help="whether to load just a small portion of data during development")
parser.add_argument("--num_workers", default=1, type=int, required=False,
                    help="number of workers for dataloader")

args, unknown = parser.parse_known_args()

if (os.path.exists(args.output_dir) and 
    os.listdir(args.output_dir) and 
    args.do_train and 
    not args.overwrite_output_dir):
    raise ValueError(
        f'Output directory ({args.output_dir}) already exists and is not empty. '
        'Use --overwrite_output_dir to overcome.')

# Setup distant debugging if needed
if args.server_ip and args.server_port:
    # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
    import ptvsd
    print("Waiting for debugger attach")
    ptvsd.enable_attach(
        address=(args.server_ip, args.server_port), redirect_output=True)
    ptvsd.wait_for_attach()

# Setup CUDA, GPU & distributed training
if args.local_rank == -1 or args.no_cuda:
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    torch.cuda.set_device(0)
else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend='nccl')
    args.n_gpu = 1
args.device = device

# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
               args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

# Set seed
set_seed(args)

# Load pretrained model and tokenizer
if args.local_rank not in [-1, 0]:
    # Make sure only the first process in distributed training will download model & vocab
    torch.distributed.barrier()
    
args.lxmert_rep_type = json.loads(args.lxmert_rep_type)
    
if args.query_encoder_type == 'lxmert':
    query_config_class = LxmertConfig
    query_tokenizer_class = LxmertTokenizer
    query_encoder_class = LxmertForRetriever
elif args.query_encoder_type == 'bert':
    query_config_class = BertConfig
    query_tokenizer_class = BertTokenizer
    query_encoder_class = BertForRetriever
else:
    raise ValueError('`query_encoder_type` must be one of bert and lxmert.')
    
passage_config_class = BertConfig 
passage_tokenizer_class = BertTokenizer 
passage_encoder_class = BertForRetriever
          
query_config = query_config_class.from_pretrained(
    args.query_model_name_or_path,
    cache_dir=args.cache_dir if args.cache_dir else None)
query_config.proj_size = args.proj_size
query_config.lxmert_rep_type = args.lxmert_rep_type
query_tokenizer = query_tokenizer_class.from_pretrained(
    args.query_model_name_or_path,
    do_lower_case=args.do_lower_case,
    cache_dir=args.cache_dir if args.cache_dir else None)
query_encoder = query_encoder_class.from_pretrained(
    args.query_model_name_or_path, from_tf=False, config=query_config, 
    cache_dir=args.cache_dir if args.cache_dir else None)

passage_config = passage_config_class.from_pretrained(
    args.passage_model_name_or_path,
    cache_dir=args.cache_dir if args.cache_dir else None)
passage_config.proj_size = args.proj_size
passage_tokenizer = passage_tokenizer_class.from_pretrained(
    args.passage_model_name_or_path,
    do_lower_case=args.do_lower_case,
    cache_dir=args.cache_dir if args.cache_dir else None)
passage_encoder = passage_encoder_class.from_pretrained(
    args.passage_model_name_or_path, from_tf=False, config=passage_config, 
    cache_dir=args.cache_dir if args.cache_dir else None)

model = Pipeline(query_config, passage_config, 
                 query_encoder_type=args.query_encoder_type, neg_type=args.neg_type)
model.query_encoder = query_encoder
model.passage_encoder = passage_encoder

if args.local_rank == 0:
    # Make sure only the first process in distributed training will download model & vocab
    torch.distributed.barrier()

model.to(args.device)

logger.info("Training/evaluation parameters %s", args)

# Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
# Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
# remove the need for this code, but it is still valid.
if args.fp16:
    try:
        import apex
        apex.amp.register_half_function(torch, 'einsum')
    except ImportError:
        raise ImportError(
            "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

if args.gen_passage_rep and (args.do_train or args.do_eval or args.do_test):
    raise ValueError('do_train, do_eval, do_test must be set to False '
                     'when gen_passage_reps is enabled.')
    
# Training
if args.do_train:
    train_dataset = RetrieverDataset(filename=args.input_file.format(args.train_data_sub_type), 
                                     image_features_path=args.image_features_path,
                                     data_sub_type=args.train_data_sub_type,
                                     query_tokenizer=query_tokenizer,
                                     passage_tokenizer=passage_tokenizer,
                                     load_small=args.load_small,
                                     question_max_seq_length=args.question_max_seq_length,
                                     passage_max_seq_length=args.passage_max_seq_length)
    global_step, tr_loss = train(args, train_dataset, model, query_tokenizer, passage_tokenizer)
    logger.info(" global_step = %s, average loss = %s",
                global_step, tr_loss)

# Save the trained model and the tokenizer
if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    # Create output directory if needed
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    
    final_checkpoint_output_dir = os.path.join(
        args.output_dir, 'checkpoint-{}'.format(global_step))    
    save_model(args, model, query_tokenizer, passage_tokenizer, final_checkpoint_output_dir)

    model, query_tokenizer, passage_tokenizer = load_model(
        args, query_config, query_tokenizer_class, query_encoder_class, 
        passage_config, passage_tokenizer_class, passage_encoder_class, 
        Pipeline, final_checkpoint_output_dir)
    
    model.to(args.device)


# In[10]:


# Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory

results = {}
max_mrr = 0.0
best_metrics = {}
if (args.do_eval or args.do_eval_pairs) and args.local_rank in [-1, 0]:
    tb_writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))
    
    dynamic_eval = DynamicEval(args.ann_file, args.ques_file, 
                               args.passage_id_to_line_id_file, args.all_blocks_file)
    
    if args.retrieve_checkpoint:
        checkpoints = [args.retrieve_checkpoint]
    else:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(
                glob.glob(args.output_dir + '/**/' + 'query', recursive=True)))
        checkpoints.sort(key=lambda x: int(x.split('-')[-1]) if len(x) > 1 else 0)
        
    logger.info("Evaluate the following checkpoints: %s", checkpoints)

    for checkpoint in checkpoints:
        # Reload the model
        global_step = checkpoint.split(
            '-')[-1] if len(checkpoint) > 1 else ""
        if args.retrieve_checkpoint:
            global_step = 'retrieve'
            
        logger.info(f'global_step: {global_step}')
        logger.info(f'evaluating checkpoint: {checkpoint}')
        
        model, query_tokenizer, passage_tokenizer = load_model(
            args, query_config, query_tokenizer_class, query_encoder_class, 
            passage_config, passage_tokenizer_class, passage_encoder_class, 
            Pipeline, checkpoint)

        model.to(args.device)

        # Evaluate
        if args.do_eval:
            result = evaluate(args, model, query_tokenizer, passage_tokenizer, prefix=global_step)
            if result['MRR'] > max_mrr:
                max_mrr = result['MRR']
                best_metrics['MRR'] = result['MRR']
                best_metrics['Precision'] = result['Precision']
                best_metrics['global_step'] = global_step
            
            if global_step.isnumeric():
                for key, value in result.items():
                    tb_writer.add_scalar(
                        'Eval/{}'.format(key), value, global_step)

            result = dict((k + ('_{}'.format(global_step) if global_step else ''), v)
                          for k, v in result.items())
            results.update(result)
            
        if args.do_eval_pairs:
            result = evaluate_pairs(args, model, query_tokenizer, passage_tokenizer, prefix=global_step)
            
            if global_step.isnumeric():
                for key, value in result.items():
                    tb_writer.add_scalar(
                        'Eval/{}'.format(key), value, global_step)

            result = dict((k + ('_{}'.format(global_step) if global_step else ''), v)
                          for k, v in result.items())
            results.update(result)

    best_metrics_file = os.path.join(
        args.output_dir, 'predictions', 'best_metrics.json')
    with open(best_metrics_file, 'a') as fout:
        json.dump(best_metrics, fout)
        
    all_results_file = os.path.join(
        args.output_dir, 'predictions', 'all_results.json')
    with open(all_results_file, 'a') as fout:
        json.dump(results, fout)

    logger.info("Results: {}".format(results))
    logger.info("best metrics: {}".format(best_metrics))


# In[11]:


# if args.do_test and args.local_rank in [-1, 0]:
#     best_global_step = best_metrics['global_step']
#     best_checkpoint = os.path.join(
#         args.output_dir, 'checkpoint-{}'.format(best_global_step))
#     logger.info("Test the best checkpoint: %s", best_checkpoint)

#     model = model_class.from_pretrained(
#         best_checkpoint, force_download=True)
#     model.to(args.device)

#     # Evaluate
#     result = evaluate(args, model, tokenizer, prefix='test')

#     test_metrics_file=os.path.join(
#         args.output_dir, 'predictions', 'test_metrics.json')
#     with open(test_metrics_file, 'w') as fout:
#         json.dump(result, fout)

#     logger.info("Test Result: {}".format(result))


# In[12]:


if args.gen_passage_rep and args.local_rank in [-1, 0]:
    logger.info("Gen passage rep with: %s", args.retrieve_checkpoint)
    
    model, _, passage_tokenizer = load_model(
        args, query_config, query_tokenizer_class, query_encoder_class, 
        passage_config, passage_tokenizer_class, passage_encoder_class, 
        Pipeline, args.retrieve_checkpoint)
    model.query_encoder = None
    
    model.to(args.device)

    # Evaluate
    gen_passage_rep(args, model, passage_tokenizer, write_to_file=True)
    
    logger.info("Gen passage rep complete")


# In[ ]:




