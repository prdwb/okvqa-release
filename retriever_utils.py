from __future__ import absolute_import, division, print_function

import os
import json
import logging
import math
import collections
from collections import defaultdict
import linecache
import re
import numpy as np
from io import open
from torch.utils.data import Dataset
import torch
import datasets
from vqa_tools import VQA


logger = logging.getLogger(__name__)

class RetrieverInputExample(object):
    """
    A single training/test example for simple sequence classification.
    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class RetrieverInputFeatures(object):
    """
    A single set of features of data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
    
    
class RetrieverDataset(Dataset):
    def __init__(self, 
                 filename,
                 image_features_path,
                 data_sub_type,
                 query_tokenizer=None,
                 passage_tokenizer=None,
                 load_small=False, 
                 question_max_seq_length=20, 
                 passage_max_seq_length=384,
                 query_only=False):
        
        self._filename = filename
        self._data_sub_type = data_sub_type
        self._query_tokenizer = query_tokenizer
        self._passage_tokenizer = passage_tokenizer
        self._load_small = load_small
        self._question_max_seq_length = question_max_seq_length
        self._passage_max_seq_length = passage_max_seq_length
        self._query_only = query_only
        
        self._total_data = 0      
        if self._load_small:
            self._total_data = 50
        else:
            with open(filename, "r") as f:
                self._total_data = len(f.readlines())
                
        self._image_features = datasets.Dataset.from_file(image_features_path)
                
    def __len__(self):
        return self._total_data
                
    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)    
        entry = json.loads(line.strip())
        question_id = int(entry["question_id"])
        image_id = entry['image_id']
        question = entry['question']
        answers = entry['answers']  # Not used for retrieval.
        pos_passage = entry['pos_passage']['passage']
        neg_passage = entry['neg_passage']['passage']
        
        return_feature_dict = {'question_id': question_id}
                  
        question_features = self._query_tokenizer(
            question,
            padding="max_length",
            max_length=self._question_max_seq_length,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        query_feature_dict = {
            'question_input_ids': torch.squeeze(
                question_features.input_ids, dim=0),
            'question_token_type_ids': torch.squeeze(
                question_features.token_type_ids, dim=0),
            'question_attention_mask': torch.squeeze(
                question_features.attention_mask, dim=0),
            'roi_features': np.array(
                self._image_features[self._image_features['img_id']==image_id]['roi_features']),
            'boxes': np.array(
                self._image_features[self._image_features['img_id']==image_id]['boxes']),
        }

        return_feature_dict.update(query_feature_dict)
        
        if not self._query_only:
            for i, (passage, passage_type) in enumerate(
                zip([pos_passage, neg_passage], ['pos', 'neg'])):
                
                passage_example = RetrieverInputExample(
                    guid=int(f'{question_id}{i}'),
                    text_a=passage)
                passage_feature = retriever_convert_example_to_feature(
                    passage_example, self._passage_tokenizer, max_length=self._passage_max_seq_length)
                passage_feature_dict = {
                    f'{passage_type}_passage_input_ids': np.asarray(passage_feature.input_ids), 
                    f'{passage_type}_passage_token_type_ids': np.asarray(passage_feature.token_type_ids), 
                    f'{passage_type}_passage_attention_mask': np.asarray(passage_feature.attention_mask)} 
                return_feature_dict.update(passage_feature_dict)
        
#         for k, v in return_feature_dict.items():
#             if isinstance(v, np.ndarray):
#                 print(k, v.shape)
#             if isinstance(v, torch.Tensor):
#                 print(k, v.size())
        return return_feature_dict

    
class GenPassageRepDataset(Dataset):
    def __init__(self, filename, tokenizer, 
                 load_small, passage_max_seq_length=384):
        
        self._filename = filename
        self._tokenizer = tokenizer
        self._load_small = load_small  
        self._passage_max_seq_length = passage_max_seq_length
                
        self._total_data = 0      
        if self._load_small:
            self._total_data = 100
        else:
            with open(filename, "r") as f:
                self._total_data = len(f.readlines())
                
    def __len__(self):
        return self._total_data
                
    def __getitem__(self, idx):
        """read a line of preprocessed open-retrieval quac file into a quac example"""
        line = linecache.getline(self._filename, idx + 1)    
        entry = json.loads(line.strip())
        passage_id = entry["id"]
        passage = entry['text']
        
        passage_example = RetrieverInputExample(guid=passage_id, text_a=passage)
        passage_feature = retriever_convert_example_to_feature(passage_example, self._tokenizer, 
                                                               max_length=self._passage_max_seq_length)
        batch_feature = {'passage_input_ids': np.asarray(passage_feature.input_ids), 
                         'passage_token_type_ids': np.asarray(passage_feature.token_type_ids), 
                         'passage_attention_mask': np.asarray(passage_feature.attention_mask),
                         'example_id': passage_id}

        return batch_feature

def retriever_convert_example_to_feature(example, tokenizer,
                                      max_length=512,
                                      pad_on_left=False,
                                      pad_token=0,
                                      pad_token_segment_id=0,
                                      mask_padding_with_zero=True):
    """
    Loads a data file into a list of ``InputFeatures``
    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)
    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.
    """


    inputs = tokenizer.encode_plus(
        example.text_a,
        example.text_b,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
        token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
    else:
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

    assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
    assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
    assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

    if False:
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
        logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
        logger.info("label: %s" % (example.label))

    feature = RetrieverInputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label=example.label)

    return feature


def save_model(args, model, query_tokenizer, passage_tokenizer, output_path):
    logger.info("Saving model checkpoint to %s", output_path)
    
    query_encoder_output_path = os.path.join(output_path, 'query')
    passage_encoder_output_path = os.path.join(output_path, 'passage')
    
    if not os.path.exists(query_encoder_output_path):
        os.makedirs(query_encoder_output_path)
    if not os.path.exists(passage_encoder_output_path):
        os.makedirs(passage_encoder_output_path)
    
    model_to_save = model.module if hasattr(model, 'module') else model
                    
    query_encoder_to_save = model_to_save.query_encoder
    query_encoder_to_save.save_pretrained(query_encoder_output_path)
    query_tokenizer.save_pretrained(query_encoder_output_path)
                    
    passage_encoder_to_save = model_to_save.passage_encoder
    passage_encoder_to_save.save_pretrained(passage_encoder_output_path)
    passage_tokenizer.save_pretrained(passage_encoder_output_path)

    torch.save(args, os.path.join(
        output_path, 'training_args.bin'))
    
    logger.info("Saved model checkpoint to %s", output_path)
    
    
def load_model(args, query_config, query_tokenizer_class, query_encoder_class, 
               passage_config, passage_tokenizer_class, passage_encoder_class, 
               pipeline_class, checkpoint_path):
    logger.info("Loading model from %s", checkpoint_path)
    
    query_encoder = query_encoder_class.from_pretrained(
        os.path.join(checkpoint_path, 'query'))
    query_tokenizer = query_tokenizer_class.from_pretrained(
        os.path.join(checkpoint_path, 'query'), 
        do_lower_case=args.do_lower_case)
                    
    passage_encoder = passage_encoder_class.from_pretrained(
        os.path.join(checkpoint_path, 'passage'))
    passage_tokenizer = passage_tokenizer_class.from_pretrained(
        os.path.join(checkpoint_path, 'passage'), 
        do_lower_case=args.do_lower_case)
    
    model = pipeline_class(query_config, passage_config, 
                           query_encoder_type=args.query_encoder_type, neg_type=args.neg_type)
    model.query_encoder = query_encoder
    model.passage_encoder = passage_encoder
    
    return model, query_tokenizer, passage_tokenizer


def rank_fusion(D, I, fusion):
    # D, I shape: (num_questions, num_objs, retrieve_top_k).
    logger.info('Reshaped D.shape: {}'.format(
        ' '.join([str(d) for d in D.shape])))
    logger.info('Reshaped I.shape: {}'.format(
        ' '.join([str(d) for d in I.shape])))
    
    num_questions, num_objs, k = D.shape
    
    fusion_scores = {}
    for qid in range(num_questions):
        for oid in range(num_objs):
            for pid in range(k):
                if qid not in fusion_scores:
                    fusion_scores[qid] = {}
                score = D[qid][oid][pid]
                retrieved_id = I[qid][oid][pid]
                
                if fusion == 'combine_max':
                    fusion_scores[qid][retrieved_id] = max(
                        fusion_scores[qid].get(retrieved_id, float('-inf')), score)
                elif fusion == 'combine_sum':
                    fusion_scores[qid][retrieved_id]= fusion_scores[qid].get(retrieved_id, 0.) + score
                else:
                    raise ValueError(
                        f'`lxmert_rep_type["fusion"]` must be one of `combine_max` or `combine_sum`, '
                        f'not {self.lxmert_rep_type["fusion"]}.')
                                        
    fusion_D = []
    fusion_I = []
    for qid, scores in fusion_scores.items():
        ranked_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        fusion_D.append([x[1] for x in ranked_scores[:k]])
        fusion_I.append([x[0] for x in ranked_scores][:k])

    return np.asarray(fusion_D), np.asarray(fusion_I)
    
    
class DynamicEval():
    def __init__(self, ann_file, ques_file, passage_id_to_line_id_file, all_blocks_file):
        
        with open(passage_id_to_line_id_file) as fin:
            self.passage_id_to_line_id = json.load(fin)
            
        self.vqa = VQA(ann_file, ques_file)
        self.all_blocks_file = all_blocks_file
            
    
    def get_answers(self, question_id):
        ann = self.vqa.loadQA(question_id)
        qa = self.vqa.returnQA(ann)[0]
        answers = set(answer.lower() for answer in qa['answers'].values() if answer)
        return answers
    
    
    def get_passage(self, passage_id):
        passage_line = linecache.getline(
            self.all_blocks_file, self.passage_id_to_line_id[passage_id])
        passage_dict = json.loads(passage_line)
        passage = passage_dict['text']
        assert passage_id == passage_dict['id']

        return passage
    
    
    def has_answers(self, answers, passage):
        passage = passage.lower()
        for answer in answers:
            # "\b" matches word boundaries.
            # answer_starts = [match.start() for match in re.finditer(
            #     r'\b{}\b'.format(answer.lower()), passage)]
            if re.search(r'\b{}\b'.format(answer), passage):
                return True
        return False
    
    
    def gen_qrels(self, question_ids, I, retrieved_id_to_passage_id):
       
        qrels = defaultdict(dict)
        for question_id, retrieved_ids in zip(question_ids, I):
            if question_id not in qrels:
                qrels[str(question_id)] = {'placeholder': 0}

            for retrieved_id in retrieved_ids:
                passage_id = retrieved_id_to_passage_id[retrieved_id]
                answers = self.get_answers(int(question_id))
                passage = self.get_passage(passage_id)

                if self.has_answers(answers, passage):
                    qrels[str(question_id)][passage_id] = 1

        return qrels