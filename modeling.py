import os
import logging
import collections
import torch

from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from copy import deepcopy

from transformers import BertModel, BertPreTrainedModel
from transformers import LxmertModel, LxmertPreTrainedModel


logger = logging.getLogger(__name__)

class BertForRetriever(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.proj = nn.Linear(config.hidden_size, config.proj_size)

        self.init_weights()
        
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        reps = self.proj(pooled_output)
        
        return reps
    
    
class LxmertForRetriever(LxmertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.lxmert = LxmertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.lxmert_rep_type = config.lxmert_rep_type
        
        if 'fusion' not in config.lxmert_rep_type:
            proj_times = len(config.lxmert_rep_type) 
        else:
            proj_times = len(config.lxmert_rep_type) - 1
        self.proj = nn.Linear(config.hidden_size * proj_times, config.proj_size)
        

        self.init_weights()
        
        
    def forward(self, input_ids, attention_mask, token_type_ids, roi_features, boxes):
        outputs = self.lxmert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            visual_feats=roi_features,
            visual_pos=boxes,
            output_attentions=False)
        
        if 'fusion' in self.lxmert_rep_type:
            # Handles args.lxmert_rep_type with rank fusion.
            # i.e., '{"pooled_output": "none", "vision_output": "none", "fusion": "combine_max"}'.
            # or '{"pooled_output": "none", "vision_output": "none", "fusion": "combine_sum"}'.
            # These are the only supported lxmert_rep_type with rank fusion.
            if self.lxmert_rep_type not in [
                {'pooled_output': 'none', 'vision_output': 'none', 'fusion': 'combine_max'}, 
                {'pooled_output': 'none', 'vision_output': 'none', 'fusion': 'combine_sum'}]:
                raise ValueError('`lxmert_rep_type` must be '
                                 '`{"pooled_output": "none", "vision_output": "none", "fusion": "combine_max"}` '
                                 'or `{"pooled_output": "none", "vision_output": "none", "fusion": "combine_sum"}` '
                                 f'for rank fusion, not {self.lxmert_rep_type}')
                
            vision_output = outputs.vision_output  # Shape: (batch_size, 36, 768).
            pooled_output = outputs.pooled_output  # Shape: (batch_size, 768).
            pooled_output_expanded = pooled_output.unsqueeze(dim=1).expand_as(vision_output)  # Shape: (batch_size, 36, 768).
            lxmert_output = torch.cat([pooled_output_expanded, vision_output], dim=2)  # Shape: (batch_size, 36, 768 * 2).
                
        else:
            # Handles other args.lxmert_rep_type.
            # E.g., '{"pooled_output": "none", "language_output": "max", "vision_output": "mean"}'.
            lxmert_output_list = []
            for rep_type, reduce_type in self.lxmert_rep_type.items():
                if rep_type == 'pooled_output':
                    if reduce_type != 'none':
                        raise ValueError(f'`reduce_type` must be `none`, not `{reduce_type}`.')
                    lxmert_output_list.append(outputs.pooled_output)

                elif rep_type == 'vision_output':
                    vision_output = outputs.vision_output
                    if reduce_type == 'mean':
                        lxmert_output_list.append(torch.mean(vision_output, dim=1))
                    elif reduce_type == 'max':
                        lxmert_output_list.append(torch.max(vision_output, dim=1).values)
                    else:
                        raise ValueError(f'`reduce_type` must be `max` or `mean`, not `{reduce_type}`.')

                elif rep_type == 'language_output':
                    language_output = outputs.language_output
                    if reduce_type == 'mean':
                        language_output_masked = language_output * torch.unsqueeze(attention_mask, -1)
                        lxmert_output_list.append(
                            torch.sum(language_output_masked, dim=1) / (torch.sum(attention_mask, dim=1, keepdims=True) + 1e-10))
                    elif reduce_type == 'max':
                        language_output_masked = language_output - torch.unsqueeze(1. - attention_mask, dim=-1) * 1e30
                        lxmert_output_list.append(torch.max(language_output_masked, dim=1).values)
                    else:
                        raise ValueError(f'`reduce_type` must be `max` or `mean`, not `{reduce_type}`.')

                else:
                    raise ValueError(f'`rep_type` must be `pooled_output`, `vision_output`, or `language_output`, not `{rep_type}`.')

            lxmert_output = torch.cat(lxmert_output_list, dim=1)
        
        lxmert_output = self.dropout(lxmert_output)
        # Shape w/o fusion: (batch_size, proj_size),
        # shape w/ fusion: (batch_size, 36, proj_size).
        reps = self.proj(lxmert_output)  
        
        return reps
    

class Pipeline(nn.Module):
    def __init__(self, query_encoder_config, passage_encoder_config, query_encoder_type, neg_type):
        super(Pipeline, self).__init__()
        
        self.query_encoder_type = query_encoder_type
        
        if query_encoder_type == 'lxmert':
            self.query_encoder = LxmertModel(query_encoder_config)
            self.lxmert_rep_type = query_encoder_config.lxmert_rep_type
        elif query_encoder_type == 'bert':
            self.query_encoder = BertModel(query_encoder_config)
        else:
            raise ValueError('`query_encoder_type` must be one of bert and lxmert.')
        
        self.passage_encoder = BertModel(passage_encoder_config)
        self.neg_type = neg_type
        
        
    def forward(self, 
                question_input_ids=None, question_attention_mask=None, question_token_type_ids=None, 
                roi_features=None, boxes=None, 
                pos_passage_input_ids=None, pos_passage_attention_mask=None, pos_passage_token_type_ids=None, 
                neg_passage_input_ids=None, neg_passage_attention_mask=None, neg_passage_token_type_ids=None, 
                passage_input_ids=None, passage_attention_mask=None, passage_token_type_ids=None):
        
        # Generate passage reps.
        if passage_input_ids is not None:
            passage_reps = self.passage_encoder(
                input_ids=passage_input_ids,
                attention_mask=passage_attention_mask,
                token_type_ids=passage_token_type_ids)
            
            return (passage_reps, )
        
        outputs = ()
        
        # Generate query reps.
        if question_input_ids is not None:
            if self.query_encoder_type == 'lxmert':
                query_reps = self.query_encoder(
                    input_ids=question_input_ids,
                    attention_mask=question_attention_mask,
                    token_type_ids=question_token_type_ids,
                    roi_features=roi_features.float(),
                    boxes=boxes.float())                               
            elif self.query_encoder_type == 'bert':
                query_reps = self.query_encoder(
                    input_ids=question_input_ids,
                    attention_mask=question_attention_mask,
                    token_type_ids=question_token_type_ids)              
            else:
                raise ValueError('`query_encoder_type` must be one of bert and lxmert.')
            
            outputs = (query_reps, ) + outputs
        
        # Compute loss.
        if (question_input_ids is not None and 
            pos_passage_input_ids is not None and 
            neg_passage_input_ids is not None):
            
            # Shape: (batch_size, proj_size).
            pos_passage_reps = self.passage_encoder(
                input_ids=pos_passage_input_ids,
                attention_mask=pos_passage_attention_mask,
                token_type_ids=pos_passage_token_type_ids)
            
            # Shape: (batch_size, proj_size).
            neg_passage_reps = self.passage_encoder(
                input_ids=neg_passage_input_ids,
                attention_mask=neg_passage_attention_mask,
                token_type_ids=neg_passage_token_type_ids)
            
            if len(query_reps.size()) == 2:
                # query_reps shape w/o fusion: (batch_size, proj_size).
                
                if self.neg_type == 'neg':
                    # Shape: (batch_size, ).
                    pos_logits = torch.sum(query_reps * pos_passage_reps, dim=-1)
                    neg_logits = torch.sum(query_reps * neg_passage_reps, dim=-1)
                    logits = torch.stack([pos_logits, neg_logits], dim=1)                    
                    labels = torch.zeros(logits.size(0), device=logits.device, dtype=torch.long)
                
                elif self.neg_type == 'all_neg':
                    pos_logits = torch.sum(query_reps * pos_passage_reps, dim=-1)
                    neg_logits = torch.matmul(query_reps, neg_passage_reps.transpose(0, 1))  # Shape: (batch_size, batch_size).
                    logits = torch.cat([pos_logits.unsqueeze(dim=-1), neg_logits], dim=1)  # Shape: (batch_size, batch_size + 1).                    
                    labels = torch.zeros(logits.size(0), device=logits.device, dtype=torch.long)
                    
                elif self.neg_type == 'other_pos+neg':
                    pos_logits = torch.matmul(query_reps, pos_passage_reps.transpose(0, 1))  # Shape: (batch_size, batch_size).
                    neg_logits = torch.sum(query_reps * neg_passage_reps, dim=-1)
                    logits = torch.cat([pos_logits, neg_logits.unsqueeze(dim=-1)], dim=1)  # Shape: (batch_size, batch_size + 1).                  
                    labels = torch.arange(logits.size(0), device=logits.device, dtype=torch.long)
                    
                elif self.neg_type == 'other_pos+all_neg':
                    pos_logits = torch.matmul(query_reps, pos_passage_reps.transpose(0, 1))  # Shape: (batch_size, batch_size).
                    neg_logits = torch.matmul(query_reps, neg_passage_reps.transpose(0, 1))  # Shape: (batch_size, batch_size).
                    logits = torch.cat([pos_logits, neg_logits], dim=1)  # Shape: (batch_size, 2 * batch_size).
                    labels = torch.arange(logits.size(0), device=logits.device, dtype=torch.long)
                
                else:
                    raise ValueError(f'`neg_type` should be one of `neg`, `all_neg`, `other_pos+neg`, or `other_pos+all_neg`, not {self.neg_type}.')
                
            else:
                # query_reps shape w/ fusion: (batch_size, 36, proj_size).
                
                # Shape: (batch_size, 36, proj_size).
                pos_passage_reps_expanded = pos_passage_reps.unsqueeze(dim=1).expand_as(query_reps)
                neg_passage_reps_expanded = neg_passage_reps.unsqueeze(dim=1).expand_as(query_reps)
                
                if self.lxmert_rep_type['fusion'] == 'combine_max':
                    # Shape: (batch_size, ).
                    pos_logits = torch.sum(query_reps * pos_passage_reps_expanded, dim=-1).max(dim=-1).values
                    neg_logits = torch.sum(query_reps * neg_passage_reps_expanded, dim=-1).max(dim=-1).values
                elif self.lxmert_rep_type['fusion'] == 'combine_sum':
                    pos_logits = torch.sum(query_reps * pos_passage_reps_expanded, dim=-1).sum(dim=-1)
                    neg_logits = torch.sum(query_reps * neg_passage_reps_expanded, dim=-1).sum(dim=-1)
                else:
                    raise ValueError(f'`lxmert_rep_type["fusion"]` must be one of `combine_max` or `combine_sum`, '
                                     f'not {self.lxmert_rep_type["fusion"]}.')
                
                logits = torch.stack([pos_logits, neg_logits], dim=1)
                labels = torch.zeros(logits.size(0), device=logits.device, dtype=torch.long)
            
            loss_fn = CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            preds = torch.argmax(logits, dim=1)
            
            outputs = (loss, preds, labels, ) + outputs
            
        return outputs