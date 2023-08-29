import jieba
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

def wwm_mlm_zh_encode(text, tokenizer, wwm_probability=0.15, max_length=None):
    assert max_length==None or type(max_length)==int
    special_tokens = {tokenizer.mask_token, tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token}
    mask_token_id = tokenizer.mask_token_id
    words = jieba.lcut(text)
    rands = np.random.rand(len(words))
    source, target = [], []
    for r,w in zip(rands, words):
        ids = tokenizer.encode(w, add_special_tokens=False)
        if w in special_tokens: 
            source.extend(ids)  
            target.extend([-100])
        elif r < wwm_probability*0.8:
            source.extend([mask_token_id]*len(ids))
            target.extend(ids)
        elif r < wwm_probability*0.9:
            source.extend(ids) 
            target.extend(ids)
        elif r < wwm_probability:
            source.extend(
                np.random.choice(tokenizer.vocab_size-1, size=len(ids))+1
            )
            target.extend(ids)
        else:
            source.extend(ids)
            target.extend([-100]*len(ids))
    if max_length is None:
        return [tokenizer.cls_token_id] + source + [tokenizer.sep_token_id],[-100] + target + [-100]
    else:
        return [tokenizer.cls_token_id] + source[:max_length-2] + [tokenizer.sep_token_id],\
                [-100] + target[:max_length-2] + [-100] 

def whole_word_masking_zh(texts:list, tokenizer, max_length=None, wwm_probability=0.15):
    tokenized_texts = {
        'input_ids': [0]*len(texts),
        'labels': [0]*len(texts),
        # 'attention_mask': [0]*len(texts),
    }
    for idx,text in enumerate(texts):
        source, target = wwm_mlm_zh_encode(text, tokenizer, wwm_probability=wwm_probability, max_length=max_length)
        tokenized_texts['input_ids'][idx] = torch.tensor(source)
        tokenized_texts['labels'][idx] = torch.tensor(target)
    tokenized_texts['input_ids'] = pad_sequence(tokenized_texts['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id)
    tokenized_texts['labels'] = pad_sequence(tokenized_texts['labels'], batch_first=True, padding_value=-100)
    tokenized_texts['attention_mask'] = (tokenized_texts['input_ids'] != tokenizer.pad_token_id).long()
    return tokenized_texts


def wwm_encode_for_seq2seq_zh(text, tokenizer, max_length=None, wwm_probability=0.15):
    assert tokenizer is not None and (max_length==None or type(max_length)==int)
    special_tokens = {tokenizer.mask_token, tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token}
    mask_token_id = tokenizer.mask_token_id
    words = jieba.lcut(text)
    rands = np.random.rand(len(words))
    source, target = [], []
    for r,w in zip(rands, words):
        ids = tokenizer.encode(w, add_special_tokens=False)
        if w in special_tokens: 
            source.extend(ids)  
            target.extend(ids)
        elif r < wwm_probability*0.8:   
            source.extend([mask_token_id])
            target.extend(ids)
        elif r < wwm_probability*0.9:   
            source.extend(ids) 
            target.extend(ids)
        elif r < wwm_probability:   
            source.extend(
                np.random.choice(tokenizer.vocab_size-1, size=len(ids))+1
            )
            target.extend(ids)
        else:   
            source.extend(ids)
            target.extend(ids)
    if max_length is None:
        return [tokenizer.cls_token_id] + source + [tokenizer.sep_token_id],[tokenizer.cls_token_id] + target + [tokenizer.sep_token_id]
    else:
        return [tokenizer.cls_token_id] + source[:max_length-2] + [tokenizer.sep_token_id],\
                [tokenizer.cls_token_id] + target[:max_length-2] + [tokenizer.sep_token_id]