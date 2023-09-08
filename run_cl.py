import torch, json, argparse
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from transformers import AutoTokenizer, BertModel, BertForMaskedLM, AdamW, get_linear_schedule_with_warmup
from pytorch_metric_learning.losses import NTXentLoss
from tqdm import tqdm
from copy import deepcopy

import logging
from utils import set_logger

parser = argparse.ArgumentParser(description='Train Baselines')
parser.add_argument('--model_name', type=str, required=True, help='set the model\'s name')
parser.add_argument('--data_path', type=str, required=True, help='set the source of texts')
parser.add_argument('--epochs', type=int, required=True, help='set the epochs')
parser.add_argument('--gpu_id', type=int, required=True, help='set the gpu id')
parser.add_argument('--mlm_epoch', type=int, required=True, help='set the mlm epoch')
parser.add_argument('--max_length', type=int, required=True, help='set the maximum length of input')
parser.add_argument('--batch_size', type=int, required=True, help='batch_size')
parser.add_argument('--model_size', type=str, required=True, help='model_size')
args = parser.parse_args()

if __name__ == "__main__":
    model_name = args.model_name
    data_path = args.data_path
    model_size = args.model_size

    log_path = f'./logs/{model_name}.log'
    set_logger(log_path)

    with open(data_path) as f:
        cv_jds = json.load(f)
    p_cvs = [cv_jd[0] for cv_jd in cv_jds]
    p_jds = [cv_jd[1] for cv_jd in cv_jds]

    tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    max_input_len = args.max_length
    if data_path == 'data/extra_trunText600_data.json':
        split_point = num_labels = 85000 
    elif data_path == 'data/extra_trunText600_data_with_state.json':
        split_point = 85000
    else:
        raise ValueError("data_path is not valid!")

    jd_inputs = tokenizer(p_jds[:split_point], truncation=True, padding='max_length', max_length=max_input_len, return_tensors='pt')
    cv_inputs = tokenizer(p_cvs[:split_point], truncation=True, padding='max_length', max_length=max_input_len, return_tensors='pt')
    labels = [[i] * 2 for i in range(num_labels)] 
    labels = torch.LongTensor(labels)


    logging.info("-------- Dataset Build! --------")
    train_set = data.TensorDataset(jd_inputs.input_ids, jd_inputs.attention_mask,
                               cv_inputs.input_ids, cv_inputs.attention_mask,
                               labels)

    device = f'cuda:{args.gpu_id}'
    if model_size == "large": model = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext-large')
    else: model = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
    model_mlm = BertForMaskedLM.from_pretrained('hfl/chinese-roberta-wwm-ext')
    model_mlm.load_state_dict(torch.load(f"./models/MLMBasedOnCVJD_lr_1e-05__epoch_{args.mlm_epoch}.pth", map_location='cpu'))
    model.encoder = deepcopy(model_mlm.bert.encoder)
    model.embeddings = deepcopy(model_mlm.bert.embeddings)
    model_mlm = None
    model.to(device)

    batch_size_per_gpu = args.batch_size
    n_gpu = 1
    gradient_accumulation_steps = 2
    lr = 1e-5
    adam_epsilon = 1e-8
    weight_decay = 1e-5
    max_epoch = args.epochs

    step_tot = (len(train_set) // gradient_accumulation_steps // batch_size_per_gpu // n_gpu) * max_epoch
    warmup_steps = int(0.1 * step_tot)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=step_tot)

    # train_dataloader
    train_sampler = data.RandomSampler(train_set)
    params = {"batch_size": batch_size_per_gpu, "sampler": train_sampler, 'drop_last':True}
    train_dataloader = data.DataLoader(train_set, **params) 

    ntxloss = NTXentLoss(temperature=0.05)

    print("Training...")
    report_steps = 200
    for i in range(max_epoch):
        print(f"Epoch {i+1} begin...")
        losses = []
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader)):
            labels = batch[4].to(device).view(-1)
            # jd_embs.shape = [16, 768]
            jd_embs = model(input_ids=batch[0].to(device),
                            attention_mask=batch[1].to(device)).pooler_output
            # cv_embs.shape = [16, 768]
            cv_embs = model(input_ids=batch[2].to(device),
                            attention_mask=batch[3].to(device)).pooler_output
            # jd_cv_embs.shape = [32, 768] 
            jd_cv_embs = torch.cat((jd_embs, cv_embs), 1).view(batch_size_per_gpu*2, -1)

            loss = ntxloss(jd_cv_embs, labels)

            # accumulation
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()
            if step % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
            
            losses.append(loss.item())  # new added

            # record
            if step % report_steps == 0:
                logging.info(f'Epoch-{i}: step:{step}, loss = {loss}')
        torch.save(model.state_dict(), f'./models/{model_name}_epoch_{i}.pth')
        logging.info(f'Avg loss of Eopch-{i}: {sum(losses)/len(losses)}')
