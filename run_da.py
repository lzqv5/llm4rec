import torch, os, argparse, logging, json
import torch.utils.data as data
from transformers import BertTokenizer, BertForMaskedLM, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from utils import set_logger
import mlm_utils

parser = argparse.ArgumentParser(description='Train Baselines')
parser.add_argument('--model_name', type=str, required=True, help='set the model\'s name ')
parser.add_argument('--data_path', type=str, required=True, help='set the data path ')
parser.add_argument('--epochs', type=int, required=True, help='set the epochs')
parser.add_argument('--lr', type=float, required=True, help='set the learning rate')
parser.add_argument('--gpu_id', type=int, required=True, help='set the gpu id')
args = parser.parse_args()


def prepareSentences(sentences, maxlen=510, minlen=6):
    for x in sentences:
        sent = x.strip()
        for i in range(0, len(sent), maxlen):
            chunk = sent[i:i+maxlen]
            if len(chunk)>minlen: yield chunk

def train_mlm(model, optimizer, train_dl, model_name, gradient_accumulation_steps=1, report_steps=200, device="cuda:0", epochs=3, scheduler=None):
    for epoch in range(epochs):
        losses = []
        model.train()
        logging.info(f'\nEpoch {epoch+1} / {epochs}')
        for step, batch in enumerate(tqdm(train_dl)):
            output = model(input_ids=batch[0].to(device), attention_mask=batch[1].to(device), labels=batch[2].to(device))
            loss = output.loss

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()

            if step % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            losses.append(loss.item()*gradient_accumulation_steps)
            # record
            if step % report_steps == 0:
                logging.info(f'Epoch-{epoch+1}: step:{step}, loss = {loss.item()*gradient_accumulation_steps}')
        torch.save(model.state_dict(), f'./models/{model_name}_epoch_{epoch+1}.pth')
        logging.info(f'Avg loss of Eopch-{epoch+1}: {sum(losses)/len(losses)}')

if __name__ == "__main__":
    # os.environ['CUDA_LAUNCH_BLOCKING']='1'
    model_name = args.model_name
    lr = args.lr
    max_epoch = args.epochs

    log_path=f'./logs/{model_name}_lr_{lr}_.log'
    set_logger(log_path)

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"

    #^ params
    batch_size_per_gpu = 48
    n_gpu = 1
    gradient_accumulation_steps = 3
    adam_epsilon = 1e-8
    weight_decay = 1e-5
    warmup_ratio = 0.1


    #^ load model & tokenizer
    plm = 'hfl/chinese-roberta-wwm-ext' 
    tokenizer = BertTokenizer.from_pretrained(plm) 
    model = BertForMaskedLM.from_pretrained(plm)
    model.to(device)

    #^ convert comments to tokens and then to chunks
    # data_path = './data/fullText_train.json'
    data_path = args.data_path
    with open(data_path) as f:
        cv_jds = json.load(f)
    corpus = []
    for cv,jd,_ in cv_jds:  # [cv, jd, state]
        corpus.extend([cv,jd])
    generator = prepareSentences(corpus) 
    chunks = [chunk for chunk in generator]
    #^ It takes about 2h...
    tokenized_texts = mlm_utils.whole_word_masking_zh(chunks, tokenizer, max_length=512, wwm_probability=0.15)
   
    #^ dataset
    train_set = data.TensorDataset(tokenized_texts['input_ids'], tokenized_texts['attention_mask'], tokenized_texts['labels'])

    #^ train_dataloader
    train_sampler = data.RandomSampler(train_set)   
    params = {"batch_size": batch_size_per_gpu, "sampler": train_sampler}
    train_dataloader = data.DataLoader(train_set, **params) 

    step_tot = (len(train_set) // gradient_accumulation_steps // batch_size_per_gpu // n_gpu) * max_epoch
    warmup_steps = int(0.1 * step_tot)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=step_tot)

    #^ training...
    train_mlm(model=model, optimizer=optimizer, train_dl=train_dataloader, model_name=f'{model_name}_lr_{lr}_',
                gradient_accumulation_steps=gradient_accumulation_steps, report_steps=50, device=device, epochs=max_epoch, scheduler=scheduler)

