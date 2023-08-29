import torch, json
import torch.utils.data as data
import torch.nn.functional as F
from transformers import AutoTokenizer, BertModel
from tqdm.notebook import tqdm

max_input_len = 512
batch_size_per_gpu = 16

def get_sentences_emb(model, text_list):
    inputs = tokenizer(text_list, truncation=True, padding='max_length', max_length=max_input_len, return_tensors='pt')
    eval_set = data.TensorDataset(inputs.input_ids,inputs.attention_mask)
    eval_sampler = data.SequentialSampler(eval_set)
    params = {"batch_size": batch_size_per_gpu, "sampler": eval_sampler}
    eval_dataloader = data.DataLoader(eval_set, **params)
    s_embs = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.no_grad():
            # outputs = model(input_ids = batch[0].to('cuda'),
            #                 attention_mask = batch[1].to('cuda'))
            outputs = model(input_ids = batch[0].to(device),
                            attention_mask = batch[1].to(device))
            pooler_output = outputs.pooler_output
            s_embs.append(pooler_output)
    s_embs = torch.cat(s_embs, 0)
    return s_embs

def cal_similarities(model, cv_list, jd_list):
    jd_embs = get_sentences_emb(model, jd_list)
    cv_embs = get_sentences_emb(model, cv_list)
    dev_set_size = cv_embs.shape[0]
    cv_jd_cos = [0]*dev_set_size
    for i in tqdm(range(dev_set_size)):
        cv = cv_embs[i]
        one_cv_vs_all_jds_cos = F.cosine_similarity(cv, jd_embs).tolist()
        cv_jd_cos[i] = one_cv_vs_all_jds_cos
    return cv_embs, jd_embs, cv_jd_cos.T


def recommendations_for_jds(jds, cvs, cos_mat, k):
    assert len(cvs) == len(jds)
    num_cvs = len(cvs)
    labels = torch.arange(num_cvs)  # labels.shape =[N]
    topk_preds = cos_mat.topk(k=k, dim=1)   # topk_preds.shape = [N,k]
    recommendations = []
    for jd_idx,cv_rec_idxs in enumerate(topk_preds.indices.tolist()):
        recommendations.append([jds[jd_idx], cvs[jd_idx], [cvs[cv_idx] for cv_idx in cv_rec_idxs]])
    return recommendations


if __name__ == '__main__':
    with open('data/deduplicated_devset.json') as f:
        cv_jds = json.load(f)
    print('# cv-jd pairs: ',len(cv_jds))
    p_cvs = [cv_jd[0] for cv_jd in cv_jds]
    p_jds = [cv_jd[1] for cv_jd in cv_jds]
    states = {(cv_jd[1], cv_jd[0]):[] for cv_jd in cv_jds}
    maxnum = 0
    for cv_jd in cv_jds:
        states[(cv_jd[1], cv_jd[0])].append(cv_jd[2])
        maxnum = max(len(states[(cv_jd[1], cv_jd[0])]), maxnum)

    device = 'cuda:0'
    # device = 'cuda:1'
    tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    model = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
    model.load_state_dict(torch.load("./models/xxx.pth", map_location=device))
    cv_embs, jd_embs, jd_cv_cos = cal_similarities(model, p_cvs, p_jds)
    jd_cv_cos_pt = torch.tensor(jd_cv_cos)
 
    recs = recommendations_for_jds(p_jds, p_cvs, jd_cv_cos_pt, 5)