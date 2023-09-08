from tqdm.notebook import tqdm
from time import time, sleep
from ast import literal_eval


import os, json, argparse
import re
from langchain.output_parsers import StructuredOutputParser, ResponseSchema # 用于解析 LLM output
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate # 用于生成 LLM message prompt
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate, # 基于 messages 生成最终输入到模型中的 prompt;
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


os.environ["OPENAI_API_KEY"] = "API_KEY" 
summarizer = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.3, n=1, max_tokens=512)

# output parser
class JsonExtractor():

    def __init__(self) -> None:
        self.pattern_json_extractor = re.compile(r'{[^\{\}]+}')
        self.pattern_remove_space = re.compile(r'\n\s{0,}')
        self.pattern_refine = re.compile(r'(?<![a-zA-Z])\'(?=[a-zA-Z])|(?<=[a-zA-Z])\'(?![a-zA-Z])')
        self.keys = ["overall", "education", "education comment", "experience", "experience comment", "skill", "skill comment"]
        self.descriptions = ["Total score", "Educational score", "comment on the educational score", "work experience score", "comment on the work experience score", "skill score", "comment on the skill score"]
        self.response_schemas = [ResponseSchema(name=name, description=desc) for name, desc in zip(self.keys, self.descriptions)]
        self.output_parser = StructuredOutputParser(response_schemas=self.response_schemas)
    
    def _extract(self, text: str) -> str:
        return self.pattern_json_extractor.search(text).group()
    
    def _remove_space(self, text: str) -> str:
        return self.pattern_remove_space.sub('', text)
    
    def _refine_property_name_quotes(self, text: str) -> str:
        return self.pattern_refine.sub('"', text)
    
    def _convert_to_md_json(self, text: str) -> str:
        return '```json'+text+'```'
    
    def __call__(self, text: str) -> dict:

        # 用于解析 LLM output
        text = self._remove_space(self._extract(text))
        try:
            return literal_eval(text)
        except json.decoder.JSONDecodeError:
            print('wrong text format:\n', text)
            return literal_eval(self._refine_property_name_quotes(text))

def build_chat_prompt(prompt_idx=0):
    human_message_prompt = """你是一个专业的HR，给定一对(职位描述,简历)，你能够判断该岗位描述和该简历是否匹配并进行评论。

    给你如下格式的输入：
    ```
    job description:
    a paragraph of text

    resume:
    a paragraph of text
    ```

    你需要输出以下内容：
    1. 简历和岗位的综合匹配分数（百分制，0-100分）
    2. 简历和岗位在“教育背景”方面的匹配分数（百分制，0-100分）
    3. 简历和岗位在“工作经历”方面的匹配分数（百分制，0-100分）
    4. 简历和岗位在“专业技能”方面的匹配分数（百分制，0-100分）

    输出结果以如下键值对的JSON格式表示：
    ```
    "overall": a number between 0 and 100,
    "education": a number between 0 and 100,
    "education comment": a detailed paragraph of comment on the education score
    "experience": a number between 0 and 100,
    "experience comment": a detailed paragraph of comment on the experience score
    "skill": a number between 0 and 100,
    "skill comment": a detailed paragraph of comment on the skill score
    ```

    """
    if prompt_idx == 0:
        human_message_prompt += """你在匹配过程中应该关注岗位描述中的"工作经历"，"职位描述"和"职位要求"的内容。你也需要关注简历中的"工作年限"，"教育背景"和"工作经历"的内容。
        是否匹配的依据应根据输入的岗位描述和简历内容动态生成。
        你应该提供有关匹配过程的任何其他信息或反馈，并对简历中可能存在的错误或缺失信息进行自动处理。如果有错误，你应该跳过缺失的信息并继续完成匹配。
        你的第一个回复应该只是“已理解”。
        """
    elif prompt_idx == 1:
        human_message_prompt += """请根据提供的简历和岗位描述，按照上述格式输出多个0-100之间的各字段匹配分数。你可以处理各种类型的文本，并且请在没有足够信息的情况下给出合理的反馈。
        你的第一个回复应该只是“已理解”。
        """
    else:
        human_message_prompt += """你需要按照“教育背景”、“工作经历”和“专业技能”三个指标，基于简历和岗位的匹配程度，返回三个指标的匹配分数，并返回一个能总体上反映简历与岗位匹配程度的综合分数。
        你的第一个回复应该只是“已理解”。
        """
    assistant_message_prompt = "已理解。"
    input_prompt = """job description:
    {jd}

    resume:
    {cv}

    请输出JSON格式的结果
    """
    human_message_prompt_template = HumanMessagePromptTemplate.from_template(human_message_prompt)
    assistant_message_prompt_template = AIMessagePromptTemplate.from_template(assistant_message_prompt)
    input_prompt_template = HumanMessagePromptTemplate.from_template(input_prompt)
    chat_prompt= ChatPromptTemplate.from_messages([human_message_prompt_template, assistant_message_prompt_template, input_prompt_template])
    return chat_prompt

def build_comment_prompt():
    summarize_prompt = """你是一个专业的HR。现在，有数位专家对同一份简历进行了评论。你需要根据这些评论，生成一段简短的评语，来描述这份简历的优势和不足。

    输出结果以如下键值对的JSON格式表示:
    ```
    "comments": a comment paragraph
    ```

    以下是这些专家的评论:
    ```
    {comments}
    ```

    请输出JSON格式的结果
    """
    prompt= ChatPromptTemplate.from_messages([HumanMessagePromptTemplate.from_template(summarize_prompt)])
    return prompt

cmts_prompt = build_comment_prompt()

def rebuild_output(mean_score, outputs, output_parser):
    flag = 1e3
    idx = 0
    for i, output in enumerate(outputs):
        diff = abs(output.get('overall',0)-mean_score)
        if diff<flag:
            flag = diff
            idx = i
    picked_output = outputs[idx]
    picked_output['overall'] = mean_score

    edu_cmts = ''
    exp_cmts = ''
    ski_cmts = ''
    for i in range(len(outputs)):
        edu_cmts += 'education comment' + outputs[i]['education comment'] + '\n'
        exp_cmts += 'experience comment' + outputs[i]['experience comment'] + '\n'
        ski_cmts += 'skill comment' + outputs[i]['skill comment'] + '\n'
    edu_cmts = summarizer(cmts_prompt.format_prompt(comments=edu_cmts).to_messages())
    picked_output['education comment'] = output_parser(edu_cmts.content)
    exp_cmts = summarizer(cmts_prompt.format_prompt(comments=exp_cmts).to_messages())
    picked_output['experience comment'] = output_parser(exp_cmts.content)
    ski_cmts = summarizer(cmts_prompt.format_prompt(comments=ski_cmts).to_messages())
    picked_output['skill comment'] = output_parser(ski_cmts.content)
    return picked_output

def single_session_processing(generation: list, output_parser: JsonExtractor):
    n = len(generation)
    outputs = [output_parser(generation[i].text) for i in range(n)]
    mean_score = sum([d.get('overall', 0) for d in outputs])/n
    return rebuild_output(mean_score, outputs, output_parser)

def sessions_processing(generations: list, output_parser: JsonExtractor):
    return [single_session_processing(gen, output_parser) for gen in generations]

#! 付费用户使用频率 - 3,500 RPM or 90,000 TPM - chat
#! 假设每次调用token数有预估, 那么 TPM 可以转化为等价的 RPM

# 给定 jd, cv 和 candidates, 自动化地生成大模型排序后的结果
def resorting(jd, cv, candidates, chat_model, chat_prompt, output_parser,  rate_limit_per_min=3500, tokens_per_min=90000, token_per_req=4000, delta=2.5):
    #! 每执行一次 resorting，就调用一次或者多次 LLM api;
    #! 调用 api 的实际频率要看实际调用 OpenAI API 的频率，而不是 LangChain 调用的频率;
    #! 需要限制 resorting 的调用频率
    if not isinstance(chat_prompt, list):
        # 只有一个 prompt
        n = chat_model.n
        rpm = min(rate_limit_per_min, tokens_per_min//token_per_req)
        step = rpm//n
        messages_list = [[chat_prompt.format_prompt(jd=jd, cv=cand).to_messages() for cand in candidates[i:i+step]] for i in range(0,len(candidates),step)]
        generations = []
        for messages in messages_list:  # 1 batch/min
            start_time = time()
            responses_of_generation = chat_model.generate(messages)
            generations.extend(responses_of_generation.generations)
            end_time = time()
            diff = end_time-start_time
            if diff < 60:
                print("sleeping...")
                sleep(60-diff)

        outputs = sessions_processing(generations, output_parser)
        cands_new = [item for item in zip(candidates,outputs)]
    else:   
        prompt_nums = len(chat_prompt)
        candidates_nums = len(candidates)
        n = chat_model.n
        rpm = min(rate_limit_per_min, tokens_per_min//token_per_req)
        step = rpm//n
        outputs = []
        for prompt in chat_prompt:
            messages_list = [[prompt.format_prompt(jd=jd, cv=cand).to_messages() for cand in candidates[i:i+step]] for i in range(0,candidates_nums,step)]
            generations = []
            for messages in messages_list:  # 1 batch/min
                # print("Begin")
                start_time = time()
                responses_of_generation = chat_model.generate(messages)
                generations.extend(responses_of_generation.generations)
                end_time = time()
                # print("End")
                diff = end_time-start_time
                if diff < 60:
                    print("sleeping...")
                    sleep(60-diff)
            outputs.append(sessions_processing(generations, output_parser))

        mean_scores = [sum([outputs[i][j].get('overall',0) for i in range(prompt_nums)])/prompt_nums for j in range(candidates_nums)]
        outputs_new = [rebuild_output(mean_score=mean_scores[j], outputs=[outputs[i][j] for i in range(prompt_nums)], output_parser=output_parser) for j in range(candidates_nums)]
        cands_new = [item for item in zip(candidates, outputs_new)]

    cands_new.sort(key=lambda item:item[1].get('overall',0),reverse=True)
    return [jd, cv, cands_new]
            

def resort_self_consistency(chat_model, recs, output_parser: JsonExtractor):
    chat_prompt = build_chat_prompt(prompt_idx=0)
    resorted_recs = []
    for idx, (jd, cv, cands, _) in enumerate(recs):
        print(f"idx: {args.start_idx+idx} - processing...")
        resorted_recs.append(resorting(jd, cv, cands, chat_model, chat_prompt, output_parser))
        with open(f'data/recs_self_consistency_(cv{args.cv_len}_jd500)_temperature_{args.temperature}_n_{args.n}.json', 'w') as f:
            json.dump(resorted_recs, f, ensure_ascii=False)
    return resorted_recs

def resort_ensemble(chat_model, recs, output_parser: JsonExtractor, prompt_ids=[0,1,2], rate_limit_per_min=3, delta=2.5):
    chat_prompts = [build_chat_prompt(prompt_idx=idx) for idx in prompt_ids]
    resorted_recs = []
    for idx, (jd, cv, cands, _) in enumerate(recs):
        # start_time = time()
        print(f"idx: {args.start_idx+idx} - processing...")
        resorted_recs.append(resorting(jd, cv, cands, chat_model, chat_prompts, output_parser))
        with open(f'data/recs_ensemble_(cv{args.cv_len}_jd500)_temperature_{args.temperature}_n_{args.n}.json', 'w') as f:
            json.dump(resorted_recs, f, ensure_ascii=False)
    return resorted_recs

parser = argparse.ArgumentParser(description='Pipeline Settings')
parser.add_argument('--cv_len', type=int, required=True, help='set the token_num of cv')
parser.add_argument('--start_idx', type=int, required=True, help='')
parser.add_argument('--temperature', type=float, required=True, help='')
parser.add_argument('--n', type=int, required=True, help='')
args = parser.parse_args()

if __name__ == '__main__':
    # load llm
    chat_diverse = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=args.temperature, n=args.n, max_tokens=512)
    chat_ensemble_only = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=args.temperature, n=args.n, max_tokens=512)

    with open(f'data/output_from_stage_1.json', 'r') as f:
        recs = json.load(f)

    output_extractor = JsonExtractor()
    recs = recs[args.start_idx:]

    print("Self-consistency ...")
    recs_self_consistency = resort_self_consistency(chat_model=chat_diverse, recs=recs, output_parser=output_extractor)
    print("Done.")
    print("Self-consistency results saving...")
    with open(f'data/recs_self_consistency_(cv{args.cv_len}_jd500)_temperature_{args.temperature}_n_{args.n}.json', 'w') as f:
        json.dump(recs_self_consistency, f, ensure_ascii=False)

    # print("Ensemble-only ...")
    # recs_ensemble =resort_ensemble(chat_model=chat_ensemble_only, recs=recs, output_parser=output_extractor, prompt_ids=[0,1,2])
    # print("Done.")
    # print("Ensemble-only results saving...")
    # with open(f'data/recs_ensemble_(cv{args.cv_len}_jd500)_temperature_{args.temperature}_n_{args.n}.json', 'w') as f:
    #     json.dump(recs_ensemble, f, ensure_ascii=False)

    # print("Combined ...")
    # recs_combined = resort_ensemble(chat_model=chat_diverse, recs=recs, output_parser=output_extractor, prompt_ids=[0,1,2])
    # print("Done.")
    # print("Combined results saving...")
    # with open(f'data/recs_combined_(cv{args.cv_len}_jd500)_temperature_{args.temperature}_n_{args.n}.json', 'w') as f:
    #     json.dump(recs_combined, f, ensure_ascii=False)

    print('All finished')
