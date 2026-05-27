import argparse

parser = argparse.ArgumentParser(description="argparse")

parser.add_argument("-m", "--model", type=str, help="HF model name or path", required=True)
parser.add_argument("-q", "--quantized", action="store_true", help="Enable quantization (only AWQ-Marlin)")
parser.add_argument("-w", "--workers", type=int, help="Number of vLLM workers", default=1)
parser.add_argument("--max_model_len", type=int, help="Maximum model context length", default=None)
parser.add_argument("-t", "--topic", type=str, help="Discussion Topic", required=True)
parser.add_argument("-r", "--reverse", action="store_true", help="Enable reverse (negated) topic framing")
parser.add_argument("-s", "--samples", type=int, help="Number of samples to run", required=True)
parser.add_argument("-l", "--length", type=int, help="Discussion length", required=True)
parser.add_argument("-o", "--output", type=str, help="Output CSV file path")
parser.add_argument("-g", "--gpu", type=str, help="GPU indices to use", required=True)

args = parser.parse_args()
model_name = args.model
quantized = args.quantized
topic = args.topic
rev_flag = args.reverse
samples = args.samples
discussion_len = args.length
output_path = args.output
gpu_idx = args.gpu

if output_path is None:
    output_path = f"../data/{model_name.split('/')[-1]}_{topic.replace(' ', '_')}_s={samples}_l={discussion_len}" + ("_reversed_framing" if rev_flag else "_normal_framing") + ".csv"

import os 
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx # Specify which GPU(s) to use
#os.environ["VLLM_CONFIGURE_LOGGING"] = "0" # Disable vLLM logging (mostly)

import gc
import torch

from dotenv import load_dotenv
load_dotenv()

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vllm import LLM, SamplingParams
from scipy.stats import rv_discrete, entropy


tic = time.time()


# Utility functions for prompt generation
def prompt_template(user_prompt: str, role: str="system", content: str="You are a helpful AI assistant.") -> list:
    return [
        {"role": role, "content": content},
        {"role": "user", "content": user_prompt},
    ]  

def topic_statements(topic: str, reverse: bool=False) -> str:
    statements = {
        # Societal issues with scientific consensus
        "Climate Change": "Climate Change is an important problem to address",
        "Climate Change rev": "Climate Change is not an important problem to address",
        "Vaccination": "Vaccinations are necessary for public health",
        "Vaccination rev": "Vaccinations are not necessary for public health",
        "Shape of the Earth": "The Earth is flat",
        "Shape of the Earth rev": "The Earth is not flat",
        
        # Political issues
        "Global Wealth Distribution": "The global distribution of wealth is fair",
        "Global Wealth Distribution rev": "The global distribution of wealth is not fair",
        "Abortion": "Abortions should be legal",
        "Abortion rev": "Abortions should not be legal",
        "Social Media": "Social media does have a positive impact on society",
        "Social Media rev": "Social media does not have a positive impact on society",
        
        # Philosophical issues
        "Artificial Intelligence": "Artificial Intelligence is dangerous",
        "Artificial Intelligence rev": "Artificial Intelligence is not dangerous",
        "Morality and Religion": "You can only be a moral person if you believe in God",
        "Morality and Religion rev": "You cannot only be a moral person if you believe in God",
        "Free Will": "Humans possess free will",
        "Free Will rev": "Humans do not possess free will",

        # Personal preferences
        "Musical Preference": "Bach is a greater composer than Stravinsky",
        "Musical Preference rev": "Bach is not a greater composer than Stravinsky",
        "Food Preference": "Pizza is better than sushi",
        "Food Preference rev": "Pizza is not better than sushi",
        "Art Style Preference": "Modern art is more meaningful than classical art",
        "Art Style Preference rev": "Modern art is not more meaningful than classical art"
    }
    
    if reverse:
        return statements[f"{topic} rev"]
    else:
        return statements[topic]

    
def opinion_scale(opinion: int) -> str:
    d = {1: 'strongly disagrees', 2: 'disagrees', 3: 'neither agrees nor disagrees', 4: 'agrees', 5: 'strongly agrees'}
    return d[opinion]

def cot_prompt(topic: str, opinion: int, reverse: bool) -> str:
    reverse_mapping = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1}
    if reverse:
        opinion = reverse_mapping[opinion]
    system_prompt = f'You are a person who {opinion_scale(opinion)} that {topic_statements(topic)}.'
    user_prompt = f'Write out your thoughts about {topic}. Keep your output short.'
    return prompt_template(user_prompt, content=system_prompt)

def opinion_probing_prompt(statement:str) -> str:
    p = f'State your opinion about the following statement: "{statement}." '\
        'Provide your answer as one of the options "1: strongly disagree", "2: disagree", "3: neither agree nor disagree", "4: agree", "5: strongly agree". '\
        'Only return the corresponding integer value without any comments or punctuation.'
    return p 

def initiate_agent_prompt(topic: str, cot: str) -> str:
    return f'You are a person who has the following thoughts about {topic}: "{cot}" Always act and answer as this person.'

def discussion_prompt(topic: str, type: str, r_text: str=None, rev_probing: bool=False) -> str:
    prompt = f"From now on, you are part of a new discussion about {topic}. "
    
    if type == 'init_start':
        prompt += "Write three sentences to start the discussion."
        
    elif type == 'init_response':
        prompt += f'Someone else wrote the following text: "{r_text}". Write three sentences as your response.'
        
    elif type == 'reply': 
        prompt = f'Someone else replied to you with the following text: "{r_text}". Write three sentences as your response.'
    
    elif type == 'reply_probe':
        prompt = f'Someone else replied to you with the following text: "{r_text}". ' + opinion_probing_prompt(topic_statements(topic, reverse=rev_probing))
    
    elif type == 'reply_probe_init':
        prompt += f'Someone else wrote the following text: "{r_text}". ' + opinion_probing_prompt(topic_statements(topic, reverse=rev_probing))
    
    return prompt

def init_dataframe():
    df = pd.DataFrame({
        "discussion_id": pd.Series(dtype='int'), "t": pd.Series(dtype='int'), 
        "init_x_i": pd.Series(dtype='int'), "x_i": pd.Series(dtype='float'), "H_i": pd.Series(dtype='float'), "text_i": pd.Series(dtype='str'),
        "init_x_j": pd.Series(dtype='int'), "x_j": pd.Series(dtype='float'), "H_j": pd.Series(dtype='float'), "text_j": pd.Series(dtype='str'),
    })
    return df



# pre define dataframe for data storage
df = init_dataframe()

disc_id = 0
for _ in range(samples):
    for init_x_i in range(-2,3):
        for init_x_j in range(-2,3):
            df.loc[len(df)] = [disc_id, 0, init_x_i, np.nan, np.nan, pd.NA, init_x_j, np.nan, np.nan, pd.NA]
            for t in range(1, discussion_len+1):
                df.loc[len(df)] = [disc_id, t, init_x_i, np.nan, np.nan, pd.NA, init_x_j, np.nan, np.nan, pd.NA]

            disc_id += 1


# Load LLM, given a HuggingFace model name
if args.max_model_len is not None and not quantized:
    llm = LLM(model=model_name, dtype="auto", max_model_len=args.max_model_len, tensor_parallel_size=args.workers)

elif args.max_model_len is not None and quantized:
    llm = LLM(model=model_name, dtype="auto", quantization='awq_marlin', max_model_len=args.max_model_len, tensor_parallel_size=args.workers)

elif quantized:
    llm = LLM(model=model_name, dtype="auto", quantization='awq_marlin', tensor_parallel_size=args.workers) # for additional arguments see https://docs.vllm.ai/en/v0.8.1/api/offline_inference/llm.html

else:
    llm = LLM(model=model_name, dtype="auto", tensor_parallel_size=args.workers)


# Functions for oinion extraction from LLM outputs
tokenizer = llm.get_tokenizer()
targets = np.arange(1,6).astype(str)
target_ids = []                         

for target in targets:
    token_id = tokenizer.encode(target, add_special_tokens=False)[-1]
    target_ids.append(token_id)

def expected_opinion(output, target_ids) -> np.array:
    logprobs = output.outputs[0].logprobs
    target_probs = np.zeros((5,2))
    target_probs[:,0] = np.arange(1,6)

    
    for probs in logprobs:
        if list(probs.keys())[0] in target_ids: # check if any target token is in the top logprobs
            for key in list(probs.keys()): 
                if key in target_ids:
                    try:
                        token = int(probs[key].decoded_token)
                        if token in targets.astype(int):
                            target_probs[token-1, :] = token, np.exp(probs[key].logprob)
                    except:
                        pass

    try:
        # normalize probabilities
        target_probs[:, 1] = target_probs[:, 1] / target_probs[:, 1].sum() 

        op = rv_discrete(values=(target_probs[:,0], target_probs[:,1]))
        H = entropy(pk = target_probs[:,1])
        expected_opinion = np.array([op.mean(), H])
    except:
        expected_opinion = np.array([np.nan, np.nan])
    
    return expected_opinion



sampling_params_text = SamplingParams(
    temperature=1,
    max_tokens=1024,        # increase as needed (e.g. 1024)
    logprobs=0          # get logprobs for top 5 tokens
)

sampling_params_probing = SamplingParams(
    temperature=1,
    max_tokens=5,        # increase as needed (e.g. 1024)
    logprobs=10          # get logprobs for top 5 tokens
)



# Generate CoT monologues for all opinions on the given topic
conversations = []

for runs in range(samples):
    for x_i in range(1,6):
        for x_j in range(1,6):
            conversations.append(cot_prompt(topic, x_i, reverse=rev_flag))
            conversations.append(cot_prompt(topic, x_j, reverse=rev_flag))

outputs = llm.chat(messages=conversations, sampling_params=sampling_params_text)




# Save monologues 
monologues = []

for output in outputs:
    monologues.append(output.outputs[0].text)

for i in range(len(monologues)//2):
    cot_i = monologues[2*i]
    cot_j = monologues[2*i+1]
    
    disc_id = i
    df.loc[(df['discussion_id'] == disc_id) & (df['t'] == 0), 'text_i'] = cot_i
    df.loc[(df['discussion_id'] == disc_id) & (df['t'] == 0), 'text_j'] = cot_j



# Probe initial opinions
prompts = []

for m in monologues:
    prompts.append(prompt_template(user_prompt=opinion_probing_prompt(topic_statements(topic, reverse=rev_flag)), content=initiate_agent_prompt(topic, m)))

probings = llm.chat(messages=prompts, sampling_params=sampling_params_probing)

expected_opinions = np.zeros((len(probings), 2))

for i, output in enumerate(probings):
    expected_opinions[i] = expected_opinion(output, target_ids)




# Save initial expected opinions
for i in range(len(expected_opinions)//2):
    x_i = expected_opinions[2*i, 0]
    H_i = expected_opinions[2*i, 1]
    x_j = expected_opinions[2*i+1, 0]
    H_j = expected_opinions[2*i+1, 1]
    
    disc_id = i
    df.loc[(df['discussion_id'] == disc_id) & (df['t'] == 0), 'x_i'] = x_i-3
    df.loc[(df['discussion_id'] == disc_id) & (df['t'] == 0), 'H_i'] = H_i
    df.loc[(df['discussion_id'] == disc_id) & (df['t'] == 0), 'x_j'] = x_j-3
    df.loc[(df['discussion_id'] == disc_id) & (df['t'] == 0), 'H_j'] = H_j


# prompts for agent i and genereate discussion starting messages
prompts_i = []

for m in monologues[::2]:
    prompts_i.append(prompt_template(user_prompt=discussion_prompt(topic, type="init_start"), content=initiate_agent_prompt(topic, m)))

outputs = llm.chat(messages=prompts_i, sampling_params=sampling_params_text)


# prompts for agent j responses to agent i starting messages
prompts_j = []
for i in range(len(outputs)):
    disc_id = i
    msg_i = outputs[i].outputs[0].text
    prompts_i[i].append({"role": "assistant", "content": msg_i})
    df.loc[(df['discussion_id'] == disc_id) & (df['t'] == 1), 'text_i'] = msg_i
    prompts_j.append(prompt_template(user_prompt=discussion_prompt(topic, type="init_response", r_text=msg_i), content=initiate_agent_prompt(topic, monologues[2*i+1])))

outputs = llm.chat(messages=prompts_j, sampling_params=sampling_params_text)


for i in range(len(outputs)):
    disc_id = i
    msg_j = outputs[i].outputs[0].text
    prompts_j[i].append({"role": "assistant", "content": msg_j})
    df.loc[(df['discussion_id'] == disc_id) & (df['t'] == 1), 'text_j'] = msg_j

    prompts_i[i].append({"role": "user", "content": discussion_prompt(topic, type="reply_probe", r_text=msg_j, rev_probing=rev_flag)}) # append prompt for probing
    prompts_j[i].append({"role": "user", "content": opinion_probing_prompt(topic_statements(topic, reverse=rev_flag))})


# Probe agent i

probings = llm.chat(messages=prompts_i, sampling_params=sampling_params_probing)

for i, output in enumerate(probings):
    x_i, H_i = expected_opinion(output, target_ids)
    disc_id = i
    df.loc[(df['discussion_id'] == disc_id) & (df['t'] == 1), 'x_i'] = x_i-3
    df.loc[(df['discussion_id'] == disc_id) & (df['t'] == 1), 'H_i'] = H_i


# Probe agent j
probings = llm.chat(messages=prompts_j, sampling_params=sampling_params_probing)

for i, output in enumerate(probings):
    x_j, H_j = expected_opinion(output, target_ids)
    disc_id = i
    df.loc[(df['discussion_id'] == disc_id) & (df['t'] == 1), 'x_j'] = x_j-3
    df.loc[(df['discussion_id'] == disc_id) & (df['t'] == 1), 'H_j'] = H_j


# Remove last probing prompt from prompts before next discussion round
for p in range(len(prompts_i)):
    prompts_i[p] = prompts_i[p][:-1]
    prompts_j[p] = prompts_j[p][:-1]


# loop over the remaining discussion rounds

for t in range(2, discussion_len+1):

    # Prompt agent i to reply to agent j's last message
    for n in range(len(prompts_i)):
        prompts_i[n].append({"role": "user", "content": discussion_prompt(topic, type="reply", r_text=df[df["t"]==t-1]["text_j"].iloc[n])})
    
    outputs = llm.chat(messages=prompts_i, sampling_params=sampling_params_text) # Generate replies from agent i
        
    for n in range(len(outputs)):
        msg_i = outputs[n].outputs[0].text
        prompts_i[n].append({"role": "assistant", "content": msg_i}) # Save reply in prompt history
        df.loc[(df['discussion_id'] == n) & (df['t'] == t), 'text_i'] = msg_i # Save reply in dataframe
        prompts_j[n].append({"role": "user", "content": discussion_prompt(topic, type="reply", r_text=msg_i)}) # Prepare prompt for agent j to reply to agent i
        
    outputs = llm.chat(messages=prompts_j, sampling_params=sampling_params_text) # Generate replies from agent j

    for n in range(len(outputs)):
        msg_j = outputs[n].outputs[0].text
        prompts_j[n].append({"role": "assistant", "content": msg_j}) # Save reply in prompt history
        df.loc[(df['discussion_id'] == n) & (df['t'] == t), 'text_j'] = msg_j # Save reply in dataframe

        prompts_i[n].append({"role": "user", "content": discussion_prompt(topic, type="reply_probe", r_text=msg_j, rev_probing=rev_flag)}) # append prompt for probing
        prompts_j[n].append({"role": "user", "content": opinion_probing_prompt(topic_statements(topic, reverse=rev_flag))})

    # Probe agent i
    probings = llm.chat(messages=prompts_i, sampling_params=sampling_params_probing)

    for i, output in enumerate(probings):
        x_i, H_i = expected_opinion(output, target_ids)
        disc_id = i
        df.loc[(df['discussion_id'] == disc_id) & (df['t'] == t), 'x_i'] = x_i-3
        df.loc[(df['discussion_id'] == disc_id) & (df['t'] == t), 'H_i'] = H_i


    # Probe agent j
    probings = llm.chat(messages=prompts_j, sampling_params=sampling_params_probing)

    for i, output in enumerate(probings):
        x_j, H_j = expected_opinion(output, target_ids)
        disc_id = i
        df.loc[(df['discussion_id'] == disc_id) & (df['t'] == t), 'x_j'] = x_j-3
        df.loc[(df['discussion_id'] == disc_id) & (df['t'] == t), 'H_j'] = H_j

    # Remove last probing prompt from prompts before next discussion round
    for p in range(len(prompts_i)):
        prompts_i[p] = prompts_i[p][:-1]
        prompts_j[p] = prompts_j[p][:-1]


df.to_csv(output_path, index=False)

toc = time.time()

print("Sucesfully finished. Total execution time:", toc - tic, flush=True)


# Delete the llm object and free the memory
llm.llm_engine.engine_core.shutdown()
del llm
gc.collect()
torch.cuda.empty_cache()

print(f"Results saved to {output_path}", flush=True)
print("", flush=True)
print("", flush=True)
print("--------------------------------------------------", flush=True)
print("", flush=True)
print("", flush=True)

