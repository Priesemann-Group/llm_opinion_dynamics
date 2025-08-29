### general imports ###
import os
import numpy as np
from scipy.stats import rv_discrete, entropy
import textwrap as twr


### imports for local LLM ###
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


### imports for ChatGPT###
from openai import OpenAI
import tiktoken


hf_access_token = '' # Hugging Face access token


def norm_probs(values, tokens):
    targets = np.arange(1, 6)
    scores = np.zeros((len(targets), 2))
    
    c = 0
    for tok, val in zip(tokens, values):
        try:
            if int(tok) in targets:
                scores[c][1] = int(tok)
                scores[c][0] = val
                c += 1
        except:
            pass

    mask = (scores[:,1] >= 1) & (scores[:,1] <= 5)
    idx  = scores[mask,1].astype(int) - 1                      # 0-based bins for 1..5
    probs = np.bincount(idx, weights=scores[mask,0], minlength=5)

    scores = np.c_[probs, np.arange(1,6)]
       
    scores[:, 0] = scores[:, 0] / scores[:, 0].sum() 
    
    return scores



def subject_text(name: str, full_statement: bool=False) -> str:
    subjects = {"wealth_distribution": ["The global distribution of wealth", "is fair"],
                "wealth_distribution_rev": ["The global distribution of wealth", "is not fair"],
                "climate_change": ["Climate Change",  "is an important problem to address"],
                "climate_change_rev": ["Climate Change",  "is not an important problem to address"],
                "ai_dangers": ["Artificial Intelligence",  "is dangerous"],
                "ai_dangers_rev": ["Artificial Intelligence",  "is not dangerous"]}
    
    if full_statement:
        return subjects[name][0] + ' ' + subjects[name][1]
    else:
        return subjects[name][0]
    


def levels_5_scale(level: int) -> str:
    d = {1: 'strongly disagrees', 2: 'disagrees', 3: 'neither agrees nor disagrees', 4: 'agrees', 5: 'strongly agrees'}
    return d[level]



def probe_text(statement:str) -> str:
    p = f'State your opinion about the following statement: "{statement}." '\
        'Provide your answer as one of the options "1: strongly disagree", "2: disagree", "3: neither agree nor disagree", "4: agree", "5: strongly agree". '\
        'Only return the corresponding integer value without any comments or punctuation.'
    return p 



def discussion_prompt(subject_name: str, type: str, r_text: str=None) -> str:
    prompt = f"From now on, you are part of a new discussion about {subject_text(subject_name).lower()}. "
    
    if type == 'init_start':
        prompt += "Write three sentences to start the discussion."
        
    elif type == 'init_response':
        prompt += f'Someone else wrote the following text: "{r_text}". Write three sentences as your response.'
        
    elif type == 'reply': 
        prompt = f'Someone else replied to you with the following text: "{r_text}". Write three sentences as your response.'
    
    elif type == 'reply_probe':
        prompt = f'Someone else replied to you with the following text: "{r_text}". ' + probe_text(subject_text(subject_name, full_statement=True))
    
    elif type == 'reply_probe_init':
        prompt += f'Someone else wrote the following text: "{r_text}". ' + probe_text(subject_text(subject_name, full_statement=True))
    
    elif type == 'continue':
        prompt = 'Someone else replied to you with the following text: "{response_B}". Do you want to continue the discussion? Only answer "Yes" or "No" without any comments or punctuation.'

    return prompt



def tof_prompt(subject_name: str) -> str:
    prompt = f'Write out your thoughts about {subject_text(name=subject_name, full_statement=False).lower()}. Keep your output short.'
    return prompt






class LocalLLM:
    def __init__(self, GPU_idx: str, cache_dir: str, model_id: str='cognitivecomputations/dolphin-2.7-mixtral-8x7b', pass_model: bool=None, pass_tokenizer: bool=None):
        self.model_id = model_id # specify model name (huggingface id)
        self.cache_dir = cache_dir # specify model cache directory
        os.environ['HF_HOME'] = self.cache_dir
        
        #os.environ["CUDA_VISIBLE_DEVICES"] = GPU_idx # select GPU on cluster
        self.device_map = {"": int(GPU_idx)}
        torch.cuda.empty_cache()
        
        self.model = pass_model
        self.tokenizer = pass_tokenizer
        
        if pass_tokenizer:
            self.decode = self.tokenizer.batch_decode
        
        self.messages = [{'role': 'system', 'content': 'You are a helpful AI assistant.'}]
        
        
        
    def load_model(self, only_tokenizer: bool=False, only_local: bool=False):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=self.cache_dir, only_local_files=only_local, token=hf_access_token, device_map=self.device_map)
        self.decode = self.tokenizer.batch_decode
        
        if only_tokenizer:
            return
        else:

            bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True, # Optional but recommended
            bnb_4bit_quant_type="nf4",      # Recommended quant type
            # Set the compute dtype to float16 to match training args
            bnb_4bit_compute_dtype=torch.float16
            )

            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.float32, cache_dir=self.cache_dir, 
                                                              quantization_config=bnb_config, local_files_only=only_local, token=hf_access_token,
                                                              device_map=self.device_map, low_cpu_mem_usage=True)
            return
    
    
    
    def config_generation(self, max_new_tokens: int=256, sample: bool=True, temperature: float=1.0):
        self.generation_config = transformers.GenerationConfig(max_new_tokens=max_new_tokens, do_sample=sample, temperature=temperature,
                                                               output_scores=True, return_dict_in_generate=True, 
                                                               eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.eos_token_id)
        return
        


    def PPL(self, output):    
        perplexity = 0
        N = len(output.scores)
        
        for i, token in enumerate(output.sequences[0][-N:].numpy(force=True)):
            logprob = np.log(torch.nn.functional.softmax(output.scores[i], dim=1)[0][token].item())
            perplexity += logprob
        
        perplexity = np.exp(-perplexity/N)
        return perplexity
    


    def reset_messages(self, content: str='', reset_default: bool=False, role: str="system"):
        if reset_default:
            self.messages = [{"role": "system", "content": "You are Dolphin, a helpful AI assistant."}]
            return
        else:
            self.messages = [{"role": role, "content": content}]
    
    

    def infer(self, content: str=None, role: str='user', generation_prompt: bool=True, print_response: bool=False, full_output: bool=False, top_n_token: int=20, append: bool=True):    
        torch.cuda.empty_cache()

        if self.model == None:
            print('Model not loaded yet!')
            return
        
        if content is not None:
            self.messages.append({"role": role, "content": content})
        
        inputs = self.tokenizer.apply_chat_template(self.messages, tokenize=True, return_tensors='pt', add_generation_prompt=generation_prompt, return_dict=True)    
        inputs_device = inputs.to('cuda')    
        
        output = self.model.generate(**inputs_device, generation_config=self.generation_config)
        response = self.tokenizer.batch_decode(output.sequences[:,inputs_device['input_ids'].shape[1]:], skip_special_tokens=True)[0]    
        self.messages.append({"role": "assistant", "content": response})
        perplexity = self.PPL(output)
        
        if (append == False) and (content is not None):
            self.messages = self.messages[:-2]
            
        elif (append == False) and (content is None):
            self.messages = self.messages[:-1]
        
        if print_response:
            print(response, f'\t ppl: {np.round(perplexity, 3)}')
            
        if full_output == False:
            new_output = np.empty((len(output.scores), 2, top_n_token)) # save only top n token with probabilites
            
            for i, s in enumerate(output.scores):
                scores, tokens = torch.sort(torch.nn.functional.softmax(s, dim=1)[0], descending=True)
                new_output[i][0] = scores[:top_n_token].numpy(force=True)
                new_output[i][1] = tokens[:top_n_token].numpy(force=True)

            return response, new_output, perplexity
        
        else:
            return response, output, perplexity
    


    def generate_tof(self, opinion_level: int, subject_name: str) -> str:
        system_prompt = f'You are a person who {levels_5_scale(opinion_level)} that {subject_text(name=subject_name, full_statement=True).lower()}.'
        
        user_input = tof_prompt(subject_name)

        self.reset_messages(content = f'{system_prompt}')
        self.messages.append({'role': 'user', 'content': f'{user_input}'})
        tof, output, perplexity = self.infer()
        
        return tof, output, perplexity
    
    

    def initiate_agent(self, subject_name: str, tof: str, role: str):
        system_prompt = f"""You are a person who has the following thoughts about {subject_text(name=subject_name, full_statement=False).lower()}: "{tof}" Always act and answer as this person."""
        self.reset_messages(content = f'{system_prompt}', role=role)
        return
    

    
    def expected_opinion(self, output):
        for i in range(5):
            try:
                if int(self.decode(output[:,1][i].astype(int))[0]) in np.arange(1,6):
                    p_norm = norm_probs(output[:,0][i], self.decode(output[:,1][i].astype(int)))
                    p_norm = p_norm[np.argsort(p_norm[:,1])]
                    op = rv_discrete(values=(p_norm[:,1], p_norm[:,0]))
                    H = entropy(pk = p_norm[:,0])
                    
                    return np.array([op.mean(), op.std(), H]), p_norm
            except:
                pass
        
        p_fail = np.zeros((5,2))
        p_fail.fill(-1)

        return np.array([-1,-1,-1]), p_fail
    
    
    
    def probe_internal(self, subject_name: str, append: bool=False):
        p = probe_text(statement=subject_text(name=subject_name, full_statement=True))
        response, output, perplexity = self.infer(content=p, append=append)
        R, probs = self.expected_opinion(output)
        
        return R, probs
        
    

##################################################################################################################################################################
        


class ChatGPT:
    def __init__(self, api_key: str, model_id: str='gpt-4o-mini'):
        self.api_key= api_key  
        self.client = OpenAI(api_key=self.api_key)
        self.model_id = model_id
        self.messages = [{'role': 'system', 'content': 'You are ChatGPT, a helpful AI assistant.'}]
        #self.tokenizer = tiktoken.encoding_for_model(self.model_id)
        self.tokenizer = tiktoken.get_encoding("o200k_base") 
        self.decode = self.tokenizer.decode
    
    
    
    def config_generation(self, sample: bool=True, temperature: float=1.0, top_n_token: int=20, seed: int=42):
        if sample == False:
            temperature = 0
        self.generation_config = {'model': self.model_id, 'messages': self.messages, 'logprobs': True, 
                                 'top_logprobs': top_n_token, 'temperature': temperature, 'seed': seed}
        return
    
        
        
    def PPL(self, completion) -> float:
        perplexity = 0
        N = len(completion.choices[0].logprobs.content)

        for i in range(N):
            perplexity += completion.choices[0].logprobs.content[i].logprob

        return np.exp(-perplexity/N)
        


    def reset_messages(self, content: str='', reset_default: bool=False, role: str="system"):
        if reset_default:
            self.messages = [{"role": "system", "content": "You are ChatGPT, a helpful AI assistant."}]
            return
        else:
            self.messages = [{"role": role, "content": content}]
        

    
    def infer(self, content: str=None, role: str='user', print_response: bool=False, full_output: bool=False, top_n_token: int=20, append: bool=True):    
        if content is not None:
            self.messages.append({"role": role, "content": content})
        
        self.generation_config['messages'] = self.messages
        self.generation_config['top_logprobs'] = top_n_token
        
        completion = self.client.chat.completions.create(**self.generation_config)
        response = completion.choices[0].message.content
        self.messages.append({"role": "assistant", "content": response})
        perplexity = self.PPL(completion)

        if (append == False) and (content is not None):
            self.messages = self.messages[:-2]
            
        elif (append == False) and (content is None):
            self.messages = self.messages[:-1]

        if print_response:
            print(twr.fill(response))
            print('\n', f'Perplexity: {perplexity}')
            
        if full_output == False:
            new_output = np.empty((completion.usage.completion_tokens, 2, top_n_token)) # save only top n token with probabilites
            
            for i, c_list in enumerate(completion.choices[0].logprobs.content):
                for j, c in enumerate(c_list.top_logprobs):
                    new_output[i][0][j] = np.exp(c.logprob)   
                    new_output[i][1][j] = self.tokenizer.encode(c.token)[0]         

            return response, new_output, perplexity
        
        else:
            return response, completion, perplexity
        
    

    def generate_tof(self, opinion_level: int, subject_name: str) -> str:
        system_prompt = f'You are a person who {levels_5_scale(opinion_level)} that {subject_text(name=subject_name, full_statement=True).lower()}.'
        
        user_input = tof_prompt(subject_name)

        self.reset_messages(content = f'{system_prompt}')
        self.messages.append({'role': 'user', 'content': f'{user_input}'})
        tof, output, perplexity = self.infer()
        
        return tof, output, perplexity
    

        
    def initiate_agent(self, subject_name: str, tof: str, role: str):
        system_prompt = f"""You are a person who has the following thoughts about {subject_text(name=subject_name, full_statement=False).lower()}: "{tof}" Always act and answer as this person."""
        self.reset_messages(content = f'{system_prompt}', role=role)
        return
    

    
    def expected_opinion(self, output):
        for i in range(5):
            try:
                if int(self.decode(output[:,1][i].astype(int))[0]) in np.arange(1,6):
                    p_norm = norm_probs(output[:,0][i], self.decode(output[:,1][i].astype(int)))
                    p_norm = p_norm[np.argsort(p_norm[:,1])]
                    op = rv_discrete(values=(p_norm[:,1], p_norm[:,0]))
                    H = entropy(pk = p_norm[:,0])
                    
                    return np.array([op.mean(), op.std(), H]), p_norm
            except:
                pass
        
        p_fail = np.zeros((5,2))
        p_fail.fill(-1)

        return np.array([-1,-1,-1]), p_fail
    
    
    
    def probe_internal(self, subject_name: str, append: bool=False):
        p = probe_text(statement=subject_text(name=subject_name, full_statement=True))
        response, output, perplexity = self.infer(content=p, append=append)
        R, probs = self.expected_opinion(output)
        
        return R, probs
    
    
    
    def probe_external(self, text: str, subject_name: str):
        system_prompt = oc_prompt(subject_statement=subject_text(name=subject_name, full_statement=True))
        self.reset_messages(content = f'{system_prompt}')
        
        self.messages.append({'role': 'user', 'content': f'Evaluate: "{text}"'})
        response, output, perplexity = self.infer()
        R, probs = self.expected_opinion(output)
        
        return R, probs