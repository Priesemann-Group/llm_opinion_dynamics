import numpy as np
import llm_setup as llm
import textwrap as tw
import pandas as pd
from collections import defaultdict
from itertools import product, chain


def init_dataframe():
    df = pd.DataFrame({
        "ID": pd.Series(dtype='int'), "TEXT_TYPE": pd.Series(dtype='str'), 
        "INIT_OP_A": pd.Series(dtype='int'), "TEXT_A": pd.Series(dtype='str'), "PPL_A": pd.Series(dtype='float'), 
        "PROBE_INT_A": pd.Series(dtype='float'), "H_INT_A": pd.Series(dtype='float'),
        "INIT_OP_B": pd.Series(dtype='int'), "TEXT_B": pd.Series(dtype='str'), "PPL_B": pd.Series(dtype='float'),
        "PROBE_INT_B": pd.Series(dtype='float'), "H_INT_B": pd.Series(dtype='float'), 
    })
    return df

model = 'dolphin_mixtral' # change model here
data_path = f'../data/simulations/raw/{model}/'
GPU_idx='0'

### Choose LLM (you can also use any other inference backend or API here)
agent = llm.LocalLLM(GPU_idx=GPU_idx)
#agent = llm.ChatGPT(model_id='gpt-4o-mini')

agent.load_model()
agent.config_generation()


### Initialize LLMs
alice = llm.LocalLLM(GPU_idx=GPU_idx, pass_model=agent.model, pass_tokenizer=agent.tokenizer)
alice.generation_config = agent.generation_config

bob = llm.LocalLLM(GPU_idx=GPU_idx, pass_model=agent.model, pass_tokenizer=agent.tokenizer)
bob.generation_config = agent.generation_config

tof_gen = llm.LocalLLM(GPU_idx=GPU_idx, pass_model=agent.model, pass_tokenizer=agent.tokenizer)
tof_gen.generation_config = agent.generation_config


### Set the topic
topic = 'climate_change'
#topic = 'climate_change_rev'

#topic = 'ai_dangers'
#topic = 'ai_dangers_rev'

#topic = 'wealth_distribution'
#topic = 'wealth_distribution_rev'


### Number of runs
N = 25

### Number of discussion rounds (1 round = 1 Alice response + 1 Bob response)
D = 5

### Initialize data arrays shaped (#pair_distances, #runs, #discussion_rounds + 1 init round, #agents, int/ext probing type (only int used here), #features)
all_R = np.zeros((25, N, D+1, 2, 2, 3))
all_probs = np.zeros((25, N, D+1, 2, 2, 5, 2))

### Initialize dataframe
df = init_dataframe()

### Initialize counter and disc_id
count = 0
disc_id = 0

### Loop over all pairs

pc = 0 # pair_counter to 25
for a, op_A in enumerate(np.arange(1,6)):

    for b, op_B in enumerate(np.arange(1,6)):

        ### Loop over all runs
        for n in range(N):
            ### Save parameters
            df.loc[count, 'ID'] = disc_id
            df.loc[count, 'TEXT_TYPE'] = 'tot'
            df.loc[count, 'INIT_OP_A'] = op_A
            df.loc[count, 'INIT_OP_B'] = op_B

            ### Generate train of thoughts 
            tot_A, out_tot_A, df.loc[count, 'PPL_A'] = tof_gen.generate_tof(opinion_level=op_A, subject_name=topic)
            df.loc[count, 'TEXT_A'] = tot_A
            tot_B, out_tot_B, df.loc[count, 'PPL_B'] = tof_gen.generate_tof(opinion_level=op_B, subject_name=topic)
            df.loc[count, 'TEXT_B'] = tot_B

            ### Initialize the LLM agents with the train of thoughts in the right format
            alice.initiate_agent(subject_name=topic, tof=tot_A, role='system')
            bob.initiate_agent(subject_name=topic, tof=tot_B, role='system')

            ### probe opinion after initialization
            all_R[pc, n, 0, 0, 0], all_probs[pc, n, 0, 0, 0] = alice.probe_internal(topic)
            all_R[pc, n, 0, 1, 0], all_probs[pc, n, 0, 1, 0] = bob.probe_internal(topic)

            df.loc[count, 'PROBE_INT_A'] = all_R[pc, n, 0, 0, 0][0]
            df.loc[count, 'H_INT_A'] = all_R[pc, n, 0, 0, 0][2]
            df.loc[count, 'PROBE_INT_B'] = all_R[pc, n, 0, 1, 0][0]
            df.loc[count, 'H_INT_B'] = all_R[pc, n, 0, 1, 0][2]

            #### set dca_type to get the right prompts at the start
            dca_a = 'init_start'
            dca_b = 'init_response'
            response_B = ""

            count += 1

            ### Loop over all discussion rounds
            for d in range(D):

                ### Save parameters
                df.loc[count, 'ID'] = disc_id
                df.loc[count, 'TEXT_TYPE'] = 'disc'
                df.loc[count, 'INIT_OP_A'] = op_A
                df.loc[count, 'INIT_OP_B'] = op_B

                response_A, output_A, df.loc[count, 'PPL_A'] = alice.infer(content = llm.discussion_prompt(topic, dca_a, response_B))
                df.loc[count, 'TEXT_A'] = response_A

                response_B, output_B, df.loc[count, 'PPL_B'] = bob.infer(content = llm.discussion_prompt(topic, dca_b, response_A))
                df.loc[count, 'TEXT_B'] = response_B

                # probe opinion alice
                prompt = llm.discussion_prompt(topic, 'reply_probe', response_B)
                res, out, ppl = alice.infer(prompt, append=False)
                all_R[pc, n, d+1, 0, 0], all_probs[pc, n, d+1, 0, 0] = alice.expected_opinion(out)

                df.loc[count, 'PROBE_INT_A'] = all_R[pc, n, d+1, 0, 0][0]
                df.loc[count, 'H_INT_A'] = all_R[pc, n, d+1, 0, 0][2]

                # probe opinion bob
                all_R[pc, n, d+1, 1, 0], all_probs[pc, n, d+1, 1, 0] = bob.probe_internal(topic)
                df.loc[count, 'PROBE_INT_B'] = all_R[pc, n, d+1, 1, 0][0]
                df.loc[count, 'H_INT_B'] = all_R[pc, n, d+1, 1, 0][2]

                ### change dca_type for the rest of the discussion
                dca_a = 'reply'
                dca_b = 'reply'

                count += 1

                np.save(data_path + f'{topic}/' + 'all_R', all_R)
                np.save(data_path + f'{topic}/' + 'all_probs', all_probs)
                df.to_csv(data_path + f'{topic}/' + 'messages.csv', index=False)

                print(f'Pair {pc+1}/{25}, Run {n+1}/{N}, Discussion {d+1}/{D}', end='\r')
            
            disc_id += 1

        pc += 1

print('########### SUCCESSFULLY FINISHED ###########')
