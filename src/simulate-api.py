import argparse
import os
import asyncio
import time
import gc

from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
from scipy.stats import entropy

from openai import AsyncOpenAI
from openai import RateLimitError, APITimeoutError, APIConnectionError, APIError
import random

# ----------------------------
# CLI arguments
# ----------------------------
parser = argparse.ArgumentParser(description="Opinion dynamics via OpenAI/xAI APIs")

parser.add_argument("-b", "--backend", type=str, choices=["openai", "xai"], default="openai",
                    help="Backend to use: 'openai' or 'xai' (Grok via x.ai API)")
parser.add_argument("-m", "--model", type=str, help="API model name", required=True)
parser.add_argument("-t", "--topic", type=str, help="Discussion Topic", required=True)
parser.add_argument("-r", "--reverse", action="store_true", help="Enable reverse (negated) topic framing")
parser.add_argument("-s", "--samples", type=int, help="Number of samples to run", required=True)
parser.add_argument("-l", "--length", type=int, help="Discussion length", required=True)
parser.add_argument("-o", "--output", type=str, help="Output CSV file path")


args = parser.parse_args()
backend = args.backend
model_name = args.model
topic = args.topic
rev_flag = args.reverse
samples = args.samples
discussion_len = args.length
output_path = args.output

if output_path is None:
    output_path = (
        f"../data/{model_name.split('/')[-1]}_{topic.replace(' ', '_')}"
        f"_s={samples}_l={discussion_len}"
        + ("_reversed_framing" if rev_flag else "_normal_framing")
        + ".csv"
    )

# ----------------------------
# Retry logic for rate limits
# ----------------------------
async def retry_with_backoff(fn, max_retries=8, base_delay=1.0, max_delay=45.0):
    for attempt in range(max_retries):
        try:
            return await fn()
        except (RateLimitError, APITimeoutError, APIConnectionError, APIError) as e:
            delay = min(max_delay, base_delay * 2 ** attempt)
            print(f"{type(e).__name__} encountered. Retrying in {delay:.1f}s...")
            await asyncio.sleep(delay)
        except Exception:
            raise
    raise RuntimeError("Exceeded maximum retry attempts.")

# ----------------------------
# Intermediate save helper
# ----------------------------
def safe_save(df, path):
    tmp_path = path + ".tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)  # atomic overwrite

# ----------------------------
# Async LLM wrapper
# ----------------------------
@dataclass
class MessageOutput:
    text: str
    logprobs: object  # backend-specific logprobs object (or None)


class AsyncRemoteLLM:
    def __init__(self, backend: str, model: str):
        self.backend = backend
        self.model = model

        if backend == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set in environment.")
            self.client = AsyncOpenAI(api_key=api_key, timeout=60)
        elif backend == "xai":
            api_key = os.getenv("XAI_API_KEY")
            if not api_key:
                raise RuntimeError("XAI_API_KEY not set in environment.")
            # xAI is OpenAI-compatible, base_url per docs: https://api.x.ai
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.x.ai/v1",
                timeout=60
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    async def _chat_single(
        self,
        messages: list,
        max_tokens: int,
        temperature: float,
        need_logprobs: bool,
        top_logprobs: int | None,
    ) -> MessageOutput:
        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if need_logprobs:
            params["logprobs"] = True
            if top_logprobs is not None:
                params["top_logprobs"] = top_logprobs

        #resp = await self.client.chat.completions.create(**params)
        resp = await retry_with_backoff(lambda: self.client.chat.completions.create(**params))

        choice = resp.choices[0]
        text = choice.message.content
        logprobs = choice.logprobs  # may be None when logprobs not requested
        return MessageOutput(text=text, logprobs=logprobs)

    async def chat_batch(
        self,
        messages_list: list[list[dict]],
        max_tokens: int,
        temperature: float = 1.0,
        need_logprobs: bool = False,
        top_logprobs: int | None = None,
    ) -> list[MessageOutput]:
        sem = asyncio.Semaphore(6)

        async def worker(msgs):
            async with sem:
                return await self._chat_single(
                    msgs, max_tokens, temperature, need_logprobs, top_logprobs
                )

        tasks = [worker(msgs) for msgs in messages_list]
        return await asyncio.gather(*tasks)


# ----------------------------
# Utility functions for prompts
# ----------------------------
def prompt_template(user_prompt: str, role: str = "system",
                    content: str = "You are a helpful AI assistant.") -> list:
    return [
        {"role": role, "content": content},
        {"role": "user", "content": user_prompt},
    ]


def topic_statements(topic: str, reverse: bool = False) -> str:
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
        "Art Style Preference rev": "Modern art is not more meaningful than classical art",
    }

    if reverse:
        return statements[f"{topic} rev"]
    else:
        return statements[topic]


def opinion_scale(opinion: int) -> str:
    d = {1: 'strongly disagrees', 2: 'disagrees', 3: 'neither agrees nor disagrees',
         4: 'agrees', 5: 'strongly agrees'}
    return d[opinion]


def cot_prompt(topic: str, opinion: int, reverse: bool) -> list:
    reverse_mapping = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1}
    if reverse:
        opinion = reverse_mapping[opinion]
    system_prompt = (
        f'You are are a person who {opinion_scale(opinion)} that {topic_statements(topic)}.'
    )
    user_prompt = f'Write out your thoughts about {topic}. Keep your output short.'
    return prompt_template(user_prompt, content=system_prompt)


def opinion_probing_prompt(statement: str) -> str:
    p = (
        f'State your opinion about the following statement: "{statement}." '
        'Provide your answer as one of the options "1: strongly disagree", '
        '"2: disagree", "3: neither agree nor disagree", "4: agree", '
        '"5: strongly agree". '
        'Only return the corresponding integer value without any comments or punctuation.'
    )
    return p


def initiate_agent_prompt(topic: str, cot: str) -> str:
    return (
        f'You are a person who has the following thoughts about {topic}: "{cot}" '
        f'Always act and answer as this person.'
    )


def discussion_prompt(topic: str, type: str, r_text: str = None,
                      rev_probing: bool = False) -> str:
    prompt = f"From now on, you are part of a new discussion about {topic}. "

    if type == 'init_start':
        prompt += "Write three sentences to start the discussion."

    elif type == 'init_response':
        prompt += f'Someone else wrote the following text: "{r_text}". Write three sentences as your response.'

    elif type == 'reply':
        prompt = f'Someone else replied to you with the following text: "{r_text}". Write three sentences as your response.'

    elif type == 'reply_probe':
        prompt = (
            f'Someone else replied to you with the following text: "{r_text}". '
            + opinion_probing_prompt(topic_statements(topic, reverse=rev_probing))
        )

    elif type == 'reply_probe_init':
        prompt += (
            f'Someone else wrote the following text: "{r_text}". '
            + opinion_probing_prompt(topic_statements(topic, reverse=rev_probing))
        )

    return prompt


def init_dataframe():
    df = pd.DataFrame({
        "discussion_id": pd.Series(dtype='int'), "t": pd.Series(dtype='int'),
        "init_x_i": pd.Series(dtype='int'), "x_i": pd.Series(dtype='float'),
        "H_i": pd.Series(dtype='float'), "text_i": pd.Series(dtype='str'),
        "init_x_j": pd.Series(dtype='int'), "x_j": pd.Series(dtype='float'),
        "H_j": pd.Series(dtype='float'), "text_j": pd.Series(dtype='str'),
    })
    return df


# ----------------------------
# Logprob-based opinion extraction
# ----------------------------
def _safe_get(obj, name, default=None):
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict) and name in obj:
        return obj[name]
    return default


def expected_opinion(logprobs_obj) -> np.ndarray:
    """
    Compute expected opinion and entropy from OpenAI/xAI logprobs structure.

    logprobs_obj is typically response.choices[0].logprobs from the Chat Completions API.
    We use the first generated token's top_logprobs (and the token itself) and
    look for tokens "1".."5".
    """
    targets = ["1", "2", "3", "4", "5"]

    if logprobs_obj is None:
        return np.array([np.nan, np.nan])

    content = _safe_get(logprobs_obj, "content")
    if not content or len(content) == 0:
        return np.array([np.nan, np.nan])

    first = content[0]
    # main token + top_logprobs alternatives
    candidates = [first]
    top_list = _safe_get(first, "top_logprobs", [])
    if top_list:
        candidates.extend(list(top_list))

    prob_map = {}
    for c in candidates:
        tok = _safe_get(c, "token")
        lp = _safe_get(c, "logprob")
        if tok is None or lp is None:
            continue
        tok_str = str(tok).strip()
        if tok_str in targets and tok_str not in prob_map:
            prob_map[tok_str] = float(np.exp(lp))

    if not prob_map:
        return np.array([np.nan, np.nan])

    probs = np.array([prob_map.get(str(i), 0.0) for i in range(1, 6)], dtype=float)
    s = probs.sum()
    if s <= 0:
        return np.array([np.nan, np.nan])

    probs /= s
    values = np.arange(1, 6, dtype=float)
    mean = float(np.sum(values * probs))
    H = float(entropy(probs))

    return np.array([mean, H])


# ----------------------------
# Main async pipeline
# ----------------------------
async def main():
    tic = time.time()

    df = init_dataframe()

    # Predefine dataframe for data storage: one row per (disc_id, t)
    disc_id = 0
    for _ in range(samples):
        for init_x_i in range(-2, 3):
            for init_x_j in range(-2, 3):
                # t = 0 row
                df.loc[len(df)] = [
                    disc_id, 0, init_x_i, np.nan, np.nan, pd.NA,
                    init_x_j, np.nan, np.nan, pd.NA
                ]
                # t = 1..discussion_len rows
                for t in range(1, discussion_len + 1):
                    df.loc[len(df)] = [
                        disc_id, t, init_x_i, np.nan, np.nan, pd.NA,
                        init_x_j, np.nan, np.nan, pd.NA
                    ]

                disc_id += 1

    # Initialize remote LLM
    llm = AsyncRemoteLLM(backend=backend, model=model_name)

    # Generation parameters
    TEXT_TEMPERATURE = 1.0
    TEXT_MAX_TOKENS = 1024

    PROBE_TEMPERATURE = 1.0
    PROBE_MAX_TOKENS = 5
    if backend == "xai":
        PROBE_TOP_LOGPROBS = 8
    else:
        PROBE_TOP_LOGPROBS = 10

    # ------------------------
    # Generate CoT monologues
    # ------------------------
    conversations = []
    for _ in range(samples):
        for x_i in range(1, 6):
            for x_j in range(1, 6):
                conversations.append(cot_prompt(topic, x_i, reverse=rev_flag))
                conversations.append(cot_prompt(topic, x_j, reverse=rev_flag))

    outputs = await llm.chat_batch(
        messages_list=conversations,
        max_tokens=TEXT_MAX_TOKENS,
        temperature=TEXT_TEMPERATURE,
        need_logprobs=False
    )

    monologues = [o.text for o in outputs]

    # Save initial monologues (t=0 texts)
    for i in range(len(monologues) // 2):
        cot_i = monologues[2 * i]
        cot_j = monologues[2 * i + 1]

        disc_id = i
        df.loc[(df['discussion_id'] == disc_id) & (df['t'] == 0), 'text_i'] = cot_i
        df.loc[(df['discussion_id'] == disc_id) & (df['t'] == 0), 'text_j'] = cot_j

    # ------------------------
    # Probe initial opinions (t=0)
    # ------------------------
    prompts = []
    for m in monologues:
        prompts.append(
            prompt_template(
                user_prompt=opinion_probing_prompt(
                    topic_statements(topic, reverse=rev_flag)
                ),
                content=initiate_agent_prompt(topic, m)
            )
        )

    probings = await llm.chat_batch(
        messages_list=prompts,
        max_tokens=PROBE_MAX_TOKENS,
        temperature=PROBE_TEMPERATURE,
        need_logprobs=True,
        top_logprobs=PROBE_TOP_LOGPROBS
    )

    expected_opinions = np.zeros((len(probings), 2))
    for i, output in enumerate(probings):
        expected_opinions[i] = expected_opinion(output.logprobs)

    # Save initial expected opinions (shift by -3 to map 1..5 -> -2..2)
    for i in range(len(expected_opinions) // 2):
        x_i, H_i = expected_opinions[2 * i]
        x_j, H_j = expected_opinions[2 * i + 1]

        disc_id = i
        df.loc[(df['discussion_id'] == disc_id) & (df['t'] == 0), 'x_i'] = x_i - 3
        df.loc[(df['discussion_id'] == disc_id) & (df['t'] == 0), 'H_i'] = H_i
        df.loc[(df['discussion_id'] == disc_id) & (df['t'] == 0), 'x_j'] = x_j - 3
        df.loc[(df['discussion_id'] == disc_id) & (df['t'] == 0), 'H_j'] = H_j

    # ------------------------
    # Discussion round t = 1
    # ------------------------
    # Prompts for agent i: start discussion
    prompts_i = []
    for m in monologues[::2]:
        prompts_i.append(
            prompt_template(
                user_prompt=discussion_prompt(topic, type="init_start"),
                content=initiate_agent_prompt(topic, m)
            )
        )

    outputs_i = await llm.chat_batch(
        messages_list=prompts_i,
        max_tokens=TEXT_MAX_TOKENS,
        temperature=TEXT_TEMPERATURE,
        need_logprobs=False
    )

    # Prompts for agent j responses to agent i
    prompts_j = []
    for i_disc in range(len(outputs_i)):
        disc_id = i_disc
        msg_i = outputs_i[i_disc].text
        prompts_i[i_disc].append({"role": "assistant", "content": msg_i})
        df.loc[(df['discussion_id'] == disc_id) & (df['t'] == 1), 'text_i'] = msg_i

        prompts_j.append(
            prompt_template(
                user_prompt=discussion_prompt(topic, type="init_response", r_text=msg_i),
                content=initiate_agent_prompt(topic, monologues[2 * i_disc + 1])
            )
        )

    outputs_j = await llm.chat_batch(
        messages_list=prompts_j,
        max_tokens=TEXT_MAX_TOKENS,
        temperature=TEXT_TEMPERATURE,
        need_logprobs=False
    )

    # Save j's replies and add probing prompts for t=1
    for i_disc in range(len(outputs_j)):
        disc_id = i_disc
        msg_j = outputs_j[i_disc].text
        prompts_j[i_disc].append({"role": "assistant", "content": msg_j})
        df.loc[(df['discussion_id'] == disc_id) & (df['t'] == 1), 'text_j'] = msg_j

        prompts_i[i_disc].append({
            "role": "user",
            "content": discussion_prompt(
                topic, type="reply_probe", r_text=msg_j, rev_probing=rev_flag
            ),
        })
        prompts_j[i_disc].append({
            "role": "user",
            "content": opinion_probing_prompt(
                topic_statements(topic, reverse=rev_flag)
            ),
        })

    # Probe agent i at t=1
    probings_i = await llm.chat_batch(
        messages_list=prompts_i,
        max_tokens=PROBE_MAX_TOKENS,
        temperature=PROBE_TEMPERATURE,
        need_logprobs=True,
        top_logprobs=PROBE_TOP_LOGPROBS
    )
    for i_disc, output in enumerate(probings_i):
        x_i, H_i = expected_opinion(output.logprobs)
        disc_id = i_disc
        df.loc[(df['discussion_id'] == disc_id) & (df['t'] == 1), 'x_i'] = x_i - 3
        df.loc[(df['discussion_id'] == disc_id) & (df['t'] == 1), 'H_i'] = H_i

    # Probe agent j at t=1
    probings_j = await llm.chat_batch(
        messages_list=prompts_j,
        max_tokens=PROBE_MAX_TOKENS,
        temperature=PROBE_TEMPERATURE,
        need_logprobs=True,
        top_logprobs=PROBE_TOP_LOGPROBS
    )
    for i_disc, output in enumerate(probings_j):
        x_j, H_j = expected_opinion(output.logprobs)
        disc_id = i_disc
        df.loc[(df['discussion_id'] == disc_id) & (df['t'] == 1), 'x_j'] = x_j - 3
        df.loc[(df['discussion_id'] == disc_id) & (df['t'] == 1), 'H_j'] = H_j

    # Remove last probing prompt from prompts before next discussion round
    for p in range(len(prompts_i)):
        prompts_i[p] = prompts_i[p][:-1]
        prompts_j[p] = prompts_j[p][:-1]

    safe_save(df, output_path)
    # ------------------------
    # Remaining discussion rounds t = 2..discussion_len
    # ------------------------
    for t in range(2, discussion_len + 1):
        # Agent i replies to agent j's last message
        # Use row order aligned by disc_id for t-1
        prev_j_msgs = df[df["t"] == t - 1].sort_values("discussion_id")["text_j"].tolist()

        for n in range(len(prompts_i)):
            prompts_i[n].append({
                "role": "user",
                "content": discussion_prompt(
                    topic, type="reply", r_text=prev_j_msgs[n]
                ),
            })

        outputs_i = await llm.chat_batch(
            messages_list=prompts_i,
            max_tokens=TEXT_MAX_TOKENS,
            temperature=TEXT_TEMPERATURE,
            need_logprobs=False
        )

        for n, out in enumerate(outputs_i):
            msg_i = out.text
            prompts_i[n].append({"role": "assistant", "content": msg_i})
            df.loc[(df['discussion_id'] == n) & (df['t'] == t), 'text_i'] = msg_i
            prompts_j[n].append({
                "role": "user",
                "content": discussion_prompt(
                    topic, type="reply", r_text=msg_i
                ),
            })

        outputs_j = await llm.chat_batch(
            messages_list=prompts_j,
            max_tokens=TEXT_MAX_TOKENS,
            temperature=TEXT_TEMPERATURE,
            need_logprobs=False
        )

        for n, out in enumerate(outputs_j):
            msg_j = out.text
            prompts_j[n].append({"role": "assistant", "content": msg_j})
            df.loc[(df['discussion_id'] == n) & (df['t'] == t), 'text_j'] = msg_j

            prompts_i[n].append({
                "role": "user",
                "content": discussion_prompt(
                    topic, type="reply_probe", r_text=msg_j, rev_probing=rev_flag
                ),
            })
            prompts_j[n].append({
                "role": "user",
                "content": opinion_probing_prompt(
                    topic_statements(topic, reverse=rev_flag)
                ),
            })

        # Probe agent i
        probings_i = await llm.chat_batch(
            messages_list=prompts_i,
            max_tokens=PROBE_MAX_TOKENS,
            temperature=PROBE_TEMPERATURE,
            need_logprobs=True,
            top_logprobs=PROBE_TOP_LOGPROBS
        )
        for i_disc, output in enumerate(probings_i):
            x_i, H_i = expected_opinion(output.logprobs)
            disc_id = i_disc
            df.loc[(df['discussion_id'] == disc_id) & (df['t'] == t), 'x_i'] = x_i - 3
            df.loc[(df['discussion_id'] == disc_id) & (df['t'] == t), 'H_i'] = H_i

        # Probe agent j
        probings_j = await llm.chat_batch(
            messages_list=prompts_j,
            max_tokens=PROBE_MAX_TOKENS,
            temperature=PROBE_TEMPERATURE,
            need_logprobs=True,
            top_logprobs=PROBE_TOP_LOGPROBS
        )
        for i_disc, output in enumerate(probings_j):
            x_j, H_j = expected_opinion(output.logprobs)
            disc_id = i_disc
            df.loc[(df['discussion_id'] == disc_id) & (df['t'] == t), 'x_j'] = x_j - 3
            df.loc[(df['discussion_id'] == disc_id) & (df['t'] == t), 'H_j'] = H_j

        # Remove probing prompts before next round
        for p in range(len(prompts_i)):
            prompts_i[p] = prompts_i[p][:-1]
            prompts_j[p] = prompts_j[p][:-1]

        safe_save(df, output_path)

    # ------------------------
    # Save results and cleanup
    # ------------------------
    df.to_csv(output_path, index=False)

    toc = time.time()
    print("Successfully finished. Total execution time:", f"{(toc - tic)/60} minutes", flush=True)
    print(f"Results saved to {output_path}", flush=True)
    print("\n\n--------------------------------------------------\n\n", flush=True)

    # Soft cleanup
    gc.collect()

if __name__ == "__main__":
    asyncio.run(main())