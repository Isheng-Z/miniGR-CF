import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple
import json
import random
from tqdm import tqdm
import os
import copy
import torch.nn.functional as F

class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        t = self.tokenizer.encode(s)
        if self.bos_id is not None:
            while len(t) > 0 and t[0] == self.bos_id: t = t[1:]
        if self.eos_id is not None:
            while len(t) > 0 and t[-1] == self.eos_id: t = t[:-1]
        if bos and self.bos_id is not None: t = [self.bos_id] + t
        if eos and self.eos_id is not None: t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.tokenizer.decode(t)

# =================================================================
# 1. 核心推荐任务 (SidSFTDataset)
# =================================================================
class SidSFTDataset(Dataset):
    def __init__(self, train_file, tokenizer, max_len=2048, sample=-1, test=False, seed=0, category="", K=4, dedup=False, cf_hints_path="", hint_dropout_rate=0.3):
        self.data = pd.read_csv(train_file)
        random.seed(seed)
        if sample > 0: self.data = self.data.sample(sample, random_state=seed)
        self.tokenizer = Tokenizer(tokenizer)
        self.test = test
        self.max_len = max_len
        self.category = category
        self.dedup = dedup
        self.hint_dropout_rate = hint_dropout_rate
        
        self.cf_hints = {}
        if cf_hints_path and os.path.exists(cf_hints_path):
            print(f"🔥 [SidSFTDataset] Loading Hints from {cf_hints_path}...")
            with open(cf_hints_path, 'r') as f:
                self.cf_hints = json.load(f)
        
        if not self.test and cf_hints_path:
             print(f"🎲 [SidSFTDataset] Dynamic Hint Dropout Enabled: p={self.hint_dropout_rate}")

    def __len__(self): return len(self.data)

    def generate_prompt(self, data_point):
        return f"### User Input: \n{data_point['input']}\n\n### Response:\n{data_point['output']}"

    def get_history(self, idx):
        row = self.data.iloc[idx]
        try: history_sids = eval(row['history_item_sid'])
        except: history_sids = row['history_item_sid'] if isinstance(row['history_item_sid'], list) else []
        history_str = ", ".join(history_sids)
        
        target_item_sid = str(row['item_sid'])
        hint_text = ""
        
        if 'history_item_id' in row and self.cf_hints:
            try:
                hist_ids = eval(str(row['history_item_id']))
                if hist_ids:
                    last_id = str(hist_ids[-1])
                    if last_id in self.cf_hints:
                        neighbors_str = self.cf_hints[last_id]
                        # 保留 Target，不剔除
                        hint_text = f" [Hint: Users who bought the last item often also buy: {neighbors_str}.]"
            except: pass

        # 动态 Dropout
        if not self.test and hint_text and random.random() < self.hint_dropout_rate:
            hint_text = ""

        target_item = str(row['item_sid'])
        
        input_text = f"The user has interacted with items {history_str} in chronological order.{hint_text} Can you predict the next possible item that the user may expect?"
        
        return {
            "input": input_text,
            "output": target_item + "\n"
        }
    
    def pre(self, idx):
        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you predict the next possible item that the user may expect?

"""
        tokens = self.tokenizer.encode(instruction, bos=True, eos=False)
        history = self.get_history(idx)
        
        if self.test:
            prompt = f"### User Input: \n{history['input']}\n\n### Response:\n"
            tokens = tokens + self.tokenizer.encode(prompt, bos=False, eos=False)
            attention_mask = [1] * len(tokens)
            return {"input_ids": tokens, "attention_mask": attention_mask}    
        else:
            prompt_input = f"### User Input: \n{history['input']}\n\n### Response:\n"
            tokens += self.tokenizer.encode(prompt_input, bos=False, eos=False)
            target_tokens = self.tokenizer.encode(history['output'], bos=False, eos=True)
            
            input_len = len(tokens)
            tokens += target_tokens
            
            attention_mask = [1] * len(tokens)
            labels = [-100] * input_len + tokens[input_len:]
            
            if len(tokens) > self.max_len:
                tokens = tokens[-self.max_len:]
                attention_mask = attention_mask[-self.max_len:]
                labels = labels[-self.max_len:]
                
            return {"input_ids": tokens, "attention_mask": attention_mask, "labels": labels}
    
    def __getitem__(self, idx):
        return self.pre(idx)
    
    def get_all(self):
        return [self.get_history(i) for i in range(len(self.data))]

# =================================================================
# 2. EvalSidDataset
# =================================================================
class EvalSidDataset(SidSFTDataset):
    def __init__(self, train_file, tokenizer, max_len=2048, sample=-1, test = False, seed=0, category="", K=4, dedup=False, cf_hints_path=""):
        super().__init__(train_file, tokenizer, max_len, sample, True, seed, category, K, dedup, cf_hints_path, hint_dropout_rate=0.0)

# =================================================================
# 3. SidItemFeatDataset (辅助任务)
# =================================================================
class SidItemFeatDataset(Dataset):
    def __init__(self, item_file, index_file, tokenizer=None, max_len=2048, sample=-1, test=False, seed=0, category=""):
        random.seed(seed)
        with open(item_file, 'r') as f: self.item_feat = json.load(f)
        with open(index_file, 'r') as f: self.indices = json.load(f)
        self.tokenizer = Tokenizer(tokenizer) if tokenizer is not None else None
        self.test = test
        self.max_len = max_len
        self.sid2title, self.title2sid = {}, {}
        for item_id, sids in self.indices.items():
            if item_id in self.item_feat and len(sids)>=3:
                combined = "".join(sids[:3])
                t = self.item_feat[item_id]['title']
                self.sid2title[combined] = t; self.title2sid[t] = combined
        self.data = []
        for s, t in self.sid2title.items(): self.data.append({'t':'sid2title', 'i':s, 'o':t})
        for t, s in self.title2sid.items(): self.data.append({'t':'title2sid', 'i':t, 'o':s})
        if sample > 0: self.data = random.sample(self.data, sample)
    
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        d = self.data[idx]
        pmt = f"Which item has title: {d['i']}?" if d['t']=='title2sid' else f"Title of item \"{d['i']}\"?"
        full = f"### User Input: \n{pmt}\n\n### Response:\n{d['o']}\n"
        ins = "Answer the question about item identification.\n\n"
        tokens = self.tokenizer.encode(ins, True, False) + self.tokenizer.encode(full, False, True)
        
        labels = tokens 
        if len(tokens)>self.max_len: tokens=tokens[-self.max_len:]; labels=labels[-self.max_len:]
        return {"input_ids": tokens, "attention_mask": [1]*len(tokens), "labels": labels}

# =================================================================
# 4. FusionSeqRecDataset (辅助任务)
# =================================================================
class FusionSeqRecDataset(Dataset):
    def __init__(self, train_file, item_file, index_file, tokenizer, max_len=2048, sample=-1, test=False, seed=0, category="", dedup=False):
        self.data = pd.read_csv(train_file)
        random.seed(seed)
        if sample > 0: self.data = self.data.sample(sample, random_state=seed)
        with open(item_file, 'r') as f: self.item_feat = json.load(f)
        with open(index_file, 'r') as f: self.indices = json.load(f)
        self.tokenizer = Tokenizer(tokenizer)
        self.max_len = max_len
        self.sid2title = {}
        for item_id, sids in self.indices.items():
            if item_id in self.item_feat and len(sids)>=3: self.sid2title["".join(sids[:3])] = self.item_feat[item_id]['title']
    
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        tokens = self.tokenizer.encode("Below is an instruction... Can you recommend the next item for the user based on their interaction history?\n\n", True, False)
        row = self.data.iloc[idx]
        try: h = eval(row['history_item_sid'])
        except: h = []
        tgt = self.sid2title.get(str(row['item_sid']), str(row['item_sid']))
        prompt = f"### User Input: \nThe user has sequentially interacted with items {', '.join(h)}. Can you recommend the next item for him? Tell me the title of the item\n\n### Response:\n{tgt}\n"
        tokens += self.tokenizer.encode(prompt, False, True)
        
        labels = tokens 
        if len(tokens)>self.max_len: tokens=tokens[-self.max_len:]; labels=labels[-self.max_len:]
        return {"input_ids": tokens, "attention_mask": [1]*len(tokens), "labels": labels}

# =================================================================
# 5. SFTData (辅助任务) - 【已修复变量名 Bug】
# =================================================================
class SFTData(Dataset):
    def __init__(self, train_file, tokenizer, max_len=2048, sample=-1, test=False, seed=0, category="", K=4, dedup=False):
        self.data = pd.read_csv(train_file)
        random.seed(seed)
        if sample > 0: self.data = self.data.sample(sample, random_state=seed)
        self.tokenizer = Tokenizer(tokenizer)
        self.max_len = max_len
        self.category = category
        self.instructs = [f"Given a list of {category}..." ] 
    
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        try: h = eval(row['history_item_title'])
        except: h = []
        prompt = f"### User Input: \nThe user has palyed the following {self.category}s before: {', '.join(['\"'+t+'\"' for t in h])}\n\n### Response:\n\"{row['item_title']}\"\n"
        ins = f"Below is an instruction... \n\n### Instruction:\nGiven a list of {self.category}...\n"
        tokens = self.tokenizer.encode(ins, True, False) + self.tokenizer.encode(prompt, False, False)
        
        labels = tokens # 统一使用 labels
        
        if len(tokens)>self.max_len: tokens=tokens[-self.max_len:]; labels=labels[-self.max_len:]
        return {"input_ids": tokens, "attention_mask": [1]*len(tokens), "labels": labels}

# =================================================================
# 6. TitleHistory2SidSFTDataset (辅助任务) - 【已修复变量名 Bug】
# =================================================================
class TitleHistory2SidSFTDataset(Dataset):
    def __init__(self, train_file, item_file, index_file, tokenizer, max_len=2048, sample=-1, test=False, seed=0, category="", dedup=False):
        self.data = pd.read_csv(train_file)
        if sample > 0: self.data = self.data.sample(sample, random_state=seed)
        with open(index_file, 'r') as f: self.indices = json.load(f)
        self.tokenizer = Tokenizer(tokenizer)
        self.max_len = max_len
        self.id2sid = {}
        for item_id, sids in self.indices.items():
            if len(sids) >= 3: self.id2sid[item_id] = "".join(sids[:3])
    
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        try: h = eval(row['history_item_title'])
        except: h = []
        tgt = self.id2sid.get(str(row['item_id']), str(row['item_id'])) + "\n"
        prompt = f"### User Input: \nThe user has interacted... {', '.join(['\"'+t+'\"' for t in h])}... predict the semantic ID...\n\n### Response:\n{tgt}"
        ins = "Below is an instruction... Based on the user's historical interaction with item titles...\n\n"
        tokens = self.tokenizer.encode(ins, True, False) + self.tokenizer.encode(prompt, False, False)
        
        labels = tokens # 统一使用 labels
        
        if len(tokens)>self.max_len: tokens=tokens[-self.max_len:]; labels=labels[-self.max_len:]
        return {"input_ids": tokens, "attention_mask": [1]*len(tokens), "labels": labels}