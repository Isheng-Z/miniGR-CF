import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List
import json
import random
import os


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
# 1. 核心 RL Dataset (SidRLDataset)
#    区别于 SFT，这里返回 raw prompt 和 target 用于 generation & reward
# =================================================================
class SidRLDataset(Dataset):
    def __init__(self, train_file, user2id_map, tokenizer, max_len=2048, sample=-1, seed=0, cf_hints_path="",
                 hint_dropout_rate=0.0):
        self.data = pd.read_csv(train_file)
        self.user2id = user2id_map
        random.seed(seed)
        if sample > 0: self.data = self.data.sample(sample, random_state=seed)

        self.tokenizer = Tokenizer(tokenizer)
        self.max_len = max_len
        self.hint_dropout_rate = hint_dropout_rate

        self.cf_hints = {}
        if cf_hints_path and os.path.exists(cf_hints_path):
            print(f"🔥 [SidRLDataset] Loading Hints from {cf_hints_path}...")
            with open(cf_hints_path, 'r') as f:
                self.cf_hints = json.load(f)

    def __len__(self):
        return len(self.data)

    def get_user_idx(self, row):
        # 尝试获取 User ID 以便 VeRA 查表
        # 假设 CSV 有 user_id 列 (如 "A123")
        uid_str = str(row.get('user_id', 'unknown'))
        # 去除可能存在的 'A' 前缀以匹配 map (视你的 user2id 文件而定)
        # 如果 user2id map 里的 key 就是 "A123"，则不需要 replace
        # 这里为了稳健，直接查
        return self.user2id.get(uid_str, 0)  # 0 for unknown

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # 1. History & Hint 构建
        try:
            history_sids = eval(row['history_item_sid'])
        except:
            history_sids = row['history_item_sid'] if isinstance(row['history_item_sid'], list) else []
        history_str = ", ".join(history_sids)

        hint_text = ""
        # RL 阶段通常希望模型尽力表现最好，所以默认不 Dropout Hint，除非为了训练鲁棒性
        if 'history_item_id' in row and self.cf_hints:
            if random.random() >= self.hint_dropout_rate:  # Keep hint
                try:
                    hist_ids = eval(str(row['history_item_id']))
                    if hist_ids:
                        last_id = str(hist_ids[-1])
                        if last_id in self.cf_hints:
                            neighbors_str = self.cf_hints[last_id]
                            hint_text = f" [Hint: Users who bought the last item often also buy: {neighbors_str}.]"
                except:
                    pass

        # 2. Construct Prompt (SFT 风格)
        input_text = f"The user has interacted with items {history_str} in chronological order.{hint_text} Can you predict the next possible item that the user may expect?"

        instruction = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you predict the next possible item that the user may expect?

"""
        prompt_input = f"{instruction}### User Input: \n{input_text}\n\n### Response:\n"

        # 3. Target (Ground Truth)
        target_item = str(row['item_sid'])

        # 4. User Index for VeRA
        user_idx = self.get_user_idx(row)

        return {
            "user_idx": user_idx,
            "prompt": prompt_input,
            "target": target_item,
            "history_sids": history_sids  # 用于后续可能的复杂 Reward 计算
        }


# =================================================================
# 2. 辅助任务: Title 2 SID (用于增强语义理解)
# =================================================================
class Title2SidRLDataset(Dataset):
    def __init__(self, item_file, index_file, user2id_map, sample=-1, seed=0):
        self.user2id = user2id_map  # 这里其实用不到 User ID，给 0 即可
        random.seed(seed)
        with open(item_file, 'r') as f:
            self.item_feat = json.load(f)
        with open(index_file, 'r') as f:
            self.indices = json.load(f)

        self.data = []
        for item_id, sids in self.indices.items():
            if item_id in self.item_feat and len(sids) >= 3:
                combined_sid = "".join(sids[:3])
                title = self.item_feat[item_id]['title']
                self.data.append((title, combined_sid))

        if sample > 0: self.data = random.sample(self.data, sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        title, sid = self.data[idx]

        # Prompt
        prompt = f"### User Input: \nWhich item has title: {title}?\n\n### Response:\n"
        instruction = "Answer the question about item identification.\n\n"
        full_prompt = instruction + prompt

        return {
            "user_idx": 0,  # 辅助任务不涉及特定用户偏好
            "prompt": full_prompt,
            "target": sid,
            "history_sids": []
        }