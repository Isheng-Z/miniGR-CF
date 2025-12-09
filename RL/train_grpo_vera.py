import os
import torch
import numpy as np
import random
import argparse
import math
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# 导入 VeRA 模型 (确保 RL/vera.py 存在)
from vera import QwenVeRA

# 导入新写的 Data 类
from data_rl import SidRLDataset, Title2SidRLDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(
        base_model_path: str,
        train_file: str,
        user2id_path: str,
        output_dir: str,
        # RL Params
        num_generations: int = 4,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        beta_kl: float = 0.05,
        max_new_tokens: int = 12,
        epochs: int = 1,
        # Data Params
        sid_index_path: str = "",
        item_meta_path: str = "",
        cf_hints_path: str = ""  # 新增
):
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load User Map
    print(f"Loading user map: {user2id_path}")
    user2id = {}
    with open(user2id_path, 'r') as f:
        for line in f:
            # 兼容 user2id 文件格式 (A123 \t 0)
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                user2id[parts[0]] = int(parts[1])
    num_users = len(user2id)
    print(f"Total Users: {num_users}")

    # 3. Load Datasets
    print("Loading Datasets...")
    datasets_list = []

    # 主任务: SID Prediction (带 Hint)
    ds1 = SidRLDataset(
        train_file,
        user2id_map=user2id,
        tokenizer=tokenizer,
        cf_hints_path=cf_hints_path,
        hint_dropout_rate=0.1  # RL 稍微加一点 dropout 增加鲁棒性
    )
    datasets_list.append(ds1)

    # 辅助任务 (可选)
    if sid_index_path and item_meta_path:
        print("Adding Auxiliary Task: Title2Sid...")
        ds2 = Title2SidRLDataset(item_meta_path, sid_index_path, user2id, sample=2000)  # 采样部分防止主导
        datasets_list.append(ds2)

    train_dataset = ConcatDataset(datasets_list)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=None)

    # 4. Initialize VeRA Model
    print("Initializing VeRA...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, device_map="auto")
    model = QwenVeRA(base_model, num_users, rank=256)

    # 只优化 User Embeddings
    optimizer = torch.optim.AdamW(model.user_embeddings.parameters(), lr=learning_rate)

    # 5. Reward Function
    def compute_rewards(targets_batch, completions_batch):
        """
        targets_batch: [B] list of target strings
        completions_batch: [B * G] list of generated strings
        """
        rewards = []
        for i, target in enumerate(targets_batch):
            # 获取该 prompt 对应的 G 个生成
            start = i * num_generations
            end = start + num_generations
            group_gens = completions_batch[start:end]

            target_clean = target.strip()

            for gen in group_gens:
                gen_clean = gen.strip()
                # Rule Reward: 包含 Target 即为 1
                if target_clean in gen_clean:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
        return torch.tensor(rewards).to(device)

    # 6. Training Loop (Custom GRPO)
    print("Start Training...")
    model.train()

    global_step = 0
    for epoch in range(epochs):
        pbar = tqdm(dataloader)
        for batch in pbar:
            # Batch Data Unpacking
            # ConcatDataset 会返回 list of dicts，DataLoader 会 collate 成 dict of lists
            # {key: [batch_size]}
            user_idxs = batch['user_idx'].to(device)  # [B]
            prompts = batch['prompt']  # List[str] [B]
            targets = batch['target']  # List[str] [B]

            # Tokenize Prompts
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side='left').to(device)
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask

            # --- Rollout (Sampling) ---
            # Repeat for Group Generation: [B] -> [B * G]
            input_ids_rep = input_ids.repeat_interleave(num_generations, dim=0)
            att_mask_rep = attention_mask.repeat_interleave(num_generations, dim=0)
            user_idxs_rep = user_idxs.repeat_interleave(num_generations, dim=0)

            with torch.no_grad():
                # QwenVeRA.generate 处理并行激活
                outputs = model.generate(
                    input_ids=input_ids_rep,
                    user_ids=user_idxs_rep,
                    attention_mask=att_mask_rep,
                    do_sample=True,
                    temperature=1.0,
                    top_k=50,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id
                )

            # --- Reward Computation ---
            input_len = input_ids.shape[1]
            gen_ids = outputs[:, input_len:]
            gen_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

            rewards = compute_rewards(targets, gen_texts)  # [B * G]

            # Advantage (Group Norm)
            rewards_view = rewards.view(-1, num_generations)  # [B, G]
            mean = rewards_view.mean(dim=1, keepdim=True)
            std = rewards_view.std(dim=1, keepdim=True) + 1e-8
            advantages = (rewards_view - mean) / std
            advantages = advantages.view(-1)  # [B * G]

            # --- Policy Update ---
            # Forward pass to get gradients
            model_outputs = model(outputs, user_idxs_rep)  # [B*G, Seq, Vocab]
            logits = model_outputs.logits[:, :-1, :]
            labels = outputs[:, 1:]

            # Log Prob
            per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

            # Mask (只算生成的 token)
            mask = torch.zeros_like(labels)
            mask[:, input_len - 1:] = 1.0

            # Sum log prob
            seq_logps = (per_token_logps * mask).sum(dim=1)

            # Loss
            pg_loss = -(seq_logps * advantages).mean()

            # Optional KL Penalty (Approximate)
            # Ref model assumed to be VeRA with user_vec=0
            with torch.no_grad():
                # 临时清空 vector
                for layer in model.vera_layers: layer.current_user_vectors = None
                ref_out = base_model(outputs)
                ref_logits = ref_out.logits[:, :-1, :]
                ref_logps = torch.gather(ref_logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
                ref_seq_logps = (ref_logps * mask).sum(dim=1)

            kl = seq_logps - ref_seq_logps
            loss = pg_loss + beta_kl * kl.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            if global_step % 10 == 0:
                pbar.set_description(f"Loss: {loss.item():.4f} | R: {rewards.mean().item():.2f}")

    # Save
    print(f"Saving VeRA Embeddings to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_vera(os.path.join(output_dir, "user_embeddings.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--user2id_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output/vera_rl")
    parser.add_argument("--cf_hints_path", type=str, default="")

    # 兼容参数
    parser.add_argument("--sid_index_path", type=str, default="")
    parser.add_argument("--item_meta_path", type=str, default="")

    args = parser.parse_args()

    train(
        base_model_path=args.base_model_path,
        train_file=args.train_file,
        user2id_path=args.user2id_path,
        output_dir=args.output_dir,
        cf_hints_path=args.cf_hints_path,
        sid_index_path=args.sid_index_path,
        item_meta_path=args.item_meta_path
    )