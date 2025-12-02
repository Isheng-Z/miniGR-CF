import os
import sys
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, AutoConfig
from torch.utils.data import ConcatDataset
import fire
import json
import random
import numpy as np
from data import SidSFTDataset, SidItemFeatDataset, FusionSeqRecDataset, SFTData, TitleHistory2SidSFTDataset

class TokenExtender:
    def __init__(self, data_path, dataset, index_file=".index.json"):
        self.data_path = data_path
        self.dataset = dataset
        self.index_file = index_file
    def get_new_tokens(self):
        full_path = os.path.join(self.data_path, self.dataset + self.index_file)
        if os.path.exists(full_path):
            with open(full_path, 'r') as f:
                indices = json.load(f)
            tokens = set()
            for idx_list in indices.values():
                for t in idx_list: tokens.add(t)
            return sorted(list(tokens))
        return []

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train(
    base_model: str = "",
    train_file: str = "",
    eval_file: str = "",
    output_dir: str = "",
    sample: int = -1,
    seed: int = 42,
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 1024,
    freeze_LLM: bool = False,
    resume_from_checkpoint: str = None,
    category: str = "",
    train_from_scratch: bool = False,
    sid_index_path: str = "",
    item_meta_path: str = "",
    cf_hints_path: str = "", 
):
    set_seed(seed)
    print(f"Category: {category}")
    
    gradient_accumulation_steps = batch_size // micro_batch_size
    device_map = "auto"

    if not train_from_scratch:
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map=device_map)
    else:
        config = AutoConfig.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_config(config)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Token Extension
    if sid_index_path and os.path.exists(sid_index_path):
        print(f"Extending tokens from {sid_index_path}")
        extender = TokenExtender(os.path.dirname(sid_index_path), os.path.basename(sid_index_path).split('.')[0], ".index.json")
        new_tokens = extender.get_new_tokens()
        if new_tokens:
            print(f"Adding {len(new_tokens)} new tokens.")
            tokenizer.add_tokens(new_tokens)
            model.resize_token_embeddings(len(tokenizer))
            
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data
            old_vocab_size = input_embeddings.shape[0] - len(new_tokens)
            mu = torch.mean(input_embeddings[:old_vocab_size], dim=0)
            for i in range(old_vocab_size, input_embeddings.shape[0]):
                input_embeddings[i] = mu + torch.randn_like(mu) * 0.01
                output_embeddings[i] = input_embeddings[i]

    if freeze_LLM:
        print("Freezing LLM parameters...")
        for param in model.parameters(): param.requires_grad = False
        model.get_input_embeddings().weight.requires_grad = True 

    print("Loading Dataset (Multi-task Mode)...")
    train_datasets = []

    # 1. 核心推荐任务 (带 Hint + Dropout)
    print(">>> Loading Task 1: SidSFTDataset")
    train_data1 = SidSFTDataset(
        train_file=train_file,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        category=category,
        cf_hints_path=cf_hints_path,
        hint_dropout_rate=0.3  # <--- 【关键】开启训练时的随机丢弃 (30%)
    )
    train_datasets.append(train_data1)
    
    # 2. 语义对齐任务
    print(">>> Loading Task 2: SidItemFeatDataset")
    train_data2 = SidItemFeatDataset(
        item_file=item_meta_path,
        index_file=sid_index_path,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        category=category
    )
    train_datasets.append(train_data2)
    
    # 3. 混合序列任务
    print(">>> Loading Task 3: FusionSeqRecDataset")
    train_data3 = FusionSeqRecDataset(train_file, item_meta_path, sid_index_path, tokenizer, cutoff_len, sample, seed=seed, category=category)
    train_datasets.append(train_data3)
    
    # 4. SFTData
    print(">>> Loading Task 4: SFTData")
    train_data4 = SFTData(train_file, tokenizer, cutoff_len, sample, seed=seed, category=category)
    train_datasets.append(train_data4)
    
    # 5. TitleHistory
    print(">>> Loading Task 5: TitleHistory")
    train_data5 = TitleHistory2SidSFTDataset(train_file, item_meta_path, sid_index_path, tokenizer, cutoff_len, sample, seed=seed, category=category)
    train_datasets.append(train_data5)
    
    train_dataset = ConcatDataset(train_datasets)
    
    # 验证集 (关闭 Dropout，保证验证指标稳定)
    val_dataset = SidSFTDataset(
        train_file=eval_file,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        category=category,
        test=False, # 仍需计算 Loss，所以不是 Test 模式
        cf_hints_path=cf_hints_path,
        hint_dropout_rate=0.0 # <--- 【关键】验证集不丢弃 Hint
    )

    print(f"Total Train Size: {len(train_dataset)}")

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        bf16=True,
        logging_steps=10,
        output_dir=output_dir,
        save_strategy="steps",
        eval_strategy="steps",
        save_steps=0.2,          
        eval_steps=0.2,          
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none", 
        dataloader_num_workers=0
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, padding=True, pad_to_multiple_of=8)
    )
    
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)
    
    final_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

if __name__ == "__main__":
    fire.Fire(train)