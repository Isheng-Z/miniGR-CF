import pandas as pd
import fire
import torch
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformers import GenerationConfig, AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from GR.data import EvalSidDataset 
sys.path.append(".") 
from LogitProcessor import ConstrainedLogitsProcessor

def get_hash(x): return '-'.join([str(i) for i in x])

def main(
    base_model: str = "",
    test_data_path: str = "",
    info_file: str = "",
    category: str = "",
    result_json_data: str = "",
    num_beams: int = 20,
    cf_hints_path: str = "", # <--- 确保有这个参数
):
    print(f"Category: {category}")
    print(f"Loading Hints from: {cf_hints_path}") # 打印确认

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    
    with open(info_file) as f:
        sids = [l.split('\t')[0].strip() for l in f]
        sids_prompts = [f"### Response:\n{s}\n" for s in sids]
    
    prefix_ids = [tokenizer(s).input_ids for s in sids_prompts]
    hash_dict = {}
    p_idx = 3
    for ids in prefix_ids:
        ids = ids + [tokenizer.eos_token_id]
        for i in range(p_idx, len(ids)):
            key = get_hash(ids[p_idx:i] if i > p_idx else ids[:i])
            if key not in hash_dict: hash_dict[key] = set()
            hash_dict[key].add(ids[i])
    for k in hash_dict: hash_dict[k] = list(hash_dict[k])

    def allowed_fn(batch_id, input_ids):
        key = get_hash(input_ids)
        return hash_dict.get(key, [])

    # 实例化 Dataset (传入 hints path)
    ds = EvalSidDataset(
        train_file=test_data_path, 
        tokenizer=tokenizer, 
        max_len=2048, 
        category=category, 
        test=True, 
        cf_hints_path=cf_hints_path # <--- 传入！
    )
    
    encodings = [ds[i] for i in range(len(ds))]
    raw_data = ds.get_all()

    def eval_batch(batch):
        max_l = max(len(x['input_ids']) for x in batch)
        input_ids = []
        att_mask = []
        for x in batch:
            pad_l = max_l - len(x['input_ids'])
            input_ids.append([tokenizer.pad_token_id]*pad_l + x['input_ids'])
            att_mask.append([0]*pad_l + [1]*len(x['input_ids']))
        
        inputs = torch.tensor(input_ids).to(model.device)
        mask = torch.tensor(att_mask).to(model.device)
        
        clp = ConstrainedLogitsProcessor(allowed_fn, num_beams, base_model)
        
        with torch.no_grad():
            out = model.generate(
                inputs, 
                attention_mask=mask,
                max_new_tokens=256,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                logits_processor=LogitsProcessorList([clp]),
                pad_token_id=tokenizer.pad_token_id
            )
        
        res = tokenizer.batch_decode(out[:, max_l:], skip_special_tokens=True)
        return [res[i:i+num_beams] for i in range(0, len(res), num_beams)]

    from tqdm import tqdm
    all_res = []
    BATCH = 4
    for i in tqdm(range(0, len(encodings), BATCH)):
        all_res.extend(eval_batch(encodings[i:i+BATCH]))

    for i, r in enumerate(all_res):
        raw_data[i]['predict'] = r
        if 'dedup' in raw_data[i]: del raw_data[i]['dedup']
        
    with open(result_json_data, 'w') as f:
        json.dump(raw_data, f, indent=4)
    print(f"Results saved to {result_json_data}")

if __name__ == '__main__':
    fire.Fire(main)