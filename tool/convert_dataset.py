#!/usr/bin/env python3
"""
Convert dataset (Office/Industrial_and_Scientific) to MiniOneRec format with semantic IDs
and Collaborative Hints (Integrated Version)
"""

import json
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Any
import argparse

def load_dataset(data_dir: str, dataset_name: str) -> Dict[str, Any]:
    """Load all dataset files"""
    data = {}
    with open(os.path.join(data_dir, f'{dataset_name}.item.json'), 'r') as f:
        data['items'] = json.load(f)
    with open(os.path.join(data_dir, f'{dataset_name}.index.json'), 'r') as f:
        data['item_to_semantic'] = json.load(f)
    splits = {}
    for split in ['train', 'valid', 'test']:
        split_file = os.path.join(data_dir, f'{dataset_name}.{split}.inter')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                lines = f.readlines()[1:]
                splits[split] = [line.strip().split('\t') for line in lines if line.strip()]
    data['splits'] = splits
    return data

def semantic_tokens_to_id(tokens: List[str]) -> str:
    return ''.join(tokens)

def create_item_info_file(items: Dict[str, Dict], item_to_semantic: Dict[str, List], output_path: str):
    with open(output_path, 'w', encoding='utf-8') as f:
        for item_id, item_data in items.items():
            if item_id in item_to_semantic:
                semantic_tokens = item_to_semantic[item_id]
                semantic_id = semantic_tokens_to_id(semantic_tokens)
                item_title = item_data.get('title', f'Item_{item_id}')
                f.write(f"{semantic_id}\t{item_title}\t{item_id}\n")

def convert_interactions_to_csv(splits: Dict[str, List], items: Dict[str, Dict], 
                               item_to_semantic: Dict[str, List], output_dir: str, category: str = "Office_Products",
                               max_valid_samples: int = None, max_test_samples: int = None, seed: int = 42,
                               keep_longest_only: bool = True, hints_dict: Dict = None):
    """Convert interaction data to MiniOneRec CSV format using semantic IDs AND Hints"""
    
    import random
    random.seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, split_data in splits.items():
        rows = []
        user_to_longest = {}
        leak_prevented_count = 0
        
        is_train = (split_name == 'train')
        
        for line in split_data:
            if len(line) != 3: continue
            user_id, item_sequence, target_item = line
            
            if item_sequence.strip():
                history_item_ids = [int(x) for x in item_sequence.split()]
            else:
                history_item_ids = []
            
            target_item_id = int(target_item)
            
            # IDs -> SIDs
            history_semantic_ids = []
            for item_id in history_item_ids:
                if str(item_id) in item_to_semantic:
                    semantic_tokens = item_to_semantic[str(item_id)]
                    history_semantic_ids.append(semantic_tokens_to_id(semantic_tokens))
            
            target_semantic_id = None
            if str(target_item_id) in item_to_semantic:
                semantic_tokens = item_to_semantic[str(target_item_id)]
                target_semantic_id = semantic_tokens_to_id(semantic_tokens)
            
            if target_semantic_id is None: continue
            
            history_item_titles = []
            for item_id in history_item_ids:
                if str(item_id) in items:
                    title = items[str(item_id)].get('title', f'Item_{item_id}')
                    history_item_titles.append(title)
            
            target_title = items.get(str(target_item_id), {}).get('title', f'Item_{target_item_id}')

            # =================================================
            # [新增逻辑] Hint 注入与清洗
            # =================================================
            hint_text = ""
            if hints_dict and history_item_ids:
                last_id = str(history_item_ids[-1])
                if last_id in hints_dict:
                    raw_neighbors = hints_dict[last_id].split(', ')
                    
                    if is_train:
                        # 训练集：剔除 Target
                        clean_neighbors = [n for n in raw_neighbors if n.strip() != target_semantic_id]
                        if len(clean_neighbors) < len(raw_neighbors):
                            leak_prevented_count += 1
                    else:
                        # 测试集：保留所有
                        clean_neighbors = raw_neighbors
                    
                    if clean_neighbors:
                        hint_content = ", ".join(clean_neighbors)
                        hint_text = f" [Hint: Users who bought the last item often also buy: {hint_content}.]"
            # =================================================

            row = {
                'user_id': f'A{user_id}',
                'history_item_title': history_item_titles,
                'item_title': target_title,
                'history_item_id': history_item_ids,
                'item_id': target_item_id,
                'history_item_sid': history_semantic_ids,
                'item_sid': target_semantic_id,
                'safe_hint': hint_text  # 直接存入 safe_hint 列
            }
            
            if split_name == 'train' and keep_longest_only:
                sequence_length = len(history_item_ids)
                if user_id not in user_to_longest or sequence_length > len(user_to_longest[user_id]['history_item_id']):
                    user_to_longest[user_id] = row
            else:
                rows.append(row)
        
        if split_name == 'train' and keep_longest_only:
            rows = list(user_to_longest.values())
        
        # Sample limits
        if split_name == 'valid' and max_valid_samples is not None and len(rows) > max_valid_samples:
            rows = random.sample(rows, max_valid_samples)
        elif split_name == 'test' and max_test_samples is not None and len(rows) > max_test_samples:
            rows = random.sample(rows, max_test_samples)
        
        if rows:
            df = pd.DataFrame(rows)
            output_file = os.path.join(output_dir, f'{category}_5_2016-10-2018-11.csv')
            df.to_csv(output_file, index=False)
            print(f"Created {split_name} file: {output_file} with {len(rows)} rows")
            if is_train:
                print(f"🛡️  Training Set: Prevented leakage in {leak_prevented_count} samples.")

def main():
    parser = argparse.ArgumentParser(description='Convert dataset to MiniOneRec format with Hints')
    parser.add_argument('--data_dir', type=str, help='Path to dataset directory')
    parser.add_argument('--dataset_name', type=str, default='Industrial_and_Scientific')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--category', type=str, default=None)
    parser.add_argument('--max_valid_samples', type=int, default=None)
    parser.add_argument('--max_test_samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--keep_longest_only', action='store_true', default=False)
    parser.add_argument('--hints_file', type=str, default=None, help='Path to cf_hints.json') # 新增参数
    
    args = parser.parse_args()
    if args.category is None: args.category = args.dataset_name
    
    print(f"Loading {args.dataset_name} data from {args.data_dir}")
    dataset_data = load_dataset(args.data_dir, args.dataset_name)
    
    # Load Hints
    hints_dict = None
    if args.hints_file:
        print(f"Loading Hints from {args.hints_file}...")
        with open(args.hints_file, 'r') as f:
            hints_dict = json.load(f)
    
    for subdir in ['train', 'valid', 'test', 'info']:
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)
    
    info_file = os.path.join(args.output_dir, 'info', f'{args.category}_5_2016-10-2018-11.txt')
    create_item_info_file(dataset_data['items'], dataset_data['item_to_semantic'], info_file)
    
    for split_name in ['train', 'valid', 'test']:
        if split_name in dataset_data['splits']:
            split_output_dir = os.path.join(args.output_dir, split_name)
            convert_interactions_to_csv(
                {split_name: dataset_data['splits'][split_name]}, 
                dataset_data['items'],
                dataset_data['item_to_semantic'],
                split_output_dir,
                args.category,
                max_valid_samples=args.max_valid_samples,
                max_test_samples=args.max_test_samples,
                seed=args.seed,
                keep_longest_only=args.keep_longest_only,
                hints_dict=hints_dict  # 传入 hints
            )
    
    print(f"\nConversion completed! Data saved to {args.output_dir}")

if __name__ == '__main__':
    main()