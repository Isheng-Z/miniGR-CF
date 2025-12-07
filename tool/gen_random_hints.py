import json
import random
import argparse
import os
from tqdm import tqdm

def format_sid(codes):
    """
    鲁棒的 SID 格式化函数。
    能处理: [38, 14, 237] / ["a_38", "b_14"] / ["<a_38>", "<b_14>"]
    统一输出: "<a_38><b_14><c_237>"
    """
    prefixes = ['a', 'b', 'c', 'd', 'e'] 
    sid_parts = []
    
    for i, code in enumerate(codes):
        # 1. 转字符串并去除首尾空格
        code_str = str(code).strip()
        
        # 2. 关键修复：去除可能已存在的尖括号 < >
        # 这样 "<<a_38>>" 也会变成 "a_38"
        clean_code = code_str.replace('<', '').replace('>', '')
        
        # 3. 判断是否需要加层级前缀 (a_, b_...)
        if "_" in clean_code:
            # 已经是 a_38 格式，直接包裹
            sid_parts.append(f"<{clean_code}>")
        else:
            # 是纯数字 38，需要加前缀
            prefix = prefixes[i] if i < len(prefixes) else 'x'
            sid_parts.append(f"<{prefix}_{clean_code}>")
            
    return "".join(sid_parts)

def generate_random_hints(index_path, output_path, k=5, seed=42):
    random.seed(seed)
    print(f"🎲 Loading Item Index from {index_path}...")
    
    with open(index_path, 'r') as f:
        item_indices = json.load(f)
    
    all_item_ids = list(item_indices.keys())
    print(f"✅ Loaded {len(all_item_ids)} items.")
    
    # 1. 预处理：生成所有物品的标准 SID 字符串
    id2sid_str = {}
    print("Formatting SIDs...")
    for iid, codes in item_indices.items():
        if isinstance(codes, str):
            codes = [codes]
        id2sid_str[iid] = format_sid(codes)

    # 打印一个样本自检
    sample_id = all_item_ids[0]
    print(f"🔍 Sample SID Check: {id2sid_str[sample_id]}")

    # 2. 生成随机 Hints
    random_hints = {}
    print(f"🎲 Generating Random Hints (K={k})...")
    
    for iid in tqdm(all_item_ids):
        # 随机采样 K 个不同的 ID (排除自己)
        if len(all_item_ids) > k + 1:
            candidates = random.sample(all_item_ids, k + 1)
            neighbors = [x for x in candidates if x != iid][:k]
        else:
            neighbors = random.choices(all_item_ids, k=k)
        
        # 获取这些 neighbors 的 SID string
        neighbor_sids = [id2sid_str[n] for n in neighbors]
        
        # 拼接成字符串，用 ", " 分隔
        hints_str = ", ".join(neighbor_sids)
        random_hints[iid] = hints_str
        
    # 3. 保存 (带换行和缩进)
    print(f"💾 Saving to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # indent=4: 开启缩进和换行
        # ensure_ascii=False: 防止中文乱码(虽然SID通常没有中文)
        json.dump(random_hints, f, indent=4, ensure_ascii=False)
    
    print("Done! Check the file content now.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sid_index_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    
    generate_random_hints(args.sid_index_path, args.output_path, args.k)