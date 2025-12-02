import numpy as np
import faiss
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cf_emb', type=str, required=True, help='LightGCN向量路径')
    parser.add_argument('--sem_idx', type=str, required=True, help='语义ID索引路径')
    parser.add_argument('--out', type=str, default='cf_hints.json')
    parser.add_argument('--k', type=int, default=3)
    args = parser.parse_args()

    print(f"加载协同向量: {args.cf_emb}")
    cf_embs = np.load(args.cf_emb).astype(np.float32)
    faiss.normalize_L2(cf_embs) # 余弦相似度

    print("构建索引并搜索...")
    index = faiss.IndexFlatIP(cf_embs.shape[1])
    index.add(cf_embs)
    # 搜索 K+1 个，因为第一个通常是自己
    _, I = index.search(cf_embs, args.k + 1)

    print(f"加载语义 ID: {args.sem_idx}")
    with open(args.sem_idx, 'r') as f:
        sem_map = json.load(f)

    hints = {}
    cnt = 0
    for item_id in range(len(cf_embs)):
        # 排除自己
        neighbors = [n for n in I[item_id] if n != item_id][:args.k]
        
        # 转为语义 ID 字符串 (如 "<a_1><b_2>")
        neighbor_sids = []
        for nid in neighbors:
            nid_str = str(nid)
            if nid_str in sem_map:
                neighbor_sids.append("".join(sem_map[nid_str]))
        
        if neighbor_sids:
            hints[str(item_id)] = ", ".join(neighbor_sids)
            cnt += 1

    with open(args.out, 'w') as f:
        json.dump(hints, f, indent=2)
    print(f"✅ Hint 字典生成完毕: {cnt}/{len(cf_embs)} 物品有邻居。保存至 {args.out}")

if __name__ == '__main__':
    main()