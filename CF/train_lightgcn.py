import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from tqdm import tqdm
import scipy.sparse as sp
from collections import defaultdict

# ================= 配置区域 =================
LIGHTGCN_DIM = 64  # 协同向量维度
EPOCHS = 100 # 训练轮数
BATCH_SIZE = 2048
LR = 0.0001
TOP_K = [10, 20]  # 评估指标 K 值


# ===========================================

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, n_layers=3):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.n_layers = n_layers
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        # 初始化权重
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        self.graph = None

    def forward(self):
        all_embs = [torch.cat([self.user_emb.weight, self.item_emb.weight])]
        for layer in range(self.n_layers):
            all_embs.append(torch.sparse.mm(self.graph, all_embs[-1]))
        all_embs = torch.stack(all_embs, dim=1)
        final_embs = torch.mean(all_embs, dim=1)
        users, items = torch.split(final_embs, [self.num_users, self.num_items])
        return users, items


def build_graph(num_users, num_items, interactions):
    print("构建图结构...")
    src, dst = [], []
    for u, i in interactions:
        src.append(u);
        dst.append(i + num_users)
        src.append(i + num_users);
        dst.append(u)

    src = np.array(src);
    dst = np.array(dst)
    # 构建邻接矩阵
    adj = sp.coo_matrix((np.ones(len(src)), (src, dst)),
                        shape=(num_users + num_items, num_users + num_items), dtype=np.float32)

    # 归一化 (D^-0.5 * A * D^-0.5)
    rowsum = np.array(adj.sum(1))
    # 修复 divide by zero 警告: 加上 1e-9 防止除零
    d_inv_sqrt = np.power(rowsum + 1e-9, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

    indices = torch.from_numpy(np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64))
    values = torch.from_numpy(norm_adj.data)
    return torch.sparse_coo_tensor(indices, values, torch.Size(norm_adj.shape))


def load_data(data_dir, dataset_name):
    # 1. 读取 ID 映射数量
    with open(os.path.join(data_dir, dataset_name, f"{dataset_name}.user2id"), 'r') as f:
        num_users = len(f.readlines())
    with open(os.path.join(data_dir, dataset_name, f"{dataset_name}.item2id"), 'r') as f:
        num_items = len(f.readlines())

    print(f"Dataset: Users {num_users}, Items {num_items}")

    # 2. 读取训练集 (用于构建图和Mask)
    train_file = os.path.join(data_dir, dataset_name, f"{dataset_name}.train.inter")
    train_interactions = []
    train_history_per_user = defaultdict(set)

    with open(train_file, 'r') as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3: continue
            u = int(parts[0])
            # 解析历史 (History)
            if parts[1]:
                for i_str in parts[1].split(' '):
                    i_hist = int(i_str)
                    train_interactions.append((u, i_hist))
                    train_history_per_user[u].add(i_hist)
            # 解析目标 (Target)
            i_target = int(parts[2])
            train_interactions.append((u, i_target))
            train_history_per_user[u].add(i_target)

    # 去重交互列表用于构图
    train_interactions = list(set(train_interactions))

    # 3. 读取测试集 (用于评估)
    test_file = os.path.join(data_dir, dataset_name, f"{dataset_name}.test.inter")
    test_interactions = []
    if os.path.exists(test_file):
        with open(test_file, 'r') as f:
            next(f)
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3: continue
                u = int(parts[0])
                i = int(parts[2])  # 只取 Target Item 做测试
                test_interactions.append((u, i))
    else:
        print("⚠️ 未找到测试文件，将跳过评估。")

    return num_users, num_items, train_interactions, train_history_per_user, test_interactions


def evaluate(model, device, test_interactions, train_history, num_items):
    """计算 Recall@K 和 NDCG@K"""
    model.eval()
    with torch.no_grad():
        users_emb, items_emb = model()

    # 按用户分组测试数据
    user_test_dict = defaultdict(list)
    for u, i in test_interactions:
        user_test_dict[u].append(i)

    hits = {k: 0.0 for k in TOP_K}
    ndcgs = {k: 0.0 for k in TOP_K}
    num_test_users = 0

    print("正在评估 (Evaluation)...")

    # 逐个用户计算 (简单实现，显存友好的方式)
    for u, targets in tqdm(user_test_dict.items(), leave=False):
        if u >= len(users_emb): continue
        num_test_users += 1

        # 1. 计算该用户对所有物品的得分
        u_emb = users_emb[u].unsqueeze(0)  # [1, dim]
        scores = torch.mm(u_emb, items_emb.t()).squeeze(0)  # [num_items]

        # 2. Mask 掉训练集中已经见过的物品 (防止推荐已读)
        if u in train_history:
            mask_indices = list(train_history[u])
            scores[mask_indices] = -float('inf')

        # 3. 取 Top-K
        max_k = max(TOP_K)
        _, topk_indices = torch.topk(scores, max_k)
        topk_indices = topk_indices.cpu().tolist()

        # 4. 计算指标
        for k in TOP_K:
            pred_list = topk_indices[:k]
            for target in targets:
                if target in pred_list:
                    hits[k] += 1.0
                    rank = pred_list.index(target)
                    ndcgs[k] += 1.0 / np.log2(rank + 2)

    # 打印结果
    print("\n========= 评估结果 =========")
    for k in TOP_K:
        print(f"HR@{k}: {hits[k] / num_test_users:.4f} | NDCG@{k}: {ndcgs[k] / num_test_users:.4f}")
    print("============================")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. 加载数据
    num_users, num_items, train_inters, train_history, test_inters = load_data(args.data_dir, args.dataset)

    # 2. 模型与图
    graph = build_graph(num_users, num_items, train_inters).to(device)
    model = LightGCN(num_users, num_items, LIGHTGCN_DIM).to(device)
    model.graph = graph
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 3. 训练
    print(f"开始训练 LightGCN ({EPOCHS} epochs)...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        # 简单 Batch BPR 训练
        n_batch = len(train_inters) // BATCH_SIZE
        interaction_tensor = torch.tensor(train_inters)
        indices = torch.randperm(len(train_inters))

        # 进度条只显示 Loss，不刷屏
        with tqdm(total=n_batch, desc=f"Epoch {epoch + 1}", leave=False) as pbar:
            for i in range(n_batch):
                batch_idx = indices[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
                batch = interaction_tensor[batch_idx].to(device)

                users_idx = batch[:, 0]
                pos_items_idx = batch[:, 1]
                neg_items_idx = torch.randint(0, num_items, (len(batch),)).to(device)

                optimizer.zero_grad()
                users_emb, items_emb = model()

                u_emb = users_emb[users_idx]
                pos_emb = items_emb[pos_items_idx]
                neg_emb = items_emb[neg_items_idx]

                pos_scores = torch.sum(u_emb * pos_emb, dim=1)
                neg_scores = torch.sum(u_emb * neg_emb, dim=1)

                # BPR Loss
                loss = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))
                # Reg Loss
                reg = (u_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)) / len(batch)
                loss += 1e-4 * reg

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                pbar.update(1)

        print(f"Epoch {epoch + 1} done. Avg Loss: {total_loss / n_batch:.4f}")

        # 每 5 轮评估一次 (或者只在最后评估)
        if (epoch + 1) % 5 == 0:
            evaluate(model, device, test_inters, train_history, num_items)

    # 4. 保存
    model.eval()
    with torch.no_grad():
        _, item_embs = model()
        item_embs = item_embs.cpu().numpy()

    # 自动创建输出目录
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        print(f"Directory {output_dir} does not exist. Creating it...")
        os.makedirs(output_dir, exist_ok=True)

    np.save(args.output_path, item_embs)
    print(f"\n✅ CF Embedding 已保存: {args.output_path} (Shape: {item_embs.shape})")


if __name__ == '__main__':
    main()