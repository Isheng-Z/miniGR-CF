这是一份为你定制的 `README.md`，它完全适配你整理好的 `miniGR-CF` 项目结构。

这份文档严格遵循了你的要求：

1.  **从零开始**：涵盖环境配置、数据下载、预处理到训练的全流程。
2.  **单行命令**：所有的 Shell 命令都已合并为一行，方便直接复制运行，避免换行符带来的兼容性问题。
3.  **逻辑对齐**：参数配置（如时间窗口、数据量、Hint注入）均对应我们最终确定的“复刻+增强”方案。

你可以直接将下面的内容保存为 `README.md`。

-----

# miniGR-CF: Collaborative-Enhanced Generative Recommendation

**miniGR-CF** 是一个增强版的生成式推荐系统框架。它在原版 MiniOneRec (Qwen-0.5B) 的基础上，创新性地引入了 **LightGCN 协同提示 (Collaborative Hints)** 机制。通过将协同信号作为 Prompt 注入，并配合 **动态 Hint Dropout** 和 **防泄露清洗** 策略，本项目在保持语义泛化能力的同时，显著提升了推荐的排序精度。

## 📂 项目结构

```text
miniGR-CF/
├── data/
│   ├── raw/                 # [需手动下载] 存放 Amazon 原始 .json.gz 文件
│   └── processed/           # [自动生成] 中间数据 (.inter, .json, .npy)
├── models/
│   └── Qwen2.5-0.5B/        # [需下载] 预训练模型权重
├── src/                     # 核心源码
│   ├── process_raw.py       # 数据清洗 (含 2017-2018 时间过滤)
│   ├── generate_sids.py     # 语义 ID 生成 (RQ-VAE/KMeans)
│   ├── train_lightgcn.py    # 协同信号提取
│   ├── generate_hints.py    # Hint 字典生成
│   ├── convert_dataset.py   # SFT 数据转换 (主任务精简)
│   ├── dataset.py           # 动态 Dataset (含 Dropout/多任务)
│   ├── train.py             # SFT 训练入口 (5任务混合)
│   ├── evaluate.py          # 推理生成
│   └── metrics.py           # 指标计算
└── output/                  # 训练日志与模型保存
```

## 🛠️ 1. 环境准备

推荐使用 Conda 环境（Python 3.10+）：

```bash
conda create -n minigr python=3.10 -y && conda activate minigr
pip install torch>=2.0.0 transformers accelerate pandas numpy scipy scikit-learn fire tqdm
conda install -c pytorch faiss-gpu
```

## 📥 2. 资源准备

### 2.1 下载模型

请下载 Qwen2.5-0.5B 模型至 `models/` 目录：

```bash
huggingface-cli download --repo-type model "Qwen/Qwen2.5-0.5B" --local-dir "models/Qwen2.5-0.5B" --local-dir-use-symlinks False
```

### 2.2 下载数据

本项目默认使用 **Industrial and Scientific** 数据集。请从 [UCSD Amazon Data](https://nijianmo.github.io/amazon/index.html) 下载以下两个文件并放入 `data/raw/`：

  * `Industrial_and_Scientific_5.json.gz`
  * `meta_Industrial_and_Scientific.json.gz`

-----

## 🚀 3. 运行全流程 (Step-by-Step Pipeline)

请按顺序执行以下命令。所有命令均已设计为单行执行。

### Step 1: 数据清洗 (Data Processing)

解析原始数据，执行 K-Core 过滤，并按 **2017.10-2018.11** 时间窗口切分数据。

```bash
python src/process_raw.py --raw_dir ./data/raw --out_dir ./data/processed --cat Industrial_and_Scientific --st_year 2017 --st_month 10 --ed_year 2018 --ed_month 11
```

### Step 2: 生成语义 ID (Semantic IDs)

使用 Qwen 提取商品标题向量，并通过层级聚类生成 3 层语义 ID。

```bash
python src/generate_sids.py --data_dir ./data/processed --model_path ./models/Qwen2.5-0.5B --cat Industrial_and_Scientific
```

### Step 3: 提取协同信号 (Collaborative Signals)

训练 LightGCN 模型以捕获用户行为模式，并导出物品协同向量。

```bash
python src/train_lightgcn.py --data_dir ./data/processed --cat Industrial_and_Scientific
```

### Step 4: 生成协同提示 (Generate Hints)

基于 LightGCN 向量检索每个物品的 Top-K 互补邻居，生成提示字典。

```bash
python src/generate_hints.py --data_dir ./data/processed --cat Industrial_and_Scientific
```

### Step 5: 准备 SFT 数据 (Prepare Data)

生成精简版（Keep Longest Only）的主任务训练数据，并在生成时注入 Hint。

```bash
python src/convert_dataset.py --data_dir ./data/processed --cat Industrial_and_Scientific --out_dir ./data/sft_ready --hints_file ./data/processed/hints.json --keep_longest_only
```

### Step 6: SFT 训练 (Training)

启动多任务混合训练。

  * **数据构成**：主任务 (5760条) + 4个辅助任务 (各采样5760条) ≈ 2.88万条数据。
  * **机制**：主任务启用 **Hint Dropout (p=0.3)**，训练 10 Epochs (约 1.8万步)。

<!-- end list -->

```bash
python src/train.py --out_dir ./output/sft_final --batch_size 128 --micro_batch_size 16 --epochs 10 --dropout 0.3
```

### Step 7: 评估 (Evaluation)

使用训练好的模型在测试集上进行 Beam Search 推理，并计算 HR@K 和 NDCG@K。
*(注：推理时自动保留完整 Hint)*

**生成预测结果：**

```bash
python src/evaluate.py --model_path ./output/sft_final/final_checkpoint --data_dir ./data/processed --output_file ./output/eval_result.json --num_beams 20
```

**计算指标：**

```bash
python src/metrics.py --file ./output/eval_result.json
```

-----

## 📊 实验对照

| Experiment | Configuration | Hint Strategy |
| :--- | :--- | :--- |
| **Baseline** | 原版复现 | 无 Hint |
| **miniGR-CF** | **LightGCN 增强** | **Train: Dropout(0.3) & Clean Target / Test: Full Hint** |

*注：本项目通过 `dataset.py` 实现了动态防泄露逻辑，训练时会自动剔除 Hint 中的 Target Item，防止标签泄露。*