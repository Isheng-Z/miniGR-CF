
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
python tool/amazon18_data_process.py --dataset Industrial_and_Scientific --reviews_file ./data/raw/Industrial_and_Scientific_5.json --metadata_file ./data/raw/meta_Industrial_and_Scientific.json --user_k 5 --item_k 5 --st_year 2017 --st_month 10 --ed_year 2018 --ed_month 11 --output_path ./data/processed     
```

### Step 2: 生成语义 ID (Semantic IDs)

使用 Qwen 提取商品标题向量，并通过层级聚类生成 3 层语义 ID。

```bash
python tool/amazon_text2emb.py --dataset Industrial_and_Scientific --root ./data/processed/Industrial_and_Scientific --plm_name qwen --plm_checkpoint "./models/Qwen2.5-0.5B"
```
```bash
python RQ/rqkmeans_faiss.py --dataset Industrial_and_Scientific --data_path data/processed/Industrial_and_Scientific/embeddings/Industrial_and_Scientific.emb-qwen-td.npy
```
### Step 3: 提取协同信号 (Collaborative Signals)

训练 LightGCN 模型以捕获用户行为模式，并导出物品协同向量。

```bash
python CF/train_lightgcn.py   --dataset "Industrial_and_Scientific"   --data_dir "./data/processed"   --output_path "./data/processed/Industrial_and_Scientific/lightgcn_emb.npy"
```

### Step 4: 生成协同提示 (Generate Hints)

基于 LightGCN 向量检索每个物品的 Top-K 互补邻居，生成提示字典。

```bash
python tool/gen_hints.py   --cf_emb "./data/processed/Industrial_and_Scientific/lightgcn_emb.npy"   --sem_idx "./data/processed/Industrial_and_Scientific/Industrial_and_Scientific.index.json"   --out "./data/processed/Industrial_and_Scientific/cf_hints.json"
```

### Step 5: 准备 SFT 数据 (Prepare Data)

生成精简版（Keep Longest Only）的主任务训练数据，并在生成时注入 Hint。

```bash
python tool/convert_dataset.py   --dataset_name Industrial_and_Scientific   --data_dir ./data/processed/Industrial_and_Scientific   --output_dir ./data/sft_ready  --keep_longest_only   --hints_file "./data/processed/Industrial_and_Scientific/cf_hints.json"
```

### Step 6: SFT 训练 (Training)

启动多任务混合训练。

  * **数据构成**：主任务 (5760条) + 4个辅助任务 (各采样5760条) ≈ 2.88万条数据。
  * **机制**：主任务启用 **Hint Dropout (p=0.3)**，训练 10 Epochs (约 1.8万步)。

<!-- end list -->

```bash
python GR/sft.py   --category "Industrial_and_Scientific"   --output_dir "./output/sft_hints"   --base_model "./models/Qwen2.5-0.5B"   --train_file "./data/sft_ready/train/Industrial_and_Scientific_5_2016-10-2018-11.csv"   --eval_file "./data/sft_ready/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv"   --sid_index_path "./data/processed/Industrial_and_Scientific/Industrial_and_Scientific.index.json"   --item_meta_path "./data/processed/Industrial_and_Scientific/Industrial_and_Scientific.item.json"   --learning_rate 2e-5   --micro_batch_size 8   --batch_size 16   --num_epochs 10   --cutoff_len 1024 --cf_hints_path data/processed/Industrial_and_Scientific/cf_hints.json
```

### Step 7: 评估 (Evaluation)

使用训练好的模型在测试集上进行 Beam Search 推理，并计算 HR@K 和 NDCG@K。
*(注：推理时自动保留完整 Hint)*

**生成预测结果：**

```bash
python tool/evaluate.py   --category "Industrial_and_Scientific"   --base_model "./output/sft/final_checkpoint"   --test_data_path "./data/sft_ready/test/Industrial_and_Scientific_5_2016-10-2018-11.csv"   --info_file "./data/sft_ready/info/Industrial_and_Scientific_5_2016-10-2018-11.txt"   --result_json_data "./output/eval_final.json"   --num_beams 20   --cf_hints_path "./data/processed/Industrial_and_Scientific/cf_hints.json"
```

**计算指标：**

```bash
python tool/calc.py --file ./output/eval_result.json
```

-----

## 📊 实验对照

| Experiment | Configuration | Hint Strategy |
| :--- | :--- | :--- |
| **Baseline-0.7b** | 原版0.7b复现 | 无 Hint |
| **Baseline** | 原版的结果 | 无hint |
| **miniGR-CF** | **采用qwen2.5-0.7bLightGCN 增强hints** | **Train: Dropout(0.3) & Clean Target / Test: Full Hint** |

## 🔖 Citation & Acknowledgement

本项目主要基于以下优秀开源工作进行改进：

  * **MiniOneRec**: An Open-Source Framework for Scaling Generative Recommendation.

      * GitHub: [https://github.com/Isheng-Z/MiniOneRec](https://github.com/Isheng-Z/MiniOneRec)
      * Paper: [arXiv:2510.24431](https://arxiv.org/abs/2510.24431)

  * **LightGCN**: Simplifying and Powering Graph Convolution Network for Recommendation.

      * Paper: [SIGIR 2020](https://arxiv.org/abs/2002.02126)

如果您在研究中使用了本项目或原始 MiniOneRec 代码，请引用：

```bibtex
@misc{MiniOneRec,
  title={MiniOneRec: An Open-Source Framework for Scaling Generative Recommendation},
  author={Xiaoyu Kong and Leheng Sheng and Junfei Tan and Yuxin Chen and Jiancan Wu and An Zhang and Xiang Wang and Xiangnan He},
  year={2025},
  eprint={2510.24431},
  archivePrefix={arXiv},
  primaryClass={cs.IR}
}
```

## 🙏 致谢

感谢 [MiniOneRec Team](https://github.com/AkaliKong/MiniOneRec/issues) 提供的代码基础和数据处理脚本。本项目的核心架构（SFT 多任务训练、RQ-kmeans ID 生成）均复用于该仓库，并在此基础上增加了协同信号增强模块。
