import torch
import torch.nn as nn
from transformers import PreTrainedModel


class VeRALinear(nn.Module):
    def __init__(self, base_layer, rank, device=None, dtype=None):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank

        # 共享矩阵 (Frozen)
        self.S = nn.Parameter(torch.randn(base_layer.in_features, rank, device=device, dtype=dtype) / rank ** 0.5,
                              requires_grad=False)
        self.T = nn.Parameter(torch.zeros(rank, base_layer.out_features, device=device, dtype=dtype),
                              requires_grad=False)

        # 临时存储当前 Batch 的 User Vectors (由 QwenVeRA 注入)
        self.current_user_vectors = None

    def forward(self, x):
        # 1. Base Forward (Frozen)
        base_out = self.base_layer(x)  # [Batch, Seq, Out]

        # 2. VeRA Forward (Parallel)
        if self.current_user_vectors is None:
            return base_out

        # user_vectors: [Batch, 2 * rank] -> split to b, d
        # 这里的 Batch 维度必须和 x 的 Batch 维度对齐
        vec = self.current_user_vectors

        # 校验 Batch Size (处理 GRPO 采样时的维度扩展)
        if vec.shape[0] != x.shape[0]:
            # 如果输入 x 是 user_vec 的 G 倍 (因为生成了 G 个样本)
            ratio = x.shape[0] // vec.shape[0]
            vec = vec.repeat_interleave(ratio, dim=0)

        b_vec, d_vec = torch.chunk(vec, 2, dim=-1)  # [Batch, Rank]

        # 投影到低秩空间 [Batch, Seq, Rank]
        low_rank = x @ self.S

        # 关键：并行注入个性化参数
        # [Batch, Seq, Rank] * [Batch, 1, Rank] -> Broadcasting
        low_rank = low_rank * b_vec.unsqueeze(1) * d_vec.unsqueeze(1)

        # 投影回输出空间
        delta_out = low_rank @ self.T

        return base_out + delta_out


class QwenVeRA(nn.Module):
    def __init__(self, base_model, num_users, rank=256):
        super().__init__()
        self.base_model = base_model
        self.rank = rank
        self.num_users = num_users

        # 冻结基座
        for param in self.base_model.parameters():
            param.requires_grad = False

        # 替换 Linear 层
        self.vera_layers = nn.ModuleList()
        # 针对 Qwen2.5 的模块名
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        for name, module in self.base_model.named_modules():
            if any(t in name.split('.')[-1] for t in target_modules) and isinstance(module, nn.Linear):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = self.base_model.get_submodule(parent_name)

                vera_layer = VeRALinear(module, rank, device=module.weight.device, dtype=module.weight.dtype)
                setattr(parent, child_name, vera_layer)
                self.vera_layers.append(vera_layer)

        # 用户独立参数 (Trainable) - 存显存
        # 维度: [Users, Layers * 2 * Rank]
        total_dim = len(self.vera_layers) * (2 * rank)
        print(f"🧠 Allocating User Embeddings: [{num_users}, {total_dim}]")
        self.user_embeddings = nn.Embedding(num_users, total_dim).to(base_model.device)
        # 初始化
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        self.user_embeddings.weight.requires_grad = True

    def forward(self, input_ids, user_ids, **kwargs):
        # 1. 查表获取当前 Batch 的参数 [Batch, Total_Dim]
        all_vecs = self.user_embeddings(user_ids)

        # 2. 拆分到每一层
        batch_size = user_ids.size(0)
        layer_dim = 2 * self.rank
        all_vecs = all_vecs.view(batch_size, len(self.vera_layers), layer_dim)

        # 3. 注入到各个 Layer 中 (State Injection)
        for i, layer in enumerate(self.vera_layers):
            layer.current_user_vectors = all_vecs[:, i, :]

        # 4. 执行基座的前向传播 (会触发 VeRALinear.forward)
        return self.base_model(input_ids, **kwargs)

    def generate(self, input_ids, user_ids, **kwargs):
        # 类似于 forward，先注入参数，再调用 generate
        # 注意：这里 user_ids 只需要传入原始 Batch 的 ID
        # generate 内部扩展 input_ids 时，VeRALinear 会自动 repeat_interleave
        with torch.no_grad():
            # 这一步是为了注入 current_user_vectors
            self.forward(input_ids, user_ids)

        return self.base_model.generate(input_ids=input_ids, **kwargs)

    def save_vera(self, path):
        torch.save(self.user_embeddings.state_dict(), path)