# M1（草案）：MoE 逐专家循环 → token 排序/分桶

> 状态：**未实施**。本文件只描述候选改动与验证方案；实施前需经确认，并以独立 diff 提交。

## 现状（dacc687 `model/model_minimind.py` MOEFeedForward.forward）
```python
y = torch.zeros_like(x_flat)
for i, expert in enumerate(self.experts):
    mask = (topk_idx == i)               # bool 张量 + any/nonzero
    if mask.any():
        token_idx = mask.any(dim=-1).nonzero().flatten()
        weight = topk_weight[mask].view(-1, 1)
        y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
    elif self.training:
        y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())  # 保证未用专家有梯度
```
问题（基于实测）：每 step ≈4,038 个 CUDA kernel，GPU 实际 busy 仅 ~29ms/step（launch-bound）；MoE 循环内的 mask/nonzero/index_select/index_add 进一步放大 kernel 数。

## 方案 A（推荐，低风险）：排序 + 连续切片
```python
T, H, K, E = x_flat.shape[0], x_flat.shape[1], self.config.num_experts_per_tok, self.config.num_experts
flat_idx = topk_idx.reshape(-1)                     # (T*K,)
flat_w = topk_weight.reshape(-1)                    # (T*K,)
tok = torch.arange(T, device=x.device).repeat_interleave(K)
order = torch.argsort(flat_idx, stable=True)
sorted_idx = flat_idx[order]
counts = torch.bincount(sorted_idx, minlength=E)
offsets = torch.cumsum(counts, 0)
y = torch.zeros_like(x_flat)
start = 0
for i in range(E):
    end = int(offsets[i])
    if end > start:
        pos = order[start:end]
        tokens = tok[pos]
        w = flat_w[pos].unsqueeze(-1)
        y.index_add_(0, tokens, (self.experts[i](x_flat[tokens]) * w).to(y.dtype))
    start = end
```
保留 K=1 的 straight-through 语义（`topk_weight` 仍参与前向乘积与反向），aux_loss 计算不变；未用专家由反向图自然覆盖（无需 dummy backward）。

## 方案 B（激进，暂缓）：堆叠权重 + padded batched GEMM
需要把各专家权重 `torch.stack` 成 `(E,H,I)` 后 `torch.bmm`。缺点：每次 forward 都要 stack（或缓存随 step 失效），且会改变参数组织方式、破坏与官方 checkpoint 的 state_dict 兼容性；4090（sm89）也不支持 Hopper 专属 grouped GEMM。暂不实施。

## 验证方案
1. 等价性单测（fp32、单 batch、E=4/K=1 与 K=2）：输出 max|Δ| < 1e-5；gate/专家权重梯度 max|Δ| < 1e-6。
2. 真实数据等价性：前 100 步 loss max|Δ| < 1e-4（同 seed）。
3. 性能：500 step 吞吐、kernel/step、GPU util、VRAM 峰值；compile=0/1 分别测。
4. 判定：数值等价 + 吞吐提升 ≥10% 才进入受控训练。
