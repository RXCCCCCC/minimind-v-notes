# MiniMind-V 训练计划与执行状态

> 项目：MiniMind-V (MoE 200M-A65M) 完整复现
> 目标：产出可用于科研简历 / 联系导师的完整实验记录
> 最后更新：2026-09-24
> 维护位置：本文件 + `/root/autodl-tmp/minimind-v-research/PROJECT_LOG.md`（已完成记录）+ `/root/autodl-tmp/logs/`（原始日志）

---

## 0. 项目目标与验收标准

不只是"跑通"，而是完整走通：

```
官方环境复现 → 官方 inference → Dataset 检查 → Pretrain 冒烟 → SFT 冒烟
→ 完整 VLM Pretrain → 完整 VLM SFT → Evaluation → Baseline
→ Modification → Controlled Experiment → Ablation → 整理成项目材料
```

**验收要点**：每阶段有可复现命令、有实测数据、异常有根因分析。

---

## 1. 路线图与当前状态

| # | 阶段 | 状态 | 证据 / 说明 |
|---|------|------|-------------|
| 0 | 环境盘点与可复现起点 | ✅ 完成 | `ENV_INVENTORY.md`、`pip_freeze_2026-09-22.txt`、`conda_env_minimind-v.yml` |
| 1 | 官方环境复现 | ✅ 完成 | requirements.txt 31 个包版本全部对齐，`pip check` 无冲突 |
| 2 | 官方模型 inference | ✅ 完成 | `eval_vlm.py --load_from model --weight sft_vlm --use_moe 1`，13 张图全过，**108.9 tokens/s** |
| 3 | Dataset 检查 | ✅ 完成 | pretrain **1,274,698** 行 / sft **2,904,511** 行；schema = `conversations`(string) + `image_bytes`(binary) |
| 4 | Pretrain Smoke Test | ✅ 完成 | 512 样本 / 256 步；loss 2.86→2.59；trainable **1.183M**；RC=0 |
| 5 | SFT Smoke Test | ✅ 完成 | 512 样本 / 256 步；loss 1.99~2.77；trainable **49.558M**；RC=0 |
| 6 | **完整 VLM Pretrain** | ✅ **完成** | 2 epochs × 79,669 步；RC=0；产物 409MB（见 §4） |
| 7 | **完整 VLM SFT** | ⏳ **受阻** | 速度异常，见 §5。中途快照 <5%，**不算完成** |
| 8 | Evaluation | ⬜ 未开始 | |
| 9 | Baseline | ⬜ 未开始 | |
| 10 | Modification | ⬜ 未开始 | 候选方向见 §6 |
| 11 | Controlled Experiment | ⬜ 未开始 | |
| 12 | Ablation | ⬜ 未开始 | 候选方向见 §6 |
| 13 | 项目材料整理 | ⬜ 未开始 | |

---

## 2. 模型规格（实测）

| 模块 | 参数量 | 文件大小 | 备注 |
|------|--------|----------|------|
| LLM 主干 `model` | 198.42M | — | embedding + 8 层 Transformer + MoE 专家 |
| 输出头 `lm_head` | 4.92M | — | 与 embedding 权重共享 |
| 视觉投影 `vision_proj` | 1.18M | — | VLM 独有（跨模态对齐） |
| **LLM 侧小计** | **204.51M** | **390.1MB** | fp16，训练保存的部分 |
| 视觉编码器 SigLIP2 | **94.55M** | 180.4MB | fp16，**训练时冻结**，单独文件 |
| **完整 VLM 合计** | **≈299.06M** | ~570MB | |

**三个口径（易混淆，务必区分）**：

- 总参数（LLM 主干）：**204.51M** ← state_dict 实测
- 脚本自报：**199.60M-A65.12M** ← `get_model_params()` 输出（差值 ≈ lm_head，因权重共享未重复计数）
- **激活参数：65.12M** ← MoE 每 token 只激活 4 专家中的 1 个，即 README 的 `200M-A65M`

**各阶段可训练参数**：

| 阶段 | freeze 策略 | 可训练 | 占比 |
|------|-------------|--------|------|
| Pretrain | `freeze_llm=2`（仅 projector） | **1.183M** | 0.4% |
| SFT | `freeze_llm=1`（projector + LLM 首尾层） | **49.558M** | 16.6% |

---

## 3. 环境（已验证）

```
AutoDL 单卡 RTX 4090 24GB，驱动 595.71.05，compute_cap 8.9
conda env: minimind-v  (Python 3.10.21 @ /root/miniconda3/envs/minimind-v)
torch 2.6.0+cu126  (cuda_available=True, bf16_supported=True)
transformers 4.57.6 / datasets 3.6.0 / pyarrow 23.0.0 / numpy 1.26.4 / Pillow 11.3.0
gradio 5.49.1 / swanlab 0.6.12 / modelscope 1.40.1
宿主 208 核，cgroup CPU 配额 = 20 核；有卡模式内存 90GB（无卡仅 2GB）
系统盘 / 30GB；数据盘 /root/autodl-tmp 50GB
```

**torch 版本选择依据**：官方 requirements 注释参考 `torch==2.6.0`；cp310 在该版本线有 cu126 wheel；驱动向下兼容；RTX 4090 (sm_89) 在支持范围。torchvision 未安装（全仓库无 import，仅有注释行）。

**⚠️ 环境变量必须设置（曾踩坑撑爆系统盘）**：

```bash
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export HF_DATASETS_CACHE=/root/autodl-tmp/.cache/huggingface/datasets
export HF_HUB_CACHE=/root/autodl-tmp/.cache/huggingface/hub
export MODELSCOPE_CACHE=/root/autodl-tmp/.cache/modelscope
export PIP_CACHE_DIR=/root/autodl-tmp/.cache/pip
```

> 未设置时 HF 缓存默认写系统盘 `/root/.cache/huggingface`，生成 SFT 数据集缓存时写满 30GB 系统盘，报 `No space left on device (Errno 28)`，数据集在 191 万条处崩溃。系统盘现存 9.5GB 这份垃圾，可安全删除。

---

## 4. 产物清单

```
/root/autodl-tmp/minimind-v/
├── out/
│   ├── llm_768_moe.pth                        # 官方 LLM 主干 388MB（SHA256 通过）
│   ├── official_pretrain_vlm_768_moe.pth      # 官方 pretrain 权重（改名备份）
│   ├── sft_vlm_768_moe.pth                    # 官方 SFT 权重 387MB（SHA256 通过）
│   └── pretrain_vlm_768_moe.pth               # 软链 → 我们自己训练的 pretrain
├── model/siglip2-base-p32-256-ve/             # 视觉编码器 180MB（SHA256 通过）
├── dataset/
│   ├── pretrain_i2t.parquet                   # 4.33GB / 1,274,698 行
│   ├── sft_i2t.parquet                        # 4.93GB / 2,904,511 行
│   └── eval_images/                           # 13 张测试图
├── runs/
│   ├── pretrain_full/pretrain_vlm_768_moe.pth # ★ 完整 Pretrain 产物 409MB（15:11 完成）
│   └── sft_full/sft_vlm_768_moe.pth           # ⚠️ SFT 中途快照（<5%，不可用）
└── checkpoints/                                # resume（路径硬编码 ../checkpoints）
```

**Pretrain 最终数据**：2 epochs × 79,669 步，RC=0，完成于 2026-09-23 15:11:58，耗时约 5.5 小时（含一次实例关机中断 + `--from_resume 1` 恢复）。定性验证：3 张图识别正常。

---

## 5. ★ 核心未解问题：SFT 速度异常

### 5.1 官方声明（README 原文）
> 单张 NVIDIA 3090 上，SFT 跑完 `1 epoch` 实测约 2 小时，dense 与 MoE 用时接近…按云上 3090 约 1.5 元/小时的行情，SFT 单轮成本落在 3 元上下。

**换算**：2,904,511 行 / batch 4 = 726,128 步，2 小时 → 需要 **100.8 步/秒 ≈ 403 samples/s**。

### 5.2 本机 batch 扫描实测（`/root/step16_bench.sh` → `step16_batch_bench.log`）

| batch | 纯数据加载 | 训练步速 | ms/step | **samples/s** | 显存 | 2 epochs 预估 |
|-------|-----------|---------|---------|--------------|------|--------------|
| **4（官方默认）** | 146.4 b/s | **6.72 b/s** | 148.8 | **26.9** | 3.0GB | **60 h** |
| 8 | 126.5 | 8.44 | 118.5 | 67.5 | 4.2GB | 23.9 h |
| 16 | 115.1 | 7.37 | 135.7 | 117.9 | 6.4GB | 13.7 h |
| 32 | 78.3 | 3.97 | 251.8 | **127.1** | 11.0GB | **12.7 h** |
| 64 | 50.1 | 1.78 | 562.7 | 113.7 | 20.1GB | 14.2 h |

### 5.3 torch.compile 实测（`step17_compile_bench.log`）

```
[eager]       bs=16 →  8.80 steps/s | 140.9 samples/s | 114 ms/step | 6.4GB | 300W, 54%
[compile]     bs=16 →  9.18 steps/s | 146.8 samples/s | 109 ms/step | 5.2GB | 250W, 34%
[compile]     bs=32 →  7.24 steps/s | 231.8 samples/s | 138 ms/step | 8.5GB | 318W, 10%
[compile-RO]  bs=16 →  5.02 steps/s |  80.3 samples/s | 199 ms/step | 5.5GB | 170W, 18%  ← 更慢
```

**当前最佳 = torch.compile + batch 32 = 231.8 samples/s，仍只有官方基准的 57%。**

### 5.4 已排除的假设

| 假设 | 结论 | 依据 |
|------|------|------|
| 数据加载瓶颈 | ❌ 排除 | 加载能力是训练消耗的 15~20 倍 |
| DataLoader workers 不足 | ❌ 排除 | 10 workers 仅占约 2.1 核，增加无效 |
| GPU 被其他进程占用 | ⚠️ 曾存在，已修复 | 首次 SFT 崩溃后进程僵死 53 分钟、白占 2GB 显存，已 kill。清理后仅 8.2→9.33 步/s |
| 显存不足 | ❌ 排除 | batch 4 仅用 3GB / 24GB |
| GPU 算力打满 | ❌ 排除 | 利用率 31~64%，功耗 150~280W（上限 450W） |

### 5.5 最强假设：主进程 CPU 侧 kernel launch bound

**指向代码**：`model/model_minimind.py:163` `MOEFeedForward.forward`

```python
for i, expert in enumerate(self.experts):          # ← Python 循环遍历专家
    mask = (topk_idx == i)
    if mask.any():
        token_idx = mask.any(dim=-1).nonzero().flatten()
        weight = topk_weight[mask].view(-1, 1)
        y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
```

**支撑证据**：
- 逐专家 mask / 索引 / `index_add_` → 大量小 kernel 与启动开销
- 主进程 CPU 占用约 **5.2 核**（配额 20 核），数据加载仅 2.1 核
- **Pretrain(batch16) 与 SFT(batch4) 步速几乎相同（9.8 vs 9.3 步/s）**，batch 差 4 倍而步速不变 → 强烈指向每步固定开销
- README 作者自述："原生训练时带来的 kernel 启停和调度开销会急剧变重…得靠支持 MoE kernel-fused 的算子库来优化"
- `torch.compile` 仅 +4%，说明编译器未能有效融合该数据依赖循环

### 5.6 待验证的优化方向（按性价比排序）

1. **【低成本高收益】检查 optimizer 是否误含冻结参数**
   `train_sft_vlm.py:146` 用 `optim.AdamW(model.parameters(), lr=...)`
   对比 `train_pretrain_vlm.py:146` 用 `optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), ...)`
   → SFT 可能把冻结参数也塞进 AdamW，带来额外开销。**先验证这一点。**

2. **MoE 前向向量化 / 分组 GEMM（grouped GEMM）**：消除逐专家 Python 循环
3. `torch.compile(mode='max-autotune')` 或 `fullgraph=True`
4. **验证瓶颈归属**：跑 `--use_moe 0`（dense）对比速度（需 dense 版 `llm_768.pth`，当前可能未下载）
5. `py-spy record --native` 采样主进程栈（此前因 ptrace 权限失败，可调 `ptrace_scope`）
6. 检查 Triton kernel 编译 / autotune 开销

> 已有 `step18_bench_profiler.log`、`step18_trace.json`、`step19_diag.log` 三个日志（非本次会话产出，疑为接手方所留），**接手时请先查看**，避免重复劳动。

---

## 6. Modification / Ablation 候选方向

| 方向 | 说明 | 可量化指标 |
|------|------|-----------|
| **MoE 前向优化** | 逐专家循环 → 向量化 / grouped GEMM / 融合算子 | steps/s、GPU util、显存、loss 曲线一致性 |
| **MoE 结构消融** | `num_experts`、`num_experts_per_tok`(top-k)、`moe_intermediate_size` | 吞吐 vs 效果权衡 |
| **freeze 策略消融** | `freeze_llm` 0 / 1 / 2 对下游能力影响 | 各 benchmark 分数 |
| **batch size 对收敛影响** | 4 / 16 / 32（含 LR 缩放） | loss 曲线、最终指标 |
| **视觉 token 数** | `image_token_len` 64 的增减 | 效果 vs 训练成本 |
| **dense vs MoE 对比** | 同为 65M 激活参数 | 效果 / 吞吐 |

---

## 7. 成本与决策记录

| 项 | 数值 |
|----|------|
| 4090 租金 | 约 2 元/小时 |
| 无卡模式 | 约 0.1 元/小时（非 GPU 任务务必切换） |
| 完整 Pretrain 实耗 | 约 5.5 小时 ≈ 11 元 |
| SFT（batch 4，官方超参） | **60 小时 ≈ 120 元** ← 不可接受 |
| SFT（batch 32 + compile） | 12.7 小时 ≈ 25 元 |
| SFT（batch 16 + compile） | 13.7 小时 ≈ 27 元 |

**待用户决策**：是否偏离官方 batch=4 超参以换取 4~5 倍成本节省（须记录为"工程优化"并注明偏离）。

---

## 8. 磁盘状况与清理建议

```
系统盘 /          18G / 30G   (58%)
数据盘 autodl-tmp 42G / 50G   (84%)  ← 仅剩 8.5G，需关注
```

| 位置 | 大小 | 建议 |
|------|------|------|
| `/root/.cache/huggingface`（系统盘） | **9.5G** | ❌ **可直接删**（早期误写的垃圾，曾撑爆系统盘） |
| `.cache/huggingface`（数据盘） | **21G** | ⚠️ SFT 数据集缓存，删后重建约 4 分钟 |
| `smoke/`（数据盘） | 2.7G | ❌ 可删（冒烟测试已完成） |
| `.cache/pip` | ~3G | ❌ 可删（`pip cache purge`） |
| `checkpoints/*_resume.pth`（Pretrain 相关） | ~1.6G | ❌ 可删（Pretrain 已完工） |

**绝不能删**：`dataset/*.parquet`（9.3G）、`out/*.pth`（1.2G）、`runs/pretrain_full/`（409M）、`model/siglip2-*/`（180M）、`minimind-v-research/`、`logs/`

---

## 9. 仓库已知陷阱（务必了解）

1. **`init_vlm_model` 硬编码加载路径**
   `trainer/trainer_utils.py:72` 用 `save_dir` 拼权重路径，但默认参数 `save_dir='../out'`，且 `train_sft_vlm.py:142` **调用时未传 save_dir**。
   → **CLI 的 `--save_dir` 只影响保存，不影响加载基础权重**
   → 隔离实验必须自建运行根；**目录软链作 cwd 会被内核解析为物理路径**，`../out` 仍指向真实仓库 → 正确做法是"真实目录 + 文件级软链"

2. **`eval_vlm.py` 的 save_dir 拼接**：代码为 `f'./{args.save_dir}/...'`，传绝对路径会变成 `.//root/...` 报 FileNotFoundError → **必须传相对路径**

3. **训练脚本必须从 `trainer/` 目录运行**（依赖 `../model`、`../dataset`、`../out`、`../checkpoints`）

4. **resume 检查点路径硬编码** `../checkpoints/`，与 `--save_dir` 无关

5. **保存的 `.pth` 会剥离 `vision_encoder.` 前缀权重**（`clean_state_dict`），推理时视觉编码器从 `model/siglip2-base-p32-256-ve/` 单独加载

6. **checkpoint 内容**：`{epoch, step, model, optimizer, scaler, wandb_id, world_size}`，model 含全部 385 个 tensor（299.07M 参数，含冻结的 SigLIP2）

---

## 10. 运维要点

- **实例关机 = 训练进程终止**，但 `/root` 与 `/root/autodl-tmp` 数据保留
- 长训练三件套：① tmux 托管 ② 合理 `--save_interval` ③ 中断后 `--from_resume 1`
- **kill tmux 会话不会杀进程树** → 崩溃的进程可能僵死并占显存，务必 `pkill -9 -f train_xxx`
- **SSH**：免密已配（`~/.ssh/config` 的 `autodl-minimind-v`）；DNS 走代理网段，加 `-4` 强制 IPv4 更稳；单条命令输出 >2KB 会卡死，读日志用 `head -c N | tail -c M` 分块
- 切换有卡/无卡模式后**实例会重启、SSH 端口可能变化**，需重新确认连接串

---

## 11. 训练命令速查

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate minimind-v
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export HF_DATASETS_CACHE=/root/autodl-tmp/.cache/huggingface/datasets
cd /root/autodl-tmp/minimind-v/trainer

# ---- Pretrain（已完成，存档）----
python -u train_pretrain_vlm.py --use_moe 1 --epochs 2 --batch_size 16 \
  --accumulation_steps 1 --learning_rate 4e-4 --max_seq_len 450 --freeze_llm 2 \
  --from_weight llm --data_path ../dataset/pretrain_i2t.parquet \
  --save_dir /root/autodl-tmp/minimind-v/runs/pretrain_full \
  --save_weight pretrain_vlm --save_interval 5000 --log_interval 100 \
  --num_workers 10 --device cuda:0
# 续训加 --from_resume 1

# ---- SFT（待优化后重跑）----
python -u train_sft_vlm.py --use_moe 1 --epochs 2 --batch_size 4 \
  --accumulation_steps 1 --learning_rate 5e-6 --max_seq_len 768 --freeze_llm 1 \
  --from_weight pretrain_vlm --data_path ../dataset/sft_i2t.parquet \
  --save_dir /root/autodl-tmp/minimind-v/runs/sft_full \
  --save_weight sft_vlm --save_interval 5000 --log_interval 200 \
  --num_workers 10 --device cuda:0

# ---- 推理验证 ----
cd /root/autodl-tmp/minimind-v
python -u eval_vlm.py --load_from model --weight sft_vlm --use_moe 1 \
  --save_dir runs/sft_full --max_new_tokens 256 --device cuda:0   # save_dir 必须相对路径
```

---

## 12. 参考文件索引

| 文件 | 内容 |
|------|------|
| `minimind-v-research/PROJECT_LOG.md` | 已完成里程碑、工程问题、中断恢复记录 |
| `minimind-v-research/ENV_INVENTORY.md` | 环境盘点 |
| `logs/step16_batch_bench.log` | batch 扫描原始数据 |
| `logs/step17_compile_bench.log` | torch.compile 对比数据 |
| `logs/step18_bench_profiler.log` / `step18_trace.json` / `step19_diag.log` | 接手方所留，待查看 |
| `logs/step12c_pretrain_resume.log` | 完整 Pretrain 训练日志 |
| `logs/step15b_sft_full.log` | SFT 首次完整尝试（含磁盘写满错误） |
| `/root/step*.sh` | 各阶段可复用脚本 |
