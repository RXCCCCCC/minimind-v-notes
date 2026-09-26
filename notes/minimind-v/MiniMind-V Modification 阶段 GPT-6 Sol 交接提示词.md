# MiniMind-V Modification 阶段：GPT-6 Sol 交接提示词

你现在接手 **MiniMind-V Research** 项目的 **Modification → Controlled Experiment → Result Analysis** 闭环。
请先完整阅读本提示词，再执行「第一阶段动作」（见文末），**不要直接开始改代码**。

---

## 0. 项目目标

把 MiniMind-V（MoE，约 200M 总参 / 65M 激活）做成一整套可对外展示的科研项目：完整复现 → 训练性能诊断 → 执行效率 Modification → 受控实验 → Ablation → 最终 README / 简历材料。

最终主线：

```
Official MiniMind-V
    ↓  Full MoE Reproduction        ✅ 已完成
    ↓  Training Profiling           ✅ 已完成
    ↓  Launch-bound Diagnosis       ✅ 已完成
    ↓  High-throughput Training     ✅ 已完成
    ↓  Research Modification        ← 你从这里开始
    ↓  Controlled Experiment
    ↓  Ablation
    ↓  Final Results / README
```

唯一 GitHub 仓库：**https://github.com/RXCCCCCC/minimind-v-research**
（由 `minimind-v-notes` 原仓库直接改名而来，保留全部 history；不要再新建第二个仓库，不要建独立源码 fork。）

---

## 1. 当前真实进度（以此为准，不要被历史文档误导）

已完成：

- 环境复现（AutoDL RTX 4090 24GB ×1；conda env `minimind-v`；Python 3.10.21；torch 2.6.0+cu126；31/31 依赖对齐官方 requirements）
- 官方模型 inference（13 张测试图）
- Dataset 检查（pretrain 1,274,698 行 / sft 2,904,511 行）
- Pretrain smoke / SFT smoke
- **MoE Pretrain 2 epochs**（79,669 步/epoch，RC=0）
- **MoE SFT 2 epochs**（45,383 步/epoch；batch 64 + torch.compile + max_seq_len 768；最终 loss **1.9240**）
  - Epoch 2 曾在 ~step 44,800 卡死（疑似 dataloader worker 死锁，无 stderr），后用 `--from_resume 1` 从 step 40,000 续训到 45,383 完成
- 自训 vs 官方 Evaluation（13 图；自训 ≈108 tokens/s vs 官方 ≈101；均正常生成，幻觉水平相当）
- SFT profiler / 瓶颈定位（launch-bound，见 §4）
- High-throughput 配置（batch scaling + torch.compile）

待你的工作：Modification / Controlled Experiment / Ablation / 最终 README。

关键产物（AutoDL，**不得覆盖**）：

```
/root/autodl-tmp/minimind-v/runs/pretrain_full/pretrain_vlm_768_moe.pth         # 自训 Pretrain
/root/autodl-tmp/minimind-v/runs/sft_full/sft_vlm_768_moe.pth                  # 自训 SFT（正式，409MB）
/root/autodl-tmp/minimind-v/checkpoints/sft_vlm_768_moe_resume.pth             # 最终 resume（1.57GB，含 optimizer）
/root/autodl-tmp/minimind-v/out/sft_vlm_768_moe.pth                            # 官方发布权重（质量参照）
/root/autodl-tmp/minimind-v/out/llm_768_moe.pth                                # 官方 LLM 主干
```

已有文档（先读）：

- `notes/minimind-v/PROJECT_LOG.md`（里程碑 + 工程问题 + profiler 结论）
- `notes/minimind-v/EXPERIMENT_PLAN.md`（含 Baseline/Modification/Controlled/Ablation 草案与预算）
- `notes/minimind-v/patches/M1-moe-vectorized-dispatch.md`（**仅是草案，非既定方案**）
- `results/minimind-v/2026-09-25-sft-vlm-moe-bs64-2epoch/`（README / metrics.json / train.log / train_resume.log / eval.log / EVAL_SUMMARY.md / loss_curve.png）
- `notes/minimind-v/ENV_INVENTORY.md`、`TRAINING_PLAN.md`

---

## 2. 代码与 Git 策略

- upstream：`jingyaogong/minimind-v`，真正用于复现的 commit 固定为
  **`dacc68788998056476b21a4ca11325bdb277948c`**（不要因 upstream 更新就切最新 master）。
- 单仓库策略：后续只有 `RXCCCCCC/minimind-v-research` 一个仓库；不用 git submodule；上游源码纳入方式由你审查后决定（优先 `git subtree` 或等价的可追溯方案），目标路径 `src/minimind-v/`。
- Baseline 与 Modification 必须隔离：建立 `baseline/minimind-v-dacc687` 形式的 tag/branch 与 `research/*` 分支；必要时用 `git worktree` 同时保留 baseline 与修改版工作树。
- 已有的 `runs/sft_full/`、`runs/pretrain_full/`、checkpoints 为正式产物，**任何实验不得覆盖**；实验一律写 `runs/experiments/<exp-id>/`。
- 不提交大文件：`*.pth`、`*.safetensors`、`*.parquet`、HF/ModelScope/PIP cache、超大 profiler trace。
- 提交信息沿用仓库习惯：`docs:` / `chore:`（notes 仓库）；源码改动另按 upstream 风格。

---

## 3. 环境与访问

- AutoDL 单卡 RTX 4090 24GB；cgroup 内存有卡模式 ~90GB；**无卡模式仅约 2GB**（连加载 checkpoint 都会 OOM，不得在无卡跑任何模型加载）。
- 项目根：`/root/autodl-tmp/minimind-v`；训练脚本在 `trainer/` 下运行；conda：`source /root/miniconda3/etc/profile.d/conda.sh && conda activate minimind-v`。
- 必须显式设置缓存到数据盘：
  `export HF_HOME=/root/autodl-tmp/.cache/huggingface`
  `export HF_DATASETS_CACHE=/root/autodl-tmp/.cache/huggingface/datasets`
- SSH（已免密）：`ssh autodl-minimind-v`。**严禁**把私钥 / API token / 密码写进任何文件或提示词。
- 当前服务器处于**无卡模式**。需要 GPU 的步骤必须：先说明「需要做什么、预计多久、预计花费」→ 请用户切到有卡 → 再执行；完成后若无后续 GPU 任务，提醒用户切回无卡。
- 若 `ssh autodl-minimind-v` 连接超时：大概率是用户本机代理/DNS（`connect.cqa1.seetacloud.com` 被解析为 198.18.x 假 IP）导致，请让用户检查/重启代理，或把 `seetacloud.com` 设为直连。

---

## 4. 必须写进报告的 profiler 证据（不要重新猜）

主瓶颈：**launch-bound**，证据：

- ~**4,038 个 CUDA kernels / step**；
- GPU 实际 busy ≈ **29 ms / step**；
- CPU / wall ≈ **109 ms / step**（大量时间在 kernel launch）；
- GPU utilization 明显不足（eager 时 28~38%，compile 后 84~99%）；
- 模型小（激活 ~65M），tiny kernel 启动开销占比过高。

已排除的主因（有受控实验）：

- DataLoader 不是主瓶颈；
- Vision Encoder 不是主瓶颈（vision ON 77.1ms vs OFF 79.8ms）；
- `count_vision_proj` **不是**瓶颈（不得再把它当作主要优化方向）；
- SFT optimizer 未过滤冻结参数不是主瓶颈；
- 9 月两个 upstream fix（top-1 router / fp16 attn）不是性能回归（HEAD 反而快 8%）。

当前 MoE 实现（`model/model_minimind.py` 的 `MOEFeedForward.forward`）：逐 expert Python 循环 + `mask / nonzero / index_select / index_add_`，4 experts × 8 layers，且 autocast 带来大量 dtype 转换——被认为是 launch-bound 的主要来源之一。

已验证的工程缓解：`torch.compile` + 增大 batch（bs64 峰值 ≈251.8 samples/s；bs96 OOM）。

Batch scaling 实测（compile=1，300 步/组）：bs4 41.3 / bs16 138.4 / bs32 211.7 / bs48 234.0 / **bs64 251.8** / bs80 243.5 samples/s。

---

## 5. 你的任务：完成一次真正的 Modification 闭环

1. 阅读整个仓库与上述证据；
2. 核对 AutoDL 实际源码、git 状态、训练产物（以实际为准）；
3. **给出你自己的技术判断**（见文末第一阶段动作）；
4. 实现选定 Modification（源码 + diff review）；
5. 数值正确性验证；
6. 500-step A/B benchmark + profiler；
7. 若短实验成立：5,000-step Controlled Experiment + Evaluation；
8. 分析结果，判断是否值得完整 2 epochs 正式复跑；
9. 设计必要的 Ablation。

### 5.1 Modification 不绑定 M1

现有 M1（MoE 逐专家循环 → token 排序/分桶 + 连续切片）只是候选草案。你必须审查它，并可以改进 / 否决 / 替换。若坚持 M1，必须逐条评估：

`argsort` 额外开销、`bincount/cumsum` kernel、`int(tensor)` 引发的 device→host 同步、torch.compile graph break、仍然存在的 Python expert loop、`index_add_` 是否仍是瓶颈、autograd 语义变化、Top-1 straight-through 梯度是否被破坏、unused expert / DDP 语义、checkpoint / state_dict 兼容性。

若提出新方向，必须先说明（九问）：

1. 当前瓶颈是什么；2. 新方案为什么直接针对瓶颈；3. 相比 M1 为什么更合理；4. 修改哪些代码；5. 是否改变数学语义；6. 是否保持 checkpoint 兼容；7. 如何验证正确性；8. 如何做 Controlled Experiment；9. 预计 GPU 时间 / 成本。

优先方向：**保持模型数学定义基本不变 + 改进实际执行效率**。
若准备修改模型结构 / routing / 训练目标 / loss / expert 数学定义 / 数据集，必须先停下来向用户说明并等待确认。

### 5.2 数值正确性标准（第一层）

固定 seed、可控输入、FP32 下验证并**报告实际误差数值**（不能只写“通过”）：

- forward 输出接近；loss 接近；backward 梯度接近；
- routing 语义一致；Top-1 straight-through 梯度语义正确；
- 无 NaN / Inf。

### 5.3 Controlled Experiment 规则（第二层）

固定：同一 Dataset、同一 `pretrain_vlm` 起点、同一 seed、同一 batch、同一 `max_seq_len`、同一 lr 与 scheduler、相同步数、同一 compile 设置（除非对比变量就是 compile）。

- 第一轮：500-step A/B + profiler；
- 成功门槛：**稳定 samples/s 相对 Baseline 提升 ≥ 10%**；
- 第二轮：5,000-step；要求 loss 趋势正常、无 NaN/Inf、13 图 Evaluation 无明显质量下降；
- 暂不要求 multi-seed；若只有几个百分点的边界提升，再考虑 multi-seed。

### 5.4 科研纪律

一次只动一个变量：一个明确假设 → 一个主 Modification → 一个严格 A/B → 一个结论 → 再进入下一轮。
**负结果必须保留并记录**（kernel 数下降但 wall time 未降、compile 变慢、+3% 提升、graph break、显存暴涨等）。

---

## 6. GPU 预算与止损

- 短实验阶段总预算 **≤ ¥10**（数值验证 / benchmark / profiler / 500-step / 必要时的 5,000-step / 短 Evaluation）。
- **不包含**完整 2 epochs 正式复跑。
- 未通过短实验验证的方案，禁止直接跑数小时；
- 明显失败的方向立即停止；不为了“做出正结果”继续烧 GPU。

---

## 7. 分工与长训练交接

- **你（GPT-6 Sol）**：技术判断、源码设计/实现、diff review、数值验证、benchmark、profiler、500-step A/B、5,000-step 受控实验、Evaluation、Ablation 设计、结果分析。
- **训练监控 Agent（当前 Codex）**：正式长训练的启动、tmux 托管、GPU/CPU/显存监控、loss/日志监控、卡死检测、watchdog、checkpoint、`--from_resume 1`、成本记录、训练后产物核验、结果同步回仓库。
- 同一时间**禁止两个 Agent 同时**：改同一工作目录、启动 GPU 任务、覆盖同一实验目录（单写入者原则）。
- 若需要完整 2 epochs 正式复跑：由你给出最终代码 + commit + 训练命令 + 预期指标 + 注意事项，然后交给训练监控 Agent 执行与监控。

---

## 8. 交付物

每个实验目录（AutoDL `runs/experiments/<exp-id>/`，并回填仓库 `results/minimind-v/<run-name>/`）至少包含：

- `README.md`（目标 / 环境 / 数据 / 命令 / 指标 / 结论 / 产物）
- `metrics.json`（吞吐、samples/s、GPU util、VRAM、kernel 数、loss 等）
- 原始日志（train/eval，轻量）
- `curves/*.png`（loss / throughput）
- 代码 diff 或 commit 链接
- 负结果同样归档

完成后 push 到 `RXCCCCCC/minimind-v-research`，并在 `PROJECT_LOG.md` 追加一节。

---

## 9. 禁止事项

- 在无卡模式加载模型 / 跑训练；
- 直接照 M1 开改而不先给技术判断；
- 一次同时修改多个变量后归因于单一因素；
- 覆盖 `runs/sft_full/`、`runs/pretrain_full/`、官方 `out/*.pth`；
- 把大文件 / 凭据 / token / 私钥提交进仓库；
- 盲目升级 upstream 到最新 master；
- 未经用户确认就改动模型数学语义（结构 / routing / loss / 数据集）。

---

## 10. 第一阶段动作（现在就要做，且只做这些）

1. 阅读仓库：`PROJECT_LOG.md`、`EXPERIMENT_PLAN.md`、`EVAL_SUMMARY.md`、`patches/M1-moe-vectorized-dispatch.md`、`ENV_INVENTORY.md`；
2. 通过 `ssh autodl-minimind-v` 核对 AutoDL 上的源码状态、git diff、正式产物、profiler 记录（如 SSH 不通，先告知用户并等待）；
3. 输出你的技术判断与实验计划，包含：候选 Modification 排序、每个方案的九问简答、预计 GPU 时间与花费、止损条件；
4. **等待用户确认后**再开始改代码；若需 GPU，先请用户切有卡。
