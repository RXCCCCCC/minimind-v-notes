# MiniMind-V 复现项目日志

> 自动生成/维护。位置：/root/autodl-tmp/minimind-v-research/PROJECT_LOG.md

## 0. 目标
完整复现 MiniMind-V (MoE 200M-A65M) 的 Pretrain + SFT，并产出可写进科研简历的
Modification / Controlled Experiment / Ablation 结果。

## 1. 环境（详见 ENV_INVENTORY.md）
- AutoDL 单卡 RTX 4090 24GB，驱动 595.71.05
- conda env `minimind-v`，Python 3.10.21
- torch 2.6.0+cu126（官方 requirements 注释参考版本；wheel 自带 CUDA runtime）
- transformers 4.57.6 / datasets 3.6.0 / pyarrow 23.0.0 / numpy 1.26.4
- 全部依赖来自官方 requirements.txt（31 个包全部版本对齐）
- 缓存重定向：HF_HOME / MODELSCOPE_CACHE / PIP_CACHE_DIR → /root/autodl-tmp/.cache

## 2. 已完成的里程碑
| 阶段 | 状态 | 证据 |
|---|---|---|
| 环境复现 | DONE | pip check 无冲突；31/31 包版本匹配 |
| 官方模型 inference | DONE | eval_vlm.py 13 张图全过，108.9 tokens/s（4090） |
| Dataset 检查 | DONE | pretrain 1,274,698 行 / sft 2,904,511 行；schema=(conversations, image_bytes) |
| Pretrain smoke | DONE | 512 样本，256 步，loss 2.86→2.59，trainable 1.183M |
| SFT smoke | DONE | 512 样本，256 步，loss 1.99~2.77，trainable 49.558M |
| 完整 Pretrain | DONE | MoE，2 epochs × 79,669 步，RC=0；产物 runs/pretrain_full/pretrain_vlm_768_moe.pth（out/pretrain_vlm_768_moe.pth 为软链；官方备份 official_pretrain_vlm_768_moe.pth） |
| 完整 SFT | DONE | MoE，2 epochs × 45,383 步，batch 64 + compile 1 + seq 768；最终 loss 1.9240；Epoch2 曾在 ~44,800 卡死，以 --from_resume 1 从 step 40,000 续训完成；产物 runs/sft_full/sft_vlm_768_moe.pth（409MB） |
| Evaluation | DONE | 13 图；自训 ≈108 t/s vs 官方 ≈101 t/s（剔除预热）；见 results/minimind-v/2026-09-25-sft-vlm-moe-bs64-2epoch/EVAL_SUMMARY.md |
| Profiling / 瓶颈定位 | DONE | ~4,038 kernels/step，GPU busy ≈29ms/step、wall ≈109ms/step → launch-bound；已排除 DataLoader / Vision Encoder / count_vision_proj / optimizer filter / 9 月 fix |
| High-throughput 配置 | DONE | batch scaling 基准：bs64 峰值 ≈251.8 samples/s（bs96 OOM）；正式 SFT 采用 bs64 + torch.compile + seq768 |
| Baseline / Modification / Controlled / Ablation | IN PROGRESS | Baseline 完成；Modification READY（M1 草案已备，非既定方案）；Controlled Experiment / Ablation TODO |

## 3. 复现中发现的工程问题（可写进项目文档）

### 3.1 init_vlm_model 硬编码 ../out（重要）
`trainer/trainer_utils.py:72`:
    weight_path = f'{save_dir}/{from_weight}_{hidden_size}{moe_suffix}.pth'
但调用处 `init_vlm_model(vlm_config, from_weight=..., device=..., freeze_llm=...)` **没有传 save_dir**，
函数默认 `save_dir='../out'` 是相对 cwd 的硬编码路径。
=> CLI 的 `--save_dir` 只影响"保存"，不影响"加载基础权重"。
=> 做隔离实验（不污染官方 out/）必须自建运行根。
=> 注意：symlink 目录做 cwd 会被内核解析成物理路径，`../out` 仍指向真实仓库；
   正确做法是「真实目录 + 文件级软链」。

### 3.2 MoE 前向是 Python 循环（性能瓶颈 / 论文级改造点）
`model/model_minimind.py:163`:
    for i, expert in enumerate(self.experts):
        mask = (topk_idx == i)
        if mask.any():
            token_idx = mask.any(dim=-1).nonzero().flatten()
            y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
- 逐专家 mask + 索引 + index_add_，产生大量小 kernel 与启动开销
- 实测：4090 上 GPU 利用率长期 43~64%（均值~52%），功耗 230~280W（上限 450W）
- 即"训练受 kernel 调度限制而非算力限制"——与 README 作者自述一致
- **改造方向（Ablation 候选）**：
  a) token 分桶后批量 GEMM（grouped GEMM），或用 torch.compile / Triton 融合
  b) 换成 TransformerEngine / DeepEP 等融合算子
  c) 对比指标：steps/s、GPU util、显存、loss 曲线一致性
  d) 消融：num_experts、top-k、moe_intermediate_size 对吞吐与效果的影响

### 3.3 其他
- 镜像精简：无 tmux（已 apt 安装 3.2a）、内网 Python 需 conda activate 后才可见
- SSH 长命令输出过大（>~2KB）会触发通道卡顿，读取日志建议 `head -c N | tail -c M` 分块
- 无卡模式 cgroup 内存上限 2GB（大模型 CPU 加载必 OOM/137）；有卡模式恢复 90GB
- GitHub 直连 clone 仅 8KB/s，需用 gh-proxy.com 加速（2.1MB/s）

## 4. 关键超参与配置（官方默认，未修改）
### Pretrain
    python train_pretrain_vlm.py --use_moe 1 --epochs 2 --batch_size 16 \
      --learning_rate 4e-4 --max_seq_len 450 --freeze_llm 2 \
      --from_weight llm --data_path ../dataset/pretrain_i2t.parquet \
      --num_workers 10 --save_interval 10000
- freeze_llm=2：仅训 vision_proj（1.183M / 199.60M-A65.12M）
- 数据：1,274,698 行，79,669 步/epoch

### SFT
    python train_sft_vlm.py --use_moe 1 --epochs 2 --batch_size 4 \
      --learning_rate 5e-6 --max_seq_len 768 --freeze_llm 1 \
      --from_weight pretrain_vlm --data_path ../dataset/sft_i2t.parquet \
      --num_workers 10 --save_interval 10000
- freeze_llm=1：训 vision_proj + LLM 首尾层（49.558M）
- 数据：2,904,511 行，726,128 步/epoch

## 5. 产物清单
- 官方权重：out/{llm,pretrain_vlm,sft_vlm}_768_moe.pth（已 SHA256 校验）
- SigLIP2: model/siglip2-base-p32-256-ve/model.safetensors（SHA256 校验通过）
- smoke 产物：/root/autodl-tmp/smoke/out/{smoke_pretrain,smoke_sft}_768_moe.pth
- 完整训练产物：runs/pretrain_full/pretrain_vlm_768_moe.pth（训练中）
- 日志：/root/autodl-tmp/logs/step*.log

---

## 8. 【关键】SFT 性能根因分析（2026-09-24，含 profiler 证据）

> 本节所有结论均基于实测数据，非推测。配置：MoE, batch=4, max_seq=768, freeze_llm=1（官方默认）。

### 8.1 分环节耗时（step18，各环节单独计时）
| 环节 | 耗时 | 占端到端比例 |
|---|---|---|
| DataLoader（含图像解码+预处理） | 14.1 ms/batch | 14% |
| vision_encoder + vision_proj | 5.3 ms/step | 5% |
| count_vision_proj | 6.3 ms/step | 6% |
| LLM forward（含视觉） | 37.2 ms/step | 37% |
| forward + backward | 72.0 ms/step | 71% |
| optimizer.step | ~2.4 ms/step（profiler）/ 0.2ms（缺陷测量） | ~2% |
| **端到端（500 步实测）** | **101.8 ms/step → 9.82 step/s → 39.3 samples/s** | 100% |

### 8.2 决定性对照实验（step19）
| 实验 | 结果 | 结论 |
|---|---|---|
| vision ON vs OFF（bs=4） | 77.1 vs 79.8 ms | **视觉路径贡献 ≈ 0（-3.4%）** |
| batch 4 / 16 / 32 | 87.4 / 109.4 / 226.4 ms | batch 16 吞吐最优 |
| **CUDA kernel 数** | **4012 个/step** | ← 核心问题 |
| GPU 实际忙碌 | 29.0 ms/step | |
| GPU 利用率上限 | **28%** | 严重未饱和 |
| CPU 时间 vs 墙钟 | 109.5 vs 109.4 ms | **100% CPU 阻塞** |

top kernels：elementwise(551 calls) / cutlass-gemm(136) / elementwise(204) / flash-attn-bwd(8) / optimizer multi_tensor(12)

### 8.3 版本回归验证（step21，合成数据受控 A/B，同权重同配置）
| 版本 | 提交日期 | bs=4 | bs=16 | kernels/step | GPU busy |
|---|---|---|---|---|---|
| v_old (740d467) | 2026-08-06 | 85.2 ms | 119.5 ms | 4479 | 30.1 ms |
| v_router (fa36e48) | 2026-09-18 | 82.8 ms | 113.8 ms | 4038 | 29.2 ms |
| v_head (dacc687) | 2026-09-20 | **78.5 ms** | **112.3 ms** | 4038 | 29.3 ms |
**结论：2026-09 的两个 fix（top-1 router gradient / fp16 attention overflow）不构成性能回归，HEAD 反而快 ~8%。**

### 8.4 真正的根因
**这是典型的 launch-bound（核函数启动受限）问题**，不是数据加载、不是视觉路径、不是显存、也不是最近的代码改动：

1. 每步产生 **4038 个 CUDA kernel**，而 GPU 实际只忙 29 ms → 平均每个 kernel 仅 ~7 µs 计算量
2. kernel 启动开销（CPU 侧 ~10-15 µs/个）远超计算时间 → CPU 100% 阻塞，GPU 只到 28-38%
3. 成因：模型极小（激活 65M）+ MoE 逐专家 Python 循环（4 专家 × 8 层，每层 gate/topk/mask/nonzero/index_add_）+ autocast 引发的海量 dtype 转换（`aten::to` 约 892 次/step、`aten::linear` 约 446 次/step）
4. batch 增大只能部分缓解：bs=16 时 samples/s 达 138-142（vs bs=4 的 47-51），但 bs≥32 因显存与调度反而下降

### 8.5 与官方 README "2 小时/epoch" 的差距
| 项 | 数值 |
|---|---|
| 官方换算需求 | 2,904,511 样本 / 7200 s = **403 samples/s** |
| 本机 bs=4（官方默认） | 47-51 samples/s → **16.8 h/epoch**（8.8x 差距） |
| 本机 bs=16（实测最优） | 138-142 samples/s → **5.9 h/epoch**（2.85x 差距） |
| 本机 bs=32 + torch.compile | 232 samples/s → 3.5 h/epoch（1.7x 差距） |

**官方 `2 小时` 声明在当前 master（dacc687）+ RTX 4090 上无法复现。**
可核实的旁证：
- README 中该句最后一次变更在 `9c3343d`（2026-04-19，minimind-3v 发布），此后代码有多次架构/实现改动
- 受控 A/B 显示当前代码并非回归，**该性能特征很可能一直如此**
- 不排除官方使用了不同 batch / 编译选项 / 测量口径（README 未披露 batch 与是否启用 `--use_compile`）

### 8.6 待确认项（未做，避免污染 Baseline）
- `train_sft_vlm.py:146` 用 `optim.AdamW(model.parameters())` 而非 `filter(requires_grad)`，与 pretrain 不一致 —— 保留为 Engineering Optimization 候选
- `count_vision_proj` 的 `.tolist()` 同步与 Python 循环 —— 实测非瓶颈（贡献≈0），保留为候选但优先级低
- MoE 逐专家循环的向量化/融合（grouped GEMM、torch.compile）—— 保留为 Modification 候选

### 8.7 复现脚本与日志
```
/root/step18_bench_profiler.sh   -> step18_bench_profiler.log   （分环节计时 + profiler）
/root/step19_diag.sh             -> step19_diag.log             （视觉开/关 + batch 对照 + kernel 计数）
/root/step20_regression.sh       -> step20_regression.log       （git archive A/B，真实数据；因 HF 缓存重建失败）
/root/step21_ab.sh               -> step21_ab_synth.log         （合成数据受控 A/B，成功）
/root/autodl-tmp/benches/{v_old,v_router,v_head}/               （git archive 导出的三个版本副本）
/root/autodl-tmp/logs/step18_trace.json                         （chrome trace）
```
### 8.8 SFT 断点续训 + 完整完成（2026-09-25）

- **卡死事件**：Epoch 2 在 step 44,800 附近卡住约 10h（日志 01:49 后不再前进；原因未定，疑似 dataloader worker 死锁），切无卡模式后进程被清理。
- **续训方式**：切回有卡后，使用官方 `--from_resume 1` 从 step 40,000 续训；`num_workers=6` + 自愈看门狗（日志 10 分钟无进展自动杀并重启）。
- **结果**：12:00:34 → 12:24:15 一次成功，Epoch 2 45,383/45,383 全部完成；最终 loss 1.9240。
- **产物**：
  - `runs/sft_full/sft_vlm_768_moe.pth`（409,087,288 B，12:24:13）
  - `checkpoints/sft_vlm_768_moe.pth`（409,088,012 B，12:24:14）
  - `checkpoints/sft_vlm_768_moe_resume.pth`（1,573,283,790 B，含优化器，12:24:15）
- **结论**：官方 MoE SFT（2 epochs）已完整复现；下一步 Evaluation / Baseline / Modification。

### 8.9 下一步实验计划（无卡模式准备，2026-09-26）

- 无卡阶段已完成（无需 GPU）：
  - 评估对比总结：`results/minimind-v/2026-09-25-sft-vlm-moe-bs64-2epoch/EVAL_SUMMARY.md`（自训平均 105.98 t/s 含预热，官方 100.31 t/s；剔除预热 ≈108.7 vs ≈100.9）
  - 实验计划（Baseline / Modification / Controlled / Ablation + 有卡执行清单 + 预算）：`notes/minimind-v/EXPERIMENT_PLAN.md`
  - M1 补丁草案（MoE 逐专家循环 → token 排序/分桶 + 连续切片）：`notes/minimind-v/patches/M1-moe-vectorized-dispatch.md`
- 有卡待执行（一次开机跑完，预计 ¥3–4，不含可选全量复跑）：Step A 环境自检 → Step B 数值等价单测 → Step C 500-step A/B + profiler → Step D 5,000-step 受控训练 + 评估。
- 纪律：修改官方源码前先出 diff；所有实验落在 `runs/experiments/`，不覆盖 `runs/sft_full/`；负结果同样归档。

## 9. 仓库交接与改名（2026-09-26）

- GitHub 仓库已重命名：`RXCCCCCC/minimind-v-notes` → **`RXCCCCCC/minimind-v-research`**（保留全部 history / commits / notes / results）。
- 策略：后续只维护这一个 Research 仓库（不再新建独立源码 fork；upstream 源码后续以 subtree/等价方案纳入 `src/minimind-v/`，由接手 Agent 审查后执行）。
- upstream 复现基准 commit：`dacc68788998056476b21a4ca11325bdb277948c`（jingyaogong/minimind-v）。
- 本地 origin 已更新为新仓库地址；AutoDL 侧 remote 与物理目录改名待 SSH 网络恢复后同步。
- 下一步：Modification → Controlled Experiment → Ablation 交由 GPT-6 Sol 执行；本 Codex 仅负责正式长训练的启动与监控。
- 交接提示词：`notes/minimind-v/MiniMind-V Modification 阶段 GPT-6 Sol 交接提示词.md`
