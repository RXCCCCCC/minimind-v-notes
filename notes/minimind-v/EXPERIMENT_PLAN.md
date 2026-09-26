# MiniMind-V 实验计划（Baseline → Modification → Controlled → Ablation）

> 本文件在**无卡模式**下准备；所有需要 GPU 的步骤集中在「有卡执行清单」，一次开机跑完。
> 口径纪律：报告中区分 `Official-default MoE reference (batch=4, compile=0)` 与 `High-throughput MoE configuration (batch=64, compile=1, seq=768)`；`torch.compile` 属 Execution/Engineering Optimization；batch scaling 属 Throughput-oriented training configuration，不算「纯工程优化」。

## 0. 当前状态
- 完整 Pretrain ✅（2 epochs，RC=0）
- 完整 SFT ✅（MoE，2 epochs × 45,383 步，最终 loss 1.9240；8.8 节记录）
- Evaluation ✅（自训 ≈108 t/s vs 官方 ≈101 t/s；见 run 目录 `EVAL_SUMMARY.md`）
- 官方源码零修改（`git diff` 为空）；所有改动将以补丁 + 独立实验目录方式管理。

## 1. 基线定义（防混淆）
| 名称 | 内容 | 用途 |
|---|---|---|
| Baseline-Q（质量参照） | 官方发布权重 `out/sft_vlm_768_moe.pth` | 质量对标 |
| Baseline-C（配置对照） | 官方默认配置 batch=4 / compile=0 / seq=768 | 吞吐与「历史 2h/epoch 说法」对照 |
| Repro-H（已产出） | 高吞吐配置 batch=64 / compile=1 | 后续 Modification 的实验母线 |

## 2. Modification 候选与优先级
| ID | 内容 | 预期收益 | 风险 | 优先级 |
|---|---|---|---|---|
| M1 | MoE 逐专家 Python 循环 → token 排序/分桶 + 连续切片 GEMM（减少 mask/index 类 kernel 与 `index_add_` 次数） | step/s ↑、GPU util ↑ | 数值漂移（需等价性测试）、与 compile 交互 | ★★★ |
| M2 | SFT optimizer 传入 `filter(lambda p: p.requires_grad, ...)`（与 pretrain 一致） | 工程一致性；避免冻结参数进入 param_groups | 几乎无 | ★★ |
| M3 | `count_vision_proj` 向量化（去掉 `.tolist()` 同步 + Python 循环） | **实测贡献 ≈ 0（profiler 已证非瓶颈）** | 无 | ★（低） |

M1 详细补丁草案：`patches/M1-moe-vectorized-dispatch.md`

## 3. Controlled Experiment 设计
- 固定项：同一份 SFT 数据、同 seed(42)、同 batch=64、compile=1、同 lr 调度、同 `pretrain_vlm` 起点、同 `max_seq_len=768`、相同步数。
- 对照组：`--impl loop`（原版） vs `--impl sorted`（M1）。
- 短程预算：每组 500 step（≈5–6 min，含加载/编译）。
- 指标：
  1) step/s、samples/s、GPU util、VRAM 峰值；
  2) profiler：kernel 数/step、GPU busy 时间/step（收敛前后各取 50 步）；
  3) 数值等价性：前 100 步 loss 与参考实现 max|Δ| < 1e-4（fp32 单 batch 单测 < 1e-5）；
  4) 若短程等价且吞吐 ≥ +10%，进入 5,000-step 受控训练 + 评估。
- 判定与止损：任何一项不达标 → 停止该方向，记录负结果（负结果同样入库）。

## 4. Ablation 矩阵（分层递增，逐步确认）
| 层级 | 变量 | 取值 | 每组步数 | 预计成本（¥，4090 2.08/h） |
|---|---|---|---|---|
| T1 | expert dispatch 实现 | loop / sorted | 500 | ≈0.2 × 2 |
| T2 | torch.compile | 0 / 1 | 500 | ≈0.2 × 2 |
| T3 | num_experts_per_tok | 1 / 2 | 500 | ≈0.2 × 2 |
| T4（可选） | num_experts | 4 / 8 | 500 | ≈0.2 × 2 |
| T5（可选，仅 T1 显著时） | 正式 MoE SFT 复跑 | baseline / M1 | 2 epoch | ≈13 × 2 |

## 5. 有卡执行清单（建议一次开机跑完）
> 每个实验独立目录：`/root/autodl-tmp/minimind-v/runs/experiments/<exp-id>/`，绝不覆盖 `runs/sft_full/`；日志与 profiler trace 同步回填本 repo。

- Step A｜环境自检（≈5 min，¥0.2）：`nvidia-smi`、conda env、git status（确认零修改）、数据盘余量。
- Step B｜M1 数值等价性单测（≈10 min，¥0.4）：CPU/GPU 各一遍，fp32 单 batch 前/反向对比。
- Step C｜500-step A/B + profiler（≈25 min，¥0.9）：`loop` vs `sorted`，记录吞吐/kernel 数/GPU busy。
- Step D｜5,000-step 受控训练 + 评估（≈60 min，¥2.0）：仅当 Step C 通过；产短程权重 + 13 图评估。
- Step E（可选）｜正式全量 SFT 复跑（≈6.5 h，¥13.5/次）：仅当 Step D 显示稳定收益。
- Step F｜整理：metrics.json、loss 曲线、diff、结论，回填 notes 并 push。

预计总额（不含 Step E）：**≈¥3–4**；Step E 视结果再决定。

## 6. 风险登记
- 无卡模式 cgroup 内存 ≈2GB：任何加载模型/checkpoint 的操作都不能在无卡做（会 OOM 被杀）。
- M1 与 `torch.compile` 的交互不确定：`sorted` 实现可能被 dynamo 改写，需 compile=0/1 分开测。
- 修改官方源码前必须先出 diff；测试通过前不进入主线训练目录。
- 若 500-step 吞吐提升 <10%，按「负结果」归档，不继续投入正式训练预算。
