# 2026-09-25-sft-vlm-moe-bs64-2epoch

## 目标
在 AutoDL 上以官方 MiniMind-V master（dacc687）完整复现 MoE SFT：Epoch 2 卡死后从 step 40,000 断点续训，跑完 2 epochs × 45,383 步，产出最终权重，并与官方 SFT 权重做评估对比。

## 环境
- 平台：AutoDL 重庆 A 区（单卡）
- GPU：NVIDIA RTX 4090 24GB × 1
- 基础镜像：PyTorch 2.8.0 / Python 3.12 / Ubuntu 22.04 / CUDA 12.8
- 项目环境：conda `minimind-v`（Python 3.10.21，torch 2.6.0+cu126，31/31 依赖对齐）
- 本次续训时长：2026-09-25 12:00:34 → 12:24:15 ≈ 23.8 分钟（step 40,000 → 45,383）

## 数据
- 数据集：`dataset/sft_i2t.parquet`（官方 SFT 图文数据，2,904,511 行）
- 预处理：官方 tokenizer + chat template；`max_seq_len=768`（截断率仅 0.34%）

## 训练命令（续训段）
```bash
cd /root/autodl-tmp/minimind-v/trainer
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export HF_DATASETS_CACHE=/root/autodl-tmp/.cache/huggingface/datasets
python -u train_sft_vlm.py --use_moe 1 --epochs 2 --batch_size 64 \
  --accumulation_steps 1 --learning_rate 5e-6 --max_seq_len 768 --freeze_llm 1 \
  --from_weight pretrain_vlm --from_resume 1 \
  --data_path ../dataset/sft_i2t.parquet \
  --save_dir /root/autodl-tmp/minimind-v/runs/sft_full \
  --save_weight sft_vlm --save_interval 5000 --log_interval 200 \
  --num_workers 6 --use_compile 1 --device cuda:0
```
> 说明：前序曾按官方默认 batch=4 / compile=0 起步，经基准测试（真实数据 + profiler）确认瓶颈为 launch-bound，采用高吞吐配置 batch=64 / compile=1 / seq=768（详情见 `notes/minimind-v/PROJECT_LOG.md`）。

## 关键指标
| 指标 | 数值 |
|---|---|
| 最终 loss | 1.9240 |
| 最终 logits_loss | 1.9200 |
| 最终 aux_loss | 0.0040 |
| 最终 lr | 5.0e-7 |
| 续训吞吐 | ~3.9 steps/s（bs64 + compile） |
| 断点续训 | step 40,000 → 45,383/45,383（Epoch 2 完成） |
| 自训模型评估速度 | ~108 tokens/s（13 张图，首图预热 71.1） |
| 官方模型评估速度 | ~100 tokens/s（13 张图） |

## 结论
- 官方 MoE SFT（2 epochs）完整复现成功。
- Epoch 2 曾在 44,800 附近卡死约 10h（疑似 dataloader worker 死锁，stderr 未留存）；改用 `--from_resume 1` + `num_workers=6` + 自愈看门狗（10 分钟无进展自动重启），一次续跑成功。
- 自训模型评估输出正常，生成速度与官方相当（略快）。
- 下一步：Evaluation 汇总 → Baseline（官方默认配置）→ Modification → Controlled Experiment → Ablation。

## 产物
- 曲线：`curves/loss_curve.png`
- 指标：`metrics.json`
- 日志：`train.log`（Epoch 1 + Epoch 2 至 44,800）、`train_resume.log`（40,000→45,383）、`eval.log`（自训 vs 官方评估输出）
- 权重（不入库，保留在 AutoDL）：`runs/sft_full/sft_vlm_768_moe.pth`（409MB）、`checkpoints/sft_vlm_768_moe_resume.pth`（1.57GB，含优化器）
