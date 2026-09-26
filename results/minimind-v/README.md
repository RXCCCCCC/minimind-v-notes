# MiniMind-V 训练结果

本目录沉淀 `minimind-v` 的可展示证据，用于简历、导师沟通与技术复盘。

训练在 AutoDL 上执行，结果必须回流到本地并推送到 GitHub 仓库 `minimind-v-research`，不能只留在实例上。

## 目录约定

每次有效训练建一个独立目录：

```text
results/minimind-v/<日期>-<任务>-<关键配置>/
```

例如 `results/minimind-v/2026-10-01-sft-vlm-freeze-llm-1xa100/`。

每个 run 目录至少包含：

| 文件 | 说明 |
|---|---|
| `README.md` | run 总结，模板见下 |
| `metrics.csv` / `metrics.json` | loss、lr、ppl 等指标 |
| `train.log` / `eval.log` | 原始日志 |
| `curves/*.png` | loss 曲线等图表 |
| `screenshots/*.png` | WebUI 对话、图像理解效果截图 |

不提交：权重（`*.pth`、`*.safetensors`）、`checkpoints/`、原始数据集。大文件走 GitHub Release 或外部链接。

## run README 模板

把下面内容复制到 `results/minimind-v/<run-name>/README.md` 并逐项填写。

````markdown
# <run-name>

## 目标
这次训练要验证什么。

## 环境
- 平台：AutoDL
- GPU：<型号> × <卡数>
- 镜像 / Python / PyTorch：
- 训练时长：

## 数据
- 数据集与版本：
- 样本量：
- 预处理：

## 训练命令
```bash
python train_sft_vlm.py --epochs 2 --from_weight llm
```

## 关键指标
| 指标 | 数值 |
|---|---|
| loss | |
| ppl | |

## 结论
发现了什么，下一步做什么。

## 产物
- 曲线：`curves/loss.png`
- 截图：`screenshots/demo.png`
````

## AutoDL 回填流程

训练侧不建目录结构，按约定路径回填即可。

```bash
# AutoDL：打包本次 run 的指标、日志与截图
tar czf <run-name>.tar.gz <文件列表>
```

```bash
# 本地：解压到对应 run 目录，提交并推送
cd /home/rxcccccc/workspace/project/learning/ai/llm/minimind
mkdir -p results/minimind-v/<run-name>
tar xzf ~/downloads/<run-name>.tar.gz -C results/minimind-v/<run-name>
git add results/minimind-v/<run-name>
git commit -m "docs: 添加 <run-name> 训练结果"
git push
```
