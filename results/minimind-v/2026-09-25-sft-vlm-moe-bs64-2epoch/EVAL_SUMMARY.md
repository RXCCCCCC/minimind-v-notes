# 评估对比：自训 MoE SFT vs 官方 MoE SFT（2026-09-25）

## 设置
- 模型：MiniMind-V MoE（199.60M 总参 / 65.12M 激活；hidden 768、8 层、4 experts、top-1）
- 权重：
  - 自训：`runs/sft_full/sft_vlm_768_moe.pth`（本次 2 epochs MoE SFT 完整复现）
  - 官方：`out/sft_vlm_768_moe.pth`（ModelScope 官方发布，SHA256 已核对一致）
- 命令：`python eval_vlm.py --load_from model --weight sft_vlm --save_dir <dir> --use_moe 1 --show_speed 1`
- 数据：`dataset/eval_images/` 共 13 张；prompt 固定「请描述这张图中的主要物体和场景」
- 生成参数：max_new_tokens=512、temperature=0.7、top_p=0.85、bf16、RTX 4090 24GB ×1

## 生成速度对比（tokens/s）

| # | 图像 | 自训 | 官方 |
|---|---|---|---|
| 1 | image-01-golden-dog-balloons.jpg | 71.10（预热） | 108.73 |
| 2 | image-02-rainbow-umbrella-street.jpg | 110.44 | 82.29（预热） |
| 3 | image-03-cherry-blossom-bike.jpg | 110.35 | 101.22 |
| 4 | image-04-yellow-car.jpg | 110.16 | 102.78 |
| 5 | image-05-superhero-rooftop.jpg | 108.40 | 101.78 |
| 6 | image-06-racecar-drift.jpg | 108.08 | 101.28 |
| 7 | 城市车水马龙-city-traffic.jpg | 108.85 | 100.78 |
| 8 | 太空宇航员-Astronaut-Space.jpg | 108.39 | 101.40 |
| 9 | 小狗美女海边-Dog-Woman-Sea.jpg | 107.14 | 100.48 |
| 10 | 彩虹瀑布-Rainbow-Falls .jpg | 108.54 | 100.53 |
| 11 | 椅子老人看书-Chair-Elderly-Reading.jpg | 108.44 | 101.11 |
| 12 | 熊猫草地-Panda-Grassland.jpg | 108.37 | 100.94 |
| 13 | 自行车鲜花-Bicycle-Flowers.jpg | 109.44 | 100.67 |
| — | **平均（含预热）** | **105.98** | **100.31** |
| — | 平均（剔除各自预热图） | ≈108.7 | ≈100.9 |

结论：自训权重推理速度正常，整体略快于官方权重（差异主要来自生成长度与采样随机性，非架构差异）。

## 文本质量（定性）
- 两版都能正确识别主体与大致场景；细节幻觉普遍存在：
  - 官方：金毛被描述成「巴哥犬」；彩虹伞被描述成「彩虹色玻璃」。
  - 自训：金毛被描述出「红蓝斑点」；黄色汽车被描述成「波音747」。
- 双方处于同一水平：主体正确、属性/计数/背景常见错误，符合 0.2B 量级 VLM 的预期。
- 结论：自训 checkpoints 功能上与官方等价，可作为后续 Modification / Ablation 的工程基线；官方权重作为质量参照。

## 产物
- 逐图原始输出（自训 13 段 + 官方 13 段 + 速度）：`eval.log`
- 结构化速度指标：`metrics.json`
