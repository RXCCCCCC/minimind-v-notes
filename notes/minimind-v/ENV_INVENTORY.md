# MiniMind-V 环境盘点 / 可复现起点

生成时间: 2026-09-22 17:55:31 CST

## 1. 硬件 / 宿主
- AutoDL 实例: autodl-container-ed0440a042-9cb636a9
- GPU: NVIDIA GeForce RTX 4090 24GB (driver 595.71.05)
- OS: Ubuntu 22.04.5 LTS kernel 5.15.0-160-generic
- 系统盘 /: 30G 总 / 7.5G 已用 / 23G 可用
- 数据盘 /root/autodl-tmp: 50G 总 / 4.6G 已用 / 46G 可用

## 2. Conda 环境
- 环境名: minimind-v
- 路径: /root/miniconda3/envs/minimind-v
- Python: Python 3.10.21
- base 环境未改动: base = Python 3.12 + torch 2.8.0+cu128

## 3. 关键版本
- torch: 2.6.0+cu126 (CUDA build 12.6)
- transformers: 4.57.6
- datasets: 3.6.0
- numpy: 1.26.4
- pyarrow: 23.0.0
- Pillow: 11.3.0
- gradio: 5.49.1
- swanlab: 0.6.12
- modelscope: 1.40.1

## 4. 仓库
- 路径: /root/autodl-tmp/minimind-v
- HEAD: dacc687 [fix] fp16 attention overflow
- remote: https://gh-proxy.com/https://github.com/jingyaogong/minimind-v.git

## 5. 已下载权重（SHA256 已与 ModelScope 官方比对一致）
```
-rw-r--r-- 1 root root 189129296 Sep 22 17:37 model/siglip2-base-p32-256-ve/model.safetensors
-rw-r--r-- 1 root root 406719725 Sep 22 17:39 out/llm_768_moe.pth
-rw-r--r-- 1 root root 409088616 Sep 22 17:42 out/pretrain_vlm_768_moe.pth
-rw-r--r-- 1 root root 409087701 Sep 22 17:44 out/sft_vlm_768_moe.pth
```

## 6. 参考命令
```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate minimind-v
cd /root/autodl-tmp/minimind-v
# 官方 inference (MoE, 需要 GPU):
python eval_vlm.py --load_from model --weight sft_vlm --use_moe 1
```
