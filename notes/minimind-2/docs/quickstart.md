# 快速开始

几分钟内让 MiniMind 跑起来!

## 📋 环境要求

### 硬件

- **GPU 显存**:最低 8GB(建议 24GB,开发更舒适)
- **推荐 GPU**:NVIDIA RTX 3090(24GB)

### 软件

- **Python**:3.10+
- **PyTorch**:2.0+(配合 CUDA 12.2+ 使用 GPU 支持)
- **CUDA**:12.2+(可选,用于 GPU 加速)

!!! tip "硬件配置参考"
    - **CPU**:Intel i9-10980XE @ 3.00GHz
    - **内存**:128 GB
    - **GPU**:NVIDIA GeForce RTX 3090 (24GB) × 8
    - **系统**:Ubuntu 20.04
    - **CUDA**:12.2
    - **Python**:3.10.16

## 🚀 第 0 步:克隆仓库

```bash
git clone https://github.com/jingyaogong/minimind.git
cd minimind
```

## 🎯 第一部分:测试现有模型

### 1. 环境搭建

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

!!! warning "验证 CUDA 支持"
    安装完成后,验证 PyTorch 能否访问 CUDA:
    ```python
    import torch
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    ```
    如果输出为 `False`,请从 [PyTorch 官方](https://download.pytorch.org/whl/torch_stable.html) 下载对应的 PyTorch 版本

### 2. 下载预训练模型

选择其中一种方式:

**从 HuggingFace 下载**(推荐海外用户):
```bash
git clone https://huggingface.co/jingyaogong/MiniMind2
```

**从 ModelScope 下载**(推荐中国用户):
```bash
git clone https://www.modelscope.cn/models/gongjy/MiniMind2.git
```

### 3. 命令行聊天

```bash
# load=0:加载 PyTorch 模型, load=1:加载 transformers 模型
python eval_model.py --load 1 --model_mode 2
```

**模型模式(Model Modes)**:
- `model_mode 0`:预训练模型(续写)
- `model_mode 1`:SFT 聊天模型(对话)
- `model_mode 2`:RLHF 模型(精炼回复,小型模型下目前与 SFT 相同)
- `model_mode 3`:推理模型(带思考链)
- `model_mode 4/5`:RLAIF 模型(PPO/GRPO 训练)

**示例对话**:
```text
👶:你好,请介绍一下你自己。
🤖️:我是 MiniMind,由 Gong Jingyao 开发的 AI 助手。
    我使用自然语言处理和机器学习算法与用户交互。

👶:法国的首都是哪里?
🤖️:法国的首都是巴黎,位于法国中北部。
    它是法国最大的城市,也是其政治、经济和文化中心。
```

### 4. Web UI Demo(可选)

```bash
# 需要 Python >= 3.10
pip install streamlit

cd scripts
streamlit run web_demo.py
```

访问 `http://localhost:8501` 使用交互式网页界面。

### 5. 使用 YaRN 进行 RoPE 长度外推

利用 RoPE 外推将上下文长度扩展到训练范围之外:

```bash
python eval_model.py --inference_rope_scaling True
```

这会启用 YaRN 算法来处理超过 2K 训练上下文的更长子序列,适合处理长文档和长对话。

## 🔧 第三方推理框架

MiniMind 兼容主流的推理引擎:

### Ollama(最简单)

```bash
ollama run jingyaogong/minimind2
```

### vLLM(最快)

```bash
vllm serve ./MiniMind2/ --served-model-name "minimind" --port 8000

# 用 curl 测试
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minimind",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

### llama.cpp(CPU 友好)

```bash
# 转换为 GGUF 格式
python scripts/convert_model.py ./MiniMind2/ --output ./MiniMind2.gguf

# 量化以减小体积
./llama-quantize ./MiniMind2.gguf ./MiniMind2-Q4.gguf Q4_K_M

# 运行推理
./llama-cli -m ./MiniMind2-Q4.gguf -p "Hello" -n 128
```

## 🔌 OpenAI API 服务器(用于集成)

将 MiniMind 作为兼容 OpenAI API 的服务运行:

```bash
python scripts/serve_openai_api.py
```

测试 API:

```bash
# 在另一个终端中
python scripts/chat_openai_api.py
```

**cURL 示例**:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minimind",
    "messages": [
      {"role": "user", "content": "Explain machine learning in one sentence."}
    ],
    "temperature": 0.7,
    "max_tokens": 256,
    "stream": true
  }'
```

这可用于集成:
- [FastGPT](https://fastgpt.run/)
- [Open-WebUI](https://github.com/open-webui/open-webui)
- [Dify](https://dify.ai/)
- 任何兼容 OpenAI API 的客户端

## 📊 模型选择指南

| 使用场景 | 推荐模型 | 显存 | 速度 |
|----------|------------------|--------|-------|
| 学习/测试 | MiniMind2-small (26M) | ~0.5 GB | 最快 |
| 均衡 | MiniMind2 (104M) | ~1.0 GB | 快 |
| 专家系统 (MoE) | MiniMind2-MoE (145M) | ~1.0 GB | 动态 |
| 推理/复杂任务 | MiniMind-Reason (104M) | ~1.0 GB | 标准 |

## ⚡ 快速测试结果

**模型**:MiniMind2(104M 参数)

```text
Q:什么是光合作用?
A:光合作用是植物将来自太阳的光能转化为化学能、以产生葡萄糖的过程。
   这一过程主要发生在叶片中,对植物的生长和生存至关重要。

Q:写一个 Python 函数计算斐波那契数列。
A: def fibonacci(n):
       if n <= 1:
           return n
       return fibonacci(n-1) + fibonacci(n-2)
   
   # 为获得更好性能,可使用动态规划:
   def fibonacci_dp(n):
       dp = [0] * (n + 1)
       for i in range(2, n + 1):
           dp[i] = dp[i-1] + dp[i-2]
       return dp[n]

Q:世界上最高的山峰是什么?
A:珠穆朗玛峰(Mount Everest)是世界上最高的山峰,位于喜马拉雅山脉...
```

## 🆘 故障排查

### 问题:CUDA 显存不足

**解决方案**:
```bash
# 减小 batch size
python eval_model.py --batch_size 1

# 或使用 CPU(慢但可用)
python eval_model.py --device cpu
```

### 问题:推理速度慢

**解决方案**:
- 使用 vLLM 或 llama.cpp 以加快推理
- 启用量化(4-bit、8-bit)
- 使用 GPU 而非 CPU
- 减小 `max_tokens` 参数

### 问题:模型回复质量差

**可能原因**:
- 使用了预训练模型(`model_mode 0`)而非 SFT 模型(`model_mode 1`)
- 模型欠训练——请下载完整检查点(checkpoint)
- 输入提示过短——请提供更多上下文

### 问题:Python/PyTorch 版本不匹配

**解决方案**:
```bash
# 使用 conda 创建干净环境
conda create -n minimind python=3.10
conda activate minimind
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu122
pip install -r requirements.txt
```

## 📖 下一步

- **[模型训练指南](training.md)** - 从零训练你自己的 MiniMind
- **[源代码](https://github.com/jingyaogong/minimind)** - 探索并学习 LLM 实现
- **[推理基准](https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5)** - 查看模型性能对比

## 💡 进阶技巧

1. **GPU 显存优化**:定期使用 `torch.cuda.empty_cache()`
2. **批量处理**:为提高效率,将多个提示分批处理
3. **温度调节**:更小(0.3-0.7)= 更一致,更大(0.8-1.0)= 更有创造性
4. **提示词工程**:更好的提示 → 更好的结果,即使对小模型也如此
5. **模型量化**:使用 4-bit 量化以在更小的 GPU 上运行

---

完成!现在你已经准备好使用 MiniMind。从快速开始入手,再转向 [模型训练](training.md) 学习如何训练自己的模型。
