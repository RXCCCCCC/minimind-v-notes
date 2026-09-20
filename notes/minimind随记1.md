[TOC]



# 前置知识

# 第一轮

# 一、我们整个 MiniMind 学习阶段需要补的前置知识地图

我会把它拆成 **5 轮**。

| 轮次 | 内容 | 重要程度 | 解决的问题 |

| --- | --- | --- | --- |

| **第 1 轮** | PyTorch Tensor + Autograd + Training Loop + Next Token Prediction | ★★★★★ | 看懂训练到底怎么发生 |

| **第 2 轮** | Tokenizer + Dataset + DataLoader + Padding + Mask + SFT 数据 | ★★★★★ | 看懂文字怎么真正进入模型 |

| **第 3 轮** | Decoder-Only Transformer：RMSNorm、RoPE、GQA、SwiGLU、Residual | ★★★★★ | 真正看懂 `model_minimind.py` |

| **第 4 轮** | LLM 训练工程：AdamW、LR、混合精度、梯度累积、clip、checkpoint、KV Cache | ★★★★☆ | 能自己跑、调、分析训练 |

| **第 5 轮** | LoRA / MoE / DPO / GRPO + baseline / metric / ablation | ★★★☆☆→★★★★★ | 从“复现项目”升级成“科研简历项目” |
其中：
**前 4 轮属于你真正应该补齐的 MiniMind 核心前置。**
第 5 轮不用全部掌握以后才能运行 MiniMind，而是为你最后的目标：

> **不是 GitHub 上 clone 下来跑通，而是形成一个你能够写进科研简历、能接受导师追问的项目。**
> 服务的。
> 并且我们继续沿用之前确定的路线：
> **MiniMind 快速复现 → 做一个有 baseline 的小型修改/优化 → 形成实验结果 → 写入简历 → 再进入 VisionZip；CS336 并行补理论。**

# 二、第一轮：你最应该先建立的“大模型训练心智模型”

这一轮其实比重新学 Transformer 更重要。
先记住整个训练过程只有这么一条链：

```
文本
↓
Tokenizer
↓
token ids
↓
Dataset / DataLoader
↓
input_ids
↓
Embedding
↓
Transformer
↓
logits
↓
Cross Entropy Loss
↓
backward()
↓
gradient
↓
optimizer.step()
↓
参数改变
↓
重复
```

你以后看到任何大模型训练框架——MiniMind、nanoGPT、Llama、Qwen，甚至 CS336——本质上都逃不出这条链。
框架只是不断把其中某些部分复杂化。

# 三、第一件必须补牢的东西：Tensor Shape

你以后读模型源码时，**不要优先读每一个函数到底什么意思**。
首先看：
> **这个 Tensor 现在是什么 shape？**
> MiniMind 的核心 Tensor 基本可以统一成下面几个符号：

| 符号 | 含义 |

| --- | --- |

| `B` | Batch Size |

| `T` | Sequence Length |

| `C` | Hidden Size |

| `V` | Vocabulary Size |

| `H` | Attention Heads |

| `D` | Head Dimension |
例如一句话经过 tokenizer：

```
"我喜欢机器学习"
```

可能变成：

```
[1, 328, 712, 55, 891, 2]
```

如果一次训练 32 条文本，每条被补到 340 token：

```
input_ids.shape
```

就是：

```
[B, T]
=
[32, 340]
```

每一个数字只是一个 **token ID**。
还完全不是所谓的“向量表示”。

## 进入 Embedding 后

MiniMind 里面有：

```
self.embed_tokens = nn.Embedding(
    config.vocab_size,
    config.hidden_size
)
```

当前默认：

```
V = 6400
C = 768
```

于是：

```
input_ids
[B, T]

↓

Embedding

↓

hidden_states
[B, T, C]
```

例如：

```
[32, 340]
↓
[32, 340, 768]
```

你可以把它理解成：

```
原来：

token 328

现在：

token 328
→
[0.18, -0.72, 0.33, ..., 0.41]
          768维
```

Transformer 后面真正处理的，就是**这些 768 维向量。**
MiniMind 当前源码正是**从 `embed_tokens(input_ids)` 获得 `hidden_states`**，**随后依次经过 Transformer Blocks。**

# 四、你必须养成一个习惯：沿着 Shape 追源码

例如以后看到：

```
hidden_states = self.embed_tokens(input_ids)
```

不要只是说：
> “哦，这是 embedding。”
> 你应该脑内自动翻译成：

```
input_ids:
[B, T]

↓ nn.Embedding(V, C)

hidden_states:
[B, T, C]
```

以后又看到：

```
xq = self.q_proj(x)
```

也不要只是：
> “这是 Q。”
> 而应该继续问：

```
输入是什么？
[B,T,C]

Linear 输出多少维？

然后 view 成什么？

[B,T,H,D]
```

如果你能做到这种程度，**你读模型源码的能力会发生质变。**
这是你 MiniMind 阶段非常重要的一项训练。

# 五、第二件必须彻底搞懂：语言模型到底在学什么？

这是 MiniMind 最核心的东西。
其实 LLM 的预训练目标极其朴素：
> **根据前面的 token，预测下一个 token。**
> 也就是：

## Next Token Prediction

假设一句话经过 tokenizer 以后是：

```
<BOS> 我 喜欢 机器 学习 <EOS>
```

训练样本实际上相当于：

```
输入：<BOS>
目标：我

输入：<BOS> 我
目标：喜欢

输入：<BOS> 我 喜欢
目标：机器

输入：<BOS> 我 喜欢 机器
目标：学习
```

但是 **GPU 不会真的这样一遍一遍分别算。**
**Transformer 会一次性计算整个序列。**
因此：

```
input_ids

<BOS>   我   喜欢   机器   学习   <EOS>
```

模型输出：

```
logits_0
logits_1
logits_2
logits_3
logits_4
logits_5
```

其中：

```
logits_0 → 应该预测 “我”
logits_1 → 应该预测 “喜欢”
logits_2 → 应该预测 “机器”
logits_3 → 应该预测 “学习”
logits_4 → 应该预测 <EOS>
```

所以训练时需要：

```
预测：

logits[:-1]

对应：

labels[1:]
```

# 六、然后你再看 MiniMind 源码，会突然非常清楚
MiniMind 当前就是这么写的：

```
x = logits[..., :-1, :]
y = labels[..., 1:]

loss = F.cross_entropy(...)
```

以前你可能看到：

```
[..., :-1, :]
[..., 1:]
```

会觉得：
> 这是什么奇怪的 tensor 操作？
> 现在它其实非常简单：

```
预测位置0 → 标签位置1
预测位置1 → 标签位置2
预测位置2 → 标签位置3
...
```

也就是把：

```
模型预测
```

和：

```
真正的下一个 token
```

对齐。
这是整个 **Causal Language Modeling** 最核心的一步。

# 七、logits 又是什么？

假设词表：

```
V = 6400
```

模型某一个位置不是直接输出：

```
"机器"
```

而是输出：

```
6400 个数字
```

比如：

```
token 0     -3.2
token 1      0.7
token 2     -1.6
...
token 812    7.8
...
token 6399  -2.1
```

这组东西就是：

```
logits
```

因此整体：

```
hidden_states
[B,T,C]

↓

lm_head

↓

logits
[B,T,V]
```

MiniMind 里：

```
self.lm_head = nn.Linear(
    hidden_size,
    vocab_size
)
```

所以：

```
[B,T,768]

↓

Linear

↓

[B,T,6400]
```

# 八、Cross Entropy 在这里到底干了什么？
假设正确答案是 token：

```
机器 = token 812
```

模型给 6400 个 token 打了一遍分。
Cross Entropy 所做的事情，可以先不用复杂数学理解，直接理解成：
> **检查正确 token 的概率够不够高。**
> 如果模型预测：

```
机器：0.80
汽车：0.05
学习：0.03
……
```

loss 很小。
如果：

```
机器：0.001
汽车：0.70
电脑：0.20
……
```

loss 很大。
于是：

```
loss
↓
backward
↓
gradient
↓
告诉所有参数：
“怎么改才能让下次正确 token 的概率更大？”
```

这就是训练。
没有什么神秘的“大模型学习算法”。
最底层依旧只是：

```
Forward
→ Loss
→ Backpropagation
→ Gradient Descent
```

你李沐课程里其实已经学过。
只是现在模型非常大。

# 九、这里必须理解 `-100`

看 MiniMind 的 `PretrainDataset`：

```
labels = input_ids.clone()

labels[
    input_ids == pad_token_id
] = -100
```

而 loss：

```
F.cross_entropy(
    ...,
    ignore_index=-100
)
```

假设原句只有：

```
<BOS> 我 喜欢 AI <EOS>
```

为了拼成统一长度可能变成：

```
<BOS> 我 喜欢 AI <EOS> PAD PAD PAD
```

我们当然不希望模型学习：

```
PAD → PAD → PAD → PAD
```

所以 label 变成：

```
<BOS> 我 喜欢 AI <EOS> -100 -100 -100
```

CrossEntropy 看到：

```
-100
```

就：
> **这里不计算 loss。**
> 这就是 `ignore_index=-100`。
> 以后你读绝大多数 HuggingFace LLM 训练代码也会不停遇到它。

# 十、第三件必须彻底弄懂：`backward()` 和 `optimizer.step()` 完全不是一回事

这是 PyTorch 初学阶段特别容易“会用但没形成心智模型”的地方。
首先：

```
loss.backward()
```

做的是：

```
计算：

∂Loss / ∂parameter
```

也就是：
> **算梯度。**
> 它并不会改模型参数。
> 真正修改：

```
weight
```

的是：

```
optimizer.step()
```

所以：

```
loss.backward()

≠

模型已经学习了
```

而是：

```
loss.backward()
↓
知道应该往哪改

optimizer.step()
↓
真的修改参数
```

# 十一、因此最基本的 PyTorch Training Loop 就只有四步
你以后一定要把这个结构刻进脑子：

```
optimizer.zero_grad()

output = model(x)
loss = criterion(output, y)

loss.backward()

optimizer.step()
```

对应：

```
1. 清空旧梯度

2. forward
   ↓
   算预测和 loss

3. backward
   ↓
   算 gradient

4. optimizer.step
   ↓
   更新 parameters
```

所有复杂训练脚本其实都是在这四步周围不断堆东西。

# 十二、现在看 MiniMind 的训练代码

MiniMind 的 Pretrain 主循环本质上也是：

```
res = model(input_ids, labels=labels)

loss = res.loss

scaler.scale(loss).backward()

scaler.step(optimizer)

scaler.update()

optimizer.zero_grad()
```

只是它额外加入了：

```
混合精度
梯度累积
梯度裁剪
LR schedule
MoE aux loss
checkpoint
DDP
日志
```

所以以后看到 200 行训练代码不要怕。
你首先只找：

```
model(...)
loss
backward()
optimizer.step()
zero_grad()
```

骨架找到以后，其余全部是增强功能。

# 十三、为什么 `zero_grad()` 必须存在？

PyTorch 默认：
> **梯度会累加。**
> 假设第一次：

```
gradient = 3
```

第二次：

```
gradient = 4
```

如果不清：

```
最终 gradient = 7
```

而不是：

```
4
```

所以普通训练：

```
optimizer.zero_grad()
```

用来清空旧 gradient。
但是——
MiniMind 又故意没有每一个 batch 都清。
为什么？
因为：

## Gradient Accumulation

# 十四、梯度累积是你必须掌握的大模型训练概念
假设 GPU 显存只能：

```
batch_size = 4
```

但我希望一次参数更新相当于：

```
batch_size = 32
```

可以：

```
跑 batch 1
backward
不 step

跑 batch 2
backward
不 step

...

跑 batch 8
backward

↓

optimizer.step()
```

这样：

```
effective batch size
≈
batch_size × accumulation_steps
```

多 GPU 时：

```
Effective Batch Size
=
batch_size
× accumulation_steps
× GPU数量
```

MiniMind 默认 pretrain 里就有：

```
batch_size = 32
accumulation_steps = 8
```

并且只有：

```
step % accumulation_steps == 0
```

的时候才真正：

```
optimizer.step()
```

所以这段代码不是奇怪写法。
而是：
> **用时间换显存。**
> 这是大模型训练极常见的技术。

# 十五、`nn.Module`、`Parameter`、`Buffer` 你也必须分清

你现在看到：

```
class Attention(nn.Module)
class FeedForward(nn.Module)
class MiniMindBlock(nn.Module)
```

需要理解：
`nn.Module` 本质就是：
> **一个可以包含参数和其他 Module，并定义 forward 的计算单元。**
> 例如：

```
nn.Linear(...)
```

里面存在：

```
weight
```

这些是：

```
nn.Parameter
```

意味着：
> optimizer 应该训练它。
> 但是 MiniMind 的 RoPE 有：

```
self.register_buffer(
    "freqs_cos",
    freqs_cos
)
```

为什么不是 Parameter？
因为：

```
RoPE cos/sin 表
```

需要：

```
跟模型一起移动到 GPU
```

但：

```
不需要梯度下降更新
```

因此是：

```
buffer
```

非常重要的区别：

```
Parameter
→ 模型要学习的东西

Buffer
→ 模型运行需要保存/携带的东西
→ 但不是训练参数
```

# 十六、`model.train()` 和 `model.eval()` 不是“开始训练/开始测试”
这个概念也必须纠正。
很多初学者认为：

```
model.train()
```

意味着：
> PyTorch 开始训练。
> 不是。
> 它只是把模型切换到：

```
training mode
```

最典型影响：

```
Dropout
BatchNorm
```

例如 Dropout：

```
train()
→ 开启随机 dropout

eval()
→ 关闭 dropout
```

真正训练仍然必须：

```
forward
backward
optimizer.step
```

MiniMind **保存 checkpoint 时就暂时：**

```
model.eval()
```

保存后再：

```
model.train()
```

# 十七、然后再理解 `torch.inference_mode()`
MiniMind 的生成函数：

```
@torch.inference_mode()
def generate(...):
```

原因非常简单：
训练：

```
forward
↓
保存计算图
↓
backward
```

**推理：**

```
forward
↓
结束
```

**完全不需要算 gradient。**
**所以 inference mode 可以：**

```
减少显存
+
减少计算
```

# 十八、BF16 / FP16 / FP32 先掌握到这个程度就够
你现在不要陷入浮点数 IEEE 754 的细节。
只需要建立：

```
FP32
→ 精度高
→ 占显存多
→ 训练慢一些

FP16
→ 更省显存
→ 更快
→ 数值范围容易出问题

BF16
→ 更省显存
→ 数值范围比 FP16 更适合 DL
→ 新 GPU 大模型训练非常常见
```

MiniMind 默认：

```
dtype = bfloat16
```

**并使用 autocast；如果使用 float16，则启用 `GradScaler`。**
现在你只要知道：

> **Mixed Precision = 某些计算不用全部 FP32，以换速度和显存。**
> 后面实际跑训练时我们再把 AMP 讲透。

# 十九、AdamW 现在也只需要理解到这里

MiniMind Pretrain：

```
optimizer = optim.AdamW(...)
```

你学过 Gradient Descent，所以可以把它先理解为：

```
SGD

parameter
←
parameter - lr × gradient
```

AdamW 则是：

```
gradient
↓
根据一阶矩 / 二阶矩调整每个参数的更新尺度
↓
再结合 weight decay
↓
更新参数
```

暂时不用重新推 Adam 数学公式。
你现在必须理解的是三个概念：

```
gradient
→ 往哪走

learning rate
→ 一步走多远

optimizer
→ 根据 gradient 决定具体怎么走
```

# 二十、Gradient Clipping 又是什么？
MiniMind：

```
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    args.grad_clip
)
```

假设某一步梯度突然**非常巨大：**

```
正常 gradient norm ≈ 0.8

突然：

gradient norm = 500
```

**一次更新可能直接把参数打飞。**
**于是规定：**

```
如果 gradient 太大
↓
按比例缩小
```

**就是：**

```
gradient clipping
```

**可以先理解成：**
> **训练的保险丝。**

# 二十一、你现在应该第一次真正理解 MiniMind Pretrain 了

把代码细节全部去掉以后，MiniMind 的：

```
train_pretrain.py
```

其实就是：

```
读取文本
↓
Tokenizer
↓
PretrainDataset
↓
DataLoader
↓
得到 input_ids / labels
↓
MiniMind(input_ids)
↓
得到 [B,T,V] logits
↓
logits[:-1] 对 labels[1:]
↓
CrossEntropy
↓
loss
↓
backward
↓
gradient accumulation
↓
gradient clipping
↓
AdamW.step()
↓
保存 checkpoint
```

而当前 `PretrainDataset` 的确就是完成：

```
文本
→ tokenize
→ BOS/EOS
→ padding
→ tensor
→ label
```

这一转换。
到这里你其实已经看懂了**整个 LLM 预训练最外层的骨架**。

# 二十二、这一轮你暂时不要学什么

下面这些东西虽然 MiniMind 有，但我现在**不会让你提前学**：

```
DDP
DeepSpeed
torch.compile
YaRN
MoE load balancing
LoRA 数学细节
DPO
PPO
GRPO
CISPO
Agentic RL
Tokenizer 训练算法细节
```

因为这些现在都会破坏你的知识主线。
特别是 MiniMind 当前 README 已经扩展得非常大，如果你看到一个目录就补一个知识点，很容易再次变成：
> 什么都见过，但是训练主线依然没有真正建立。
> 我们不会这么学。

# 二十三、本轮你真正需要达到的验收标准

这一轮结束以后，你应该能**不看答案自己解释清楚下面 8 件事**：
1. `[B,T] → [B,T,C] → [B,T,V]` 分别代表什么。
2. `Embedding` 为什么把 token ID 变成向量。
3. LLM 为什么本质上是在做 Next Token Prediction。
4. 为什么 MiniMind 要用 `logits[..., :-1]` 对 `labels[..., 1:]`。
5. 为什么 padding 的 label 被设成 `-100`。
6. `loss.backward()` 与 `optimizer.step()` 的区别。
7. Gradient Accumulation 为什么能缓解显存问题。
8. `Parameter` 与 `Buffer` 为什么不一样。
这里面如果有任何一个你只能“似懂非懂”，都值得直接问我。
**不用死背定义。**
我们接下来会不断回到 MiniMind 的真实代码里重复这些东西，重复几次以后自然就牢了。

# 二十四、接下来第二轮我们要解决什么

下一轮我会直接带你补：

# **Tokenizer → Dataset → DataLoader → SFT**

而且会用 MiniMind 当前源码一步一步走：

```
一段中文

↓

BPE / ByteLevel Tokenizer

↓

token ids

↓

BOS / EOS / PAD

↓

max_seq_len / truncation

↓

PretrainDataset

↓

SFTDataset

↓

chat_template

↓

system / user / assistant

↓

loss mask

↓

为什么 SFT 不能简单把所有 token 都计算 loss
```

这个部分对你尤其重要，因为你以前做 Agent/AIGC 项目时看到的是：

```
messages
→ API
→ response
```

我们现在要把它彻底拆开，让你看到：

```
messages

到底是怎样一步一步变成

Tensor

然后训练进模型参数里的。
```

**等这个补完以后再进入第三轮 `model_minimind.py`，到时我会按 Tensor Shape 从头把 RMSNorm → RoPE → GQA → SwiGLU → Transformer Block 一层层带你读，而不是拿公式轰你。**

# 第二轮

好。我们进入 **MiniMind 前置知识第 2 轮：Tokenizer → Dataset → DataLoader → Pretrain / SFT 数据流**。
这一轮对你尤其重要，因为按照你现在的能力画像，你已经知道 Transformer、Embedding、Attention 这些概念，但 **PyTorch 独立训练和“数据到底怎么一路变成 loss”仍然是明显薄弱点**。所以这一轮不会把 Tokenizer 当 NLP 名词背，而是一直追踪 Tensor 和代码。

# 0. 这一轮学完，你应该真正看懂这条链

先把今天的地图放在脑子里：

```
原始文本
    ↓
Tokenizer
    ↓
token
    ↓
token id
    ↓
Dataset.__getitem__()
    ↓
单条 input_ids: [T]
单条 labels:    [T]
    ↓
DataLoader
    ↓
一批 input_ids: [B, T]
一批 labels:    [B, T]
    ↓
Embedding
    ↓
[B, T, C]
    ↓
Transformer
    ↓
logits [B, T, V]
    ↓
CrossEntropy(logits[:, :-1], labels[:, 1:])
    ↓
loss
```

上一轮我们主要学了后半截。
今天把前半截彻底补上。

# 1. 首先把 5 个极容易混的概念分清楚

以后一定不要混：

```
字符 character
单词 word
token
token id
embedding
```

它们完全不是一回事。
假设：

```
我喜欢机器学习
```

## 字符
可以粗略认为：

```
我
喜
欢
机
器
学
习
```

这是字符。

## Token

Tokenizer 可能把它切成：

```
["我", "喜欢", "机器", "学习"]
```

也可能：

```
["我", "喜欢", "机器学习"]
```

甚至：

```
["我", "喜", "欢", "机器", "学习"]
```

**到底怎么切，不是语言学规定的，而是 tokenizer 的词表和算法决定的。**

## Token ID

计算机不能直接拿：

```
"机器"
```

去做矩阵乘法。
所以 tokenizer 还有一个词表：

```
token         id

"我"          184
"喜欢"        932
"机器"        1281
"学习"        617
```

于是：

```
我喜欢机器学习
```

可能变成：

```
[184, 932, 1281, 617]
```

这就是：

```
input_ids
```

## Embedding
注意：

```
1281
```

本身不是“机器”的语义向量。
它只是：

> 词表里第 1281 号 token。

进入：

```
nn.Embedding(vocab_size, hidden_size)
```

以后才变成：

```
1281

↓

[0.17, -0.33, ..., 0.82]
          768维
```

所以一定记住：

```
token
↓
token id
↓
embedding vector
```

三个不同层级。

# 2. MiniMind 当前到底用了什么 Tokenizer？

当前 MiniMind 自带一个 tokenizer，仓库还专门保留了：

```
trainer/train_tokenizer.py
```

供学习。
但作者现在明确写了：

> 不建议为了正常训练重新训练 tokenizer，因为使用不同词表会造成不同 MiniMind 模型之间不兼容。

当前这个教学脚本使用的是：

```
Tokenizer(models.BPE())
```

并结合：

```
ByteLevel
```

预切分；设定：

```
VOCAB_SIZE = 6400
```

也就是说 MiniMind 当前默认词表规模大约就是：

# **V = 6400**

这也正好解释了上一轮：

```
logits.shape = [B, T, 6400]
```

为什么最后是 6400 维。

# 3. 为什么需要 Tokenizer？

你可能会问：

> 为什么不能直接一个汉字一个 token？

当然可以。
最粗暴：

```
我 喜 欢 机 器 学 习
```

但这样会有问题。
英文：

```
unbelievable
```

你怎么办？
一个字符一个 token：

```
u n b e l i e v a b l e
```

太长。
一个单词一个 token：

```
unbelievable
```

那世界上单词实在太多，而且：

```
believable
unbelievable
believe
believer
```

完全无法共享结构。
所以现代 LLM 通常采用：

# Subword Tokenization

也就是：

> 不一定按字，也不一定按完整单词，而是学习一些常出现的“子词片段”。

例如：

```
unbelievable

→

un
believ
able
```

这就是 token。

# 4. BPE 到底是什么？

MiniMind 的 tokenizer 使用 BPE：

> **Byte Pair Encoding**

你现阶段不需要学成 NLP tokenizer 专家，但必须知道基本思想。
假设训练语料里经常出现：

```
l o w
l o w e r
```

最开始什么都拆得很碎：

```
l
o
w
e
r
```

发现：

```
l + o
```

经常一起出现。
就合并：

```
lo
```

后来发现：

```
lo + w
```

又经常一起出现。
再合并：

```
low
```

于是经过很多轮：

```
字符 / byte
↓
统计最常见相邻组合
↓
不断 merge
↓
形成词表
```

最终得到：

```
"the"
"ing"
"tion"
"机器"
"学习"
...
```

这样的 token。

# 5. 那 ByteLevel 又是什么？

MiniMind 的脚本不是纯文本直接 BPE，而是：

```
pre_tokenizers.ByteLevel(...)
```

你现在可以先理解：

> **先让所有文本都可以从 byte 层面表示，再在这些表示上学习 BPE merge。**

好处之一是：

# 几乎不会遇到真正意义上的 OOV

即：

```
Out Of Vocabulary
```

因为再奇怪的字符串，最终都还能退化到底层 byte 表示。
比如一个罕见字符、奇怪符号、代码：

```
🦄
printf("%lld")
你好αβγ
```

不需要保证完整字符串都提前存在词表里。
实在不行就拆得更细。

# 6. Tokenizer 其实干两件事

很多初学者会把 tokenizer 理解成：

> “分词器”。

不完整。
它主要完成：

```
文本
↓
切成 tokens
↓
tokens 映射成 IDs
```

例如：

```
tokenizer("我喜欢机器学习")
```

结果核心是：

```
input_ids
```

例如：

```
[184, 932, 1281, 617]
```

反过来：

```
tokenizer.decode(...)
```

又可以：

```
[184, 932, 1281, 617]

↓

"我喜欢机器学习"
```

于是有：

```
encode:

Text → IDs

decode:

IDs → Text
```

# 7. Special Token：大模型训练里非常重要
普通 token 表示内容。
但模型还需要一些：

> **控制符号**

MiniMind 当前 tokenizer 配置了不少 special token，包括聊天、工具、视觉、音频等预留标记。
现在最重要的三个：

```
BOS
EOS
PAD
```

MiniMind 当前配置里：

```
BOS = <|im_start|>

EOS = <|im_end|>

PAD = <|endoftext|>
```

# 8. BOS 是干什么的？
BOS：

```
Beginning Of Sequence
```

例如原来：

```
今天天气很好
```

训练时可以变成：

```
<BOS> 今天天气很好
```

告诉模型：

> 一条新的序列从这里开始。

在 MiniMind：

```
<|im_start|>
```

承担类似作用。

# 9. EOS 是干什么的？

EOS：

```
End Of Sequence
```

例如：

```
<BOS>
我喜欢机器学习
<EOS>
```

为什么必须有？
因为生成模型不仅需要知道：

> 下一个 token 是什么。

还需要学习：

> **什么时候应该停止生成。**

模型生成：

```
我
喜欢
机器
学习
<EOS>
```

看到 EOS 后：

```
stop
```

否则它理论上可以：

```
一直一直一直生成……
```

# 10. PAD 又是什么？
这是为了：

# Batch

假设两个样本：

```
A:
我喜欢AI
```

只有 4 token。
另一个：

```
B:
我真的非常喜欢学习人工智能
```

有 10 token。
你想放进同一个 Tensor：

```
[B,T]
```

但 Tensor 必须规则。
不能：

```
[
    [4个元素],
    [10个元素]
]
```

所以：

```
A:
我 喜欢 AI EOS PAD PAD PAD PAD PAD PAD

B:
我 真的 非常 喜欢 学习 人工 智能 EOS ...
```

补齐相同长度。
这就是：

# Padding

# 11. MiniMind 的 PretrainDataset 现在是怎么干的？
当前源码：

```
class PretrainDataset(Dataset):
```

初始化时：

```
self.samples = load_dataset(
    'json',
    data_files=data_path,
    split='train'
)
```

也就是说预训练数据中的一个样本核心字段是：

```
text
```

随后 `__getitem__()` 中大体进行：

```
text
↓
tokenizer
↓
截断
↓
手动加 BOS
↓
手动加 EOS
↓
PAD 到 max_length
↓
转 torch.long
↓
clone 一份变 labels
↓
PAD 对应 label 改成 -100
```

这就是整个 PretrainDataset。

# 12. 我们现在手动跑一遍

假设：

```
max_length = 8
```

文本：

```
我喜欢AI
```

经过 tokenizer：

```
我       → 52
喜欢     → 381
AI       → 997
```

于是：

```
tokens =
[52, 381, 997]
```

## 加 BOS / EOS
MiniMind：

```
tokens = [
    bos_token_id
] + tokens + [
    eos_token_id
]
```

假设：

```
BOS = 1
EOS = 2
```

得到：

```
[1, 52, 381, 997, 2]
```

## Padding
最大长度 8。
当前长度 5。
所以补：

```
PAD PAD PAD
```

假设：

```
PAD = 0
```

最后：

```
input_ids

[1, 52, 381, 997, 2, 0, 0, 0]
```

shape：

```
[T]
=
[8]
```

# 13. 为什么 tokenizer 时 `max_length - 2`？
MiniMind 写的是类似：

```
max_length=self.max_length - 2
```

因为还需要给：

```
BOS
EOS
```

各留一个位置。
假设：

```
max_length = 340
```

实际原始内容最多：

```
338 token
```

然后：

```
1 BOS
+
338 内容
+
1 EOS
=
340
```

否则你如果内容已经吃满 340，再加：

```
BOS + EOS
```

就变：

```
342
```

爆长度了。

# 14. `truncation=True` 是什么？

假如：

```
max_length = 340
```

但文章经过 tokenizer 得到：

```
1000 token
```

怎么办？
MiniMind 当前 PretrainDataset：

```
只保留前面允许的长度
```

也就是：

# Truncation

截断。
所以：

```
1000 token
↓
338 token
↓
+BOS/EOS
↓
340 token
```

# 15. 这里要区分三个“最大长度”
这个概念以后很容易把你绕晕。
MiniMind 当前可以同时看到三个不同意义的长度：

### Tokenizer 配置

脚本里的：

```
model_max_length = 131072
```

### 模型配置
当前模型自己的：

```
max_position_embeddings
```

默认是 32768。

### 真正某次训练

例如 pretrain：

```
--max_seq_len = 340
```

千万不要看到 tokenizer 支持 131072，就认为：

> MiniMind 训练时每条数据都是 131072 token。

完全不是。
**当前默认 pretrain 甚至只用：**

```
340
```

# 16. 为什么训练长度越长越贵？
这个下一轮 Attention 会详细解释。
现在先知道：

```
T ↑
```

意味着：

```
激活显存 ↑
计算量 ↑
Attention 尤其昂贵
```

所以：

```
max_seq_len
```

是**训练中的关键超参数。**

# 17. PyTorch `Dataset` 到底是什么？

这是你目前尤其需要建立的工程直觉。
很多初学者看到：

```
class PretrainDataset(Dataset)
```

就觉得它是什么神秘的数据框架。
实际上最核心只有：

```
__len__()

__getitem__()
```

## `__len__()`
回答：

> 一共有**多少条样本**？

例如：

```
len(dataset)
```

得到：

```
100000
```

## `__getitem__(i)`
回答：

> 给**我第 i 条训练样本。**

例如：

```
dataset[7]
```

MiniMind 返回：

```
input_ids, labels
```

每一个：

```
[T]
```

所以你可以把 Dataset 想象成：

# 一个实现了“按编号拿训练样本”的对象

仅此而已。

# 18. Dataset 本身不等于 Batch

这是你一定要区分的。

```
dataset[0]
```

**拿的是：**

# **一条**

**例如：**

```
input_ids.shape = [340]
labels.shape    = [340]
```

**还没有：**

```
B
```

**batch 维度。**

# 19. DataLoader 才负责把它们拼成 Batch

例如：

```
dataset[5] → [340]
dataset[8] → [340]
dataset[2] → [340]
...
```

**如果：**

```
batch_size = 32
```

**DataLoader 拼起来：**

```
input_ids:
[32, 340]

labels:
[32, 340]
```

**也就是：**

```
[B,T]
```

**现在就和上一轮接上了。**

# 20. MiniMind 当前 DataLoader 真的就是这么接起来的

当前 pretrain：

```
train_ds = PretrainDataset(...)

...

loader = DataLoader(
    train_ds,
    batch_sampler=...,
    num_workers=...,
    pin_memory=True
)
```

随后：

```
for step, (input_ids, labels) in enumerate(loader):
```

得到 batch，再：

```
model(input_ids, labels=labels)
```

所以完整调用链就是：

```
DataLoader
↓
调用 Dataset.__getitem__
↓
拿 N 条样本
↓
拼成 batch
↓
for 循环拿到
input_ids [B,T]
labels    [B,T]
```

# 21. `num_workers` 是什么？
MiniMind 默认：

```
num_workers = 8
```

粗略理解：

> 用多个 worker 并行准备数据。

因为 GPU 很快。
如果 GPU 每次都：

```
算完一批
↓
等 CPU 读磁盘
↓
等 tokenize
↓
等准备 Tensor
↓
再算
```

GPU 会闲着。
所以可以：

```
worker 1 准备 batch
worker 2 准备 batch
worker 3 准备 batch
...
```

提高吞吐。
现阶段不用研究 multiprocessing 细节。

# 22. `pin_memory=True` 是什么？

你现在只需要知道：

> 当 CPU → GPU 搬数据时，Pinned Memory 往往可以帮助提高数据传输效率。

它是：

```
训练工程优化
```

不是模型数学的一部分。
以后看到不用害怕。

# 23. Pretrain 的 labels 到底长什么样？

刚才：

```
input_ids:

[1, 52, 381, 997, 2, 0, 0, 0]
```

MiniMind：

```
labels = input_ids.clone()
```

于是最开始：

```
labels:

[1, 52, 381, 997, 2, 0, 0, 0]
```

**然后：**

```
labels[
    input_ids == pad_token_id
] = -100
```

**得到：**

```
labels:

[1, 52, 381, 997, 2, -100, -100, -100]
```

# 24. 再加上上一轮的 shift，你就彻底看懂 Pretrain 了
模型内部：

```
x = logits[..., :-1, :]
y = labels[..., 1:]
```

**然后 CrossEntropy，`-100` 被忽略。**
**所以：**

```
input:

BOS   我   喜欢   AI   EOS   PAD   PAD   PAD
```

真正训练关系：

```
BOS
 ↓预测
我

我
 ↓预测
喜欢

喜欢
 ↓预测
AI

AI
 ↓预测
EOS
```

然后：

```
EOS
↓
PAD
```

这里 label 是：

```
-100
```

所以：

> 不算 loss。

# 25. 到这里你已经真正看懂 PretrainDataset
完整过程：

```
JSONL:

{"text": "我喜欢人工智能"}

↓

load_dataset

↓

sample["text"]

↓

Tokenizer

↓

[52,381, ...]

↓

+BOS / EOS

↓

Pad / Truncate

↓

input_ids [T]

↓

clone

↓

PAD → -100

↓

labels [T]

↓

DataLoader

↓

input_ids [B,T]
labels [B,T]

↓

MiniMind

↓

Next Token Prediction
```

# 26. 接下来进入本轮最重要的部分：SFT
你之前一定见过：

```
Pretrain
SFT
```

但我要让你从数据层面真正理解它们的区别。
先一句话：

## Pretrain

让模型学：

> **语言是什么样的。**

## SFT
**Supervised Fine-Tuning。**
**让模型进一步学：**

> **作为 Assistant，面对某种输入时应该如何回答。**

所以：

```
Pretrain:
文本 → 文本

SFT:
指令/对话 → 理想回答
```

# 27. 为什么只有 Pretrain 不够？
假设你拿：

```
Wikipedia
小说
网页
代码
论坛
```

**训练一个 Next Token Predictor。**
**它可能学会了非常好的语言能力。**
但你问：

```
用户：什么是快速排序？
```

它**未必天然知道：**

> **现在我是“助手”，应该停下来组织一个回答。**

**因为 pretraining 数据更多是：**

```
文章续写
代码续写
网页续写
```

而不是：

```
User
↓
Assistant
```

所以**需要 SFT。**

# 28. SFT 数据通常长什么样？

MiniMind 当前的 SFTDataset 读取：

```
conversations
```

里面每条 message 有：

```
role
content
reasoning_content
tools
tool_calls
```

等字段。
简化之后可能是：

```
{
  "conversations": [
    {
      "role": "system",
      "content": "你是一个AI助手"
    },
    {
      "role": "user",
      "content": "什么是快速排序？"
    },
    {
      "role": "assistant",
      "content": "快速排序是一种分治排序算法..."
    }
  ]
}
```

你以前做 API 项目时，这个结构应该非常熟悉：

```
messages
```

现在的区别是：

> 以前你把 messages 发给 DeepSeek/OpenAI API。

而现在：

> **我们要拿这些 messages 自己训练模型。**

这正是你从 AI 应用层往模型层跨的一步。

# 29. 但 Transformer 不认识 `role="user"`

模型只能吃：

```
token IDs
```

模型并不知道 Python dict：

```
{
    "role": "user",
    "content": "你好"
}
```

是什么意思。
所以必须先**把结构化 messages：**

```
system
user
assistant
```

**变成一串普通文本。**
这就是：

# Chat Template

# 30. `apply_chat_template()` 到底在干什么？
MiniMind 的：

```
tokenizer.apply_chat_template(...)
```

会把：

```
[
  system,
  user,
  assistant
]
```

变成类似：

```
<聊天开始>system
你是一个AI助手
<聊天结束>

<聊天开始>user
什么是快速排序？
<聊天结束>

<聊天开始>assistant
快速排序是一种……
<聊天结束>
```

这里我故意用了简化示意。
MiniMind 当前**实际模板使用的主要角色边界是：**

```
<|im_start|>
<|im_end|>
```

同时**还支持：**

```
<think>
tool_call
tool_response
```

等格式。
于是：

```
JSON conversations
```

终于变成：

```
一条普通字符串
```

# 31. 为什么一定要 Chat Template？
因为模型**必须能够区分：**

```
谁说的话？
```

如果直接拼：

```
你是一个AI助手什么是快速排序快速排序是一种...
```

模型**完全不知道：**

```
system 到哪里结束？
user 到哪里结束？
assistant 从哪里开始？
```

而 Chat Template 给它**插入控制 token：**

```
SYSTEM
USER
ASSISTANT
EOS
```

模型**慢慢就能学习：**

> **当看到 USER 内容结束 + ASSISTANT 开始标记以后，我应该生成回答。**

# 32. 这和你用 ChatGPT API 时的 messages 本质是同一层问题
你以前调用：

```
messages = [
    {"role": "system", ...},
    {"role": "user", ...}
]
```

表面上感觉模型直接理解这个 JSON。
其实并不是。
最终都需要：

```
结构化 messages
↓
chat template
↓
线性 token sequence
↓
Transformer
```

只不过 API 把这一步替你藏起来了。
MiniMind 现在让你亲眼看到而已。
这个认知非常重要。

# 33. 接下来是 SFT 最关键的问题

假设：

```
SYSTEM:
你是AI助手。

USER:
2+2是多少？

ASSISTANT:
4。
```

整个序列都会进入模型：

```
system tokens
user tokens
assistant tokens
```

那么：

# 是不是所有 token 都应该算 loss？

MiniMind 的答案是：

# **不是。**

# 34. MiniMind 的 SFTDataset 只让 Assistant 回答部分参与主要语言模型 loss
当前源码先构造：

```
labels = [-100] * len(input_ids)
```

所以一开始：

```
所有位置
=
-100
```

然后它扫描：

```
assistant 开始标记
```

找到 Assistant 回答范围。
只对：

```
Assistant 内容
+
它的结束标记
```

把：

```
labels[j] = input_ids[j]
```

这非常重要。

# 35. 用一个最简例子彻底讲透

我们不考虑真实 token ID，简化为：

```
输入：

<SYS>
你是助手
<USER>
2+2?
<ASSISTANT>
4
<EOS>
<PAD>
```

假设 ID：

```
input_ids:

[10, 11, 20, 21, 30, 40, 2, 0]
```

SFT labels 可能是：

```
[-100,
 -100,
 -100,
 -100,
 -100,
 40,
 2,
 -100]
```

也就是说：

```
SYSTEM      → 不计算 loss
USER        → 不计算 loss
ASSISTANT标记 → 不计算 loss
回答 4      → 计算 loss
EOS         → 计算 loss
PAD         → 不计算 loss
```

# 36. 但这不代表模型“看不到 User”
这是非常容易误解的地方。
虽然：

```
USER tokens 的 label = -100
```

但：

```
USER tokens 仍然在 input_ids 里面
```

所以模型仍然：

# 能看见 User 问题。

只是：

# 不要求模型自己去预测 User 问题。

这是：

```
Attention 输入
```

和：

```
Loss supervision
```

完全不同的两个概念。
一定要分开。

# 37. 用一句最关键的话记忆

SFT：

> **Prompt 负责作为条件（context），Answer 负责提供监督（supervision）。**

即：

```
User:
法国首都是什么？
      ↓
作为条件给模型看

Assistant:
巴黎。
      ↓
拿来算 loss
```

# 38. 为什么不训练模型预测 User？
假如所有 token 都算 loss：

```
system
user
assistant
```

模型都会学习预测。
那么训练目标就包含：

```
预测 system prompt
预测用户说什么
预测 assistant 回答
```

但我们真正关心的是：

> 给定 system + user 后，Assistant 应该输出什么。

因此 MiniMind 采用的是典型的：

# Response-only loss

或者可以理解成：

```
只监督 Assistant token
```

这让优化重点更加集中在：

```
P(answer | prompt)
```

而不是重新把整个对话都当普通语料建模。
注意：

> 并不是所有 SFT 系统都必须这样设计。

确实存在 full-sequence loss 等不同训练方案。
但：

# MiniMind 当前就是 Assistant 部分监督。

# 39. 现在你已经可以真正理解 Pretrain 和 SFT 的核心区别了
两者模型本身甚至可以：

# 完全是同一个 MiniMind

训练循环也极其相似。
当前仓库 `train_pretrain.py` 和 `train_full_sft.py` 都是：

```
for input_ids, labels in loader:

    res = model(
        input_ids,
        labels=labels
    )

    loss = ...

    backward()

    optimizer.step()
```

真正明显变化的一项就是：

# 数据和 labels。

# 40. 这张表非常重要

|                        | Pretrain               | SFT                             |
| ---------------------- | ---------------------- | ------------------------------- |
| 输入数据               | 普通文本               | **对话/指令数据**               |
| 目标                   | **学语言和知识**       | **学会按指令回答**              |
| Input                  | 整个文本               | System + User + Assistant       |
| 哪些 token 通常算 loss | 几乎所有真实文本 token | MiniMind 中主要 Assistant token |
| PAD                    | `-100`                 | `-100`                          |
| **User tokens**        | **不涉及角色**         | **看得到，但不算监督 loss**     |
| **Assistant tokens**   | **不涉及角色**         | **主要监督目标**                |
| 模型架构               | MiniMind               | 还是 MiniMind                   |
| 本质                   | Next Token Prediction  | 依旧是 Next Token Prediction    |

最后一行尤其重要：

# SFT 没有发明新的 Transformer 训练任务。

它仍然：

```
根据前文
预测下一个 token
```

区别只是：

> **哪些 token 的预测错了，我们才惩罚它。**

# 41. 用数学语言理解其实非常漂亮
Pretrain 大致想优化：

$$
-\sum_t \log P(x_t|x_{<t})
$$

即：

> **每个真实 token 基本都算。**

SFT 可以理解为：

$$
-\sum_t m_t \log P(x_t|x_{<t})
$$

其中：

```
m_t = 1
```

说明**这个 token 参与 loss；**

```
m_t = 0
```

说明不参与。
MiniMind 用的实现技巧不是显式：

```
mask × loss
```

而是：

```
label = -100
```

然后：

```
CrossEntropy(ignore_index=-100)
```

效果类似。

# 42. 现在重新理解 `labels`

你以前可能会觉得：

```
input_ids
labels
```

跟图像分类里的：

```
image
class_id
```

差不多。
语言模型不是。
图像分类：

```
image
↓
一个 label
```

例如：

```
cat = 3
```

语言模型：

```
input_ids: [B,T]

labels:    [B,T]
```

因为：

> **每个位置都可能产生一个 Next Token Prediction 监督信号。**

所以一条长度：

```
T=340
```

的序列，理论**上一次 forward 就可以产生几百个训练目标。**
**这就是 Transformer 训练效率高的重要原因之一。**

# 43. 再理解 SFT 多轮对话

假设：

```
User:
1+1?

Assistant:
2

User:
那2+2?

Assistant:
4
```

MiniMind 会把整个多轮对话拼成：

```
USER
1+1?

ASSISTANT
2

USER
那2+2?

ASSISTANT
4
```

然后 label 类似：

```
User1       -100
Question1   -100

Assistant1  -100
Answer1      ✓

User2       -100
Question2   -100

Assistant2  -100
Answer2      ✓
```

也就是说：

> 同一个 conversation 中多个 Assistant 回复都可以提供训练信号。

当前 `generate_labels()` 就是在 token 序列中不断扫描 Assistant 起始标记和结束标记。

# 44. MiniMind 当前还加入了一点数据增强

当前 `pre_processing_chat()` 会**对没有 tools 的普通对话，以一定概率补一个 system prompt；源码默认比例是：**

```
0.2
```

而且**准备了若干中英文 system prompt，例如“你是 MiniMind”“你是一个 AI 助手”等类型。**
**这其实是一个很简单的数据增强思想：**

```
相同任务
+
不同 system 表述
```

帮助模型不要只适应唯一固定 prompt。
这已经开始接近以后你做科研实验时可以思考的东西了：

```
数据配方改变
→
模型行为改变
```

# 45. MiniMind 当前还支持 reasoning / tool 数据
你现在不用深入学 Agent RL，但留一个认识。
SFT 数据结构已经包括：

```
reasoning_content
tools
tool_calls
```

chat template 也处理：

```
<think>
</think>

<tool_call>
...

<tool_response>
...
```

这意味着 MiniMind 当前不只是：

```
User →普通回答
```

也在为：

```
Reasoning
Tool Use
Agent
```

预留统一对话表示。
现在先知道存在即可。
我们后面再学。

# 46. Attention Mask 和 Loss Mask 千万别混

这是第二轮非常重要的一个前置概念。
有：

```
Causal Mask
Attention Mask
Loss Mask
```

三个东西。
不是同一个玩意。

# 47. Causal Mask 是什么？

这是 Decoder-only Transformer 的：

> **不能偷看未来。**

假设：

```
我 喜欢 机器 学习
```

预测：

```
机器
```

时，只能看：

```
我 喜欢
```

不能提前看到：

```
机器 学习
```

所以 attention：

```
位置0 → 看0

位置1 → 看0,1

位置2 → 看0,1,2

位置3 → 看0,1,2,3
```

形成一个三角形 mask。
MiniMind Attention 当前设置：

```
self.is_causal = True
```

并使用 causal attention。
这个我们下一轮详细展开。

# 48. Attention Mask 是什么？

Attention Mask 回答：

> **哪些已有位置根本不应该被注意？**

最常见：

```
PAD
```

例如：

```
我 喜欢 AI EOS PAD PAD PAD
```

理论上可以告诉 attention：

```
1 1 1 1 0 0 0
```

即：

```
真实 token = 1
PAD = 0
```

MiniMind 的 Attention 当前本身**支持传入 `attention_mask`；如果有 mask，会在 attention score 上压掉被屏蔽位置。**

# 49. 但 MiniMind 当前这两个训练 Dataset 没有返回 attention_mask

注意当前：

```
return input_ids, labels
```

并没有：

```
return input_ids, attention_mask, labels
```

训练脚本也只是：

```
model(
    input_ids,
    labels=labels
)
```

所以**训练阶段：**

```
attention_mask = None
```

**至少当前这条标准 pretrain / SFT 训练路径如此。**

# 50. 那 PAD 不就会影响模型吗？

这里需要认真想一下。
MiniMind 的 padding：

```
真实内容
↓
EOS
↓
PAD PAD PAD
```

全部在序列末尾。
而 Decoder 有：

# Causal Mask

因此前面的真实 token：

```
根本不能看到未来 PAD。
```

例如：

```
REAL REAL EOS PAD PAD
 ↑
这个位置只能看左边
```

所以 PAD 不会反过来污染前面真实 token 的预测。
而 PAD 自己所在的位置虽然会计算 hidden state，但：

```
labels = -100
```

这些位置最终：

```
不产生 CrossEntropy loss。
```

因此这一训练设计可以工作。
这是一个很好的：

> **从数据布局 + causal attention + loss mask 联合考虑**

的例子。

# 51. Loss Mask 又是什么？

它回答的不是：

> 模型能不能看。

而是：

> **这个位置预测错了要不要罚。**

MiniMind 使用：

```
label == -100
```

表达：

```
不要罚。
```

所以：

```
User token:
模型可以看
但不算 loss
```

和：

```
PAD:
不希望作为训练目标
因此不算 loss
```

属于 loss 侧的控制。

# 52. 三种 Mask 一张表搞定

| Mask               | 控制什么                   | 典型用途              |
| ------------------ | -------------------------- | --------------------- |
| **Causal Mask**    | **能不能看未来**           | **自回归生成**        |
| **Attention Mask** | **能不能注意某些输入位置** | **PAD 等**            |
| Loss Mask / `-100` | 预测错了算不算损失         | **PAD、User、System** |
| 作用阶段           | Attention / Loss 不同阶段  | 不可混淆              |
| 以后别人跟你说：   |                            |                       |

> “这里 mask 掉了。”

你第一反应不能只是：

> 嗯，mask。

而要问：

# “Attention mask 还是 loss mask？”

这就是开始真正会读训练代码了。

# 53. 现在把 Pretrain 完整数据流画出来

```
pretrain_t2t_mini.jsonl

一行：
{
  "text": "人工智能是..."
}

            ↓

load_dataset

            ↓

sample["text"]

            ↓

tokenizer(
    add_special_tokens=False,
    truncation=True
)

            ↓

内容 token IDs

            ↓

[BOS] + tokens + [EOS]

            ↓

PAD 到 max_seq_len

            ↓

input_ids [T]

            ↓
        clone

labels [T]

            ↓

PAD label → -100

            ↓

DataLoader

            ↓

input_ids [B,T]
labels    [B,T]

            ↓

MiniMind

            ↓

logits [B,T,V]

            ↓

logits[:, :-1]
labels[:, 1:]

            ↓

Cross Entropy

            ↓

loss
```
如果你能自己完整讲出这条链：

> 你的 PyTorch / LLM 训练理解已经比第一轮开始前前进了一大步。

# 54. SFT 完整数据流

```
sft_t2t_mini.jsonl

一行：
{
  "conversations": [
      system,
      user,
      assistant,
      ...
  ]
}

            ↓

pre_processing_chat

            ↓

apply_chat_template

            ↓

一整条对话字符串

            ↓

tokenizer

            ↓

input_ids

            ↓

truncate / pad

            ↓

扫描 Assistant 区域

            ↓

labels

System     → -100
User       → -100
Assistant  → token id
PAD        → -100

            ↓

DataLoader

            ↓

[B,T]

            ↓

MiniMind

            ↓

Next Token Prediction

            ↓

只有 Assistant answer
真正贡献主要 CE loss
```

这条图你以后最好能脱离资料自己画出来。

# 55. 一个非常值得你现在建立的认识

你以前可能觉得：

```
Pretraining
SFT
DPO
RLHF
GRPO
```

**好像是五套完全不同的“模型”。**
**其实不是。**
**很多时候：**

```
Transformer 本体
```

**压根没怎么变。**
**真正变化的是：**

```
数据是什么？

↓

模型输出什么？

↓

loss 怎么定义？

↓

哪些参数更新？
```

这是以后**学习所有 LLM post-training 的总思路。**

# 56. 为什么这个认知对你后面做科研非常重要？

因为真正的研究修改经常不是：

> “我把 Transformer 全部推倒重写。”

而是：

```
换数据选择方式

或

换 token 处理策略

或

换 attention

或

换 loss

或

换训练 schedule

或

换某个 module
```

然后：

```
Baseline
↓
Modification
↓
Controlled Experiment
↓
Metric
↓
Ablation
```

你最终要从 MiniMind 练出来的，就是这种：

> **知道改动发生在训练系统的哪一层。**

# 57. Dataset / DataLoader 你现在最容易犯的 4 个认知错误

## 错误 1

```
Dataset = 一整个 batch
```
错。
Dataset：

```
dataset[i]
→ 一条 sample
```

DataLoader：

```
N 条 sample
→ batch
```

## 错误 2

```
Tokenizer 输出 Embedding
```

错。
Tokenizer：

```
Text
→ token IDs
```

Embedding：

```
token IDs
→ vectors
```

## 错误 3

```
label=-100 意味着模型看不到这个 token
```

错。
它意味着：

> CrossEntropy 忽略这里。

能否看见取决于：

```
attention
```

## 错误 4

```
SFT 是另一种完全不同于 Next Token Prediction 的算法
```

错。
MiniMind SFT 仍然：

```
预测下一个 token
```

只是：

```
只选择部分位置算 loss。
```

# 58. 你现在还需要掌握一个实战读代码方法
以后读：

```
__getitem__()
```

不要一行行被 Python 语法牵着走。
永远做一张：

# Tensor 流水账

例如 MiniMind PretrainDataset：

```
sample['text']
类型：str

↓

tokens
类型：list[int]
长度：<= T-2

↓

+BOS/EOS
类型：list[int]
长度：<= T

↓

padding
类型：list[int]
长度：T

↓

torch.tensor
dtype：torch.long
shape：[T]

↓

labels.clone()
dtype：torch.long
shape：[T]

↓

DataLoader

↓

input_ids
dtype：torch.long
shape：[B,T]

labels
dtype：torch.long
shape：[B,T]
```

以后看任何模型项目都这么干。

# 59. 为什么 `input_ids` 是 `torch.long`？

这个也值得现在理解。
Embedding：

```
nn.Embedding(...)
```

输入的不是：

```
连续浮点特征
```

而是：

```
词表索引。
```

例如：

```
[42, 13, 961]
```

这些必须表示：

```
第42行
第13行
第961行
```

所以是整数类型。
PyTorch 常见：

```
dtype=torch.long
```

即：

```
int64
```

而不是：

```
float32
```

# 60. 为什么 labels 也是 long？
因为 CrossEntropy 接受：

```
模型输出：

logits
float

目标：

class index
integer
```

对一个 token 来说：

```
正确类别 = vocab 中第 1281 个 token
```

所以：

```
label = 1281
```

而不是：

```
[0,0,0,...,1,...,0]
```

MiniMind 没有显式构造 6400 维 one-hot label。
这非常省内存。

# 61. Tokenizer 和模型参数必须匹配

假设旧 tokenizer：

```
token "机器"
→ ID 1281
```

你重新训练一个 tokenizer：

```
token "机器"
→ ID 913
```

但是旧模型的 Embedding 第：

```
1281
```

行已经学会了“机器”的表示。
你现在把“机器”输入成：

```
913
```

模型取：

```
Embedding[913]
```

那已经完全是另一个 token 的含义。
所以：

# Tokenizer 是模型的一部分。

虽然它通常没有神经网络参数。
这正是 MiniMind 作者现在不建议用户随便重新训练 tokenizer 的核心原因之一。

# 62. 词表大小 V 也不是越大越好

假设：

```
V 很大
```

好处：

```
一个 token 能包含更长片段
→ 序列可能更短
```

但坏处：
Embedding：

```
[V,C]
```

LM Head：

```
[C,V]
```

会变大。
最终 logits：

```
[B,T,V]
```

也更大。
因此：

```
V ↑
```

并不是免费的。
MiniMind 这种小模型使用：

```
6400
```

本身就是一种**模型规模和 tokenizer 压缩率之间的取舍。**

# 63. 现在你应该能理解一个之前非常容易疑惑的问题

假设：

```
vocab_size = 6400
```

那么：

```
一个 token
```

并不是：

> 对应一个 logits。

而是：

> **模型在某一个位置，需要给 6400 个候选 token 各打一个 logit。**

即：

```
位置 t：

logits[t]

=
[
  token0的分数,
  token1的分数,
  ...
  token6399的分数
]
```

然后：

```
softmax
```

才形成：

```
下一个 token 的概率分布。
```

所以：

```
一个位置
→ 6400 logits

不是

一个 token
→ 一个 logit
```

这和上一轮的 `[B,T,V]` 正好连起来。

# 64. 这一轮你暂时不用深挖什么

仍然不要分散主线去学：

```
SentencePiece 细节
Unigram LM Tokenizer
WordPiece 数学
BPE tokenizer 工程优化
HuggingFace tokenizers Rust 实现
packing
sequence packing
dynamic padding
bucketing
distributed sampler 原理
```

这些以后需要时再补。
你现在够用的是：

```
BPE
ByteLevel
special tokens
token IDs
padding
truncation
Dataset
DataLoader
chat template
loss mask
SFT labels
```

# 65. 本轮验收标准
你现在最好能不看上文解释下面这些问题。

### Q1

```
Text
→ Tokenizer
→ ?
→ Embedding
→ ?
```
你应该能回答：

```
token IDs
↓
hidden vectors
```

### Q2
为什么：

```
vocab_size=6400
```

会导致：

```
logits.shape[-1]=6400
```

因为：

> 每个位置要给词表中的 6400 个可能 next token 各打一个分数。

### Q3
Dataset 和 DataLoader 区别？
应该能回答：

```
Dataset
→ 定义一条样本怎么取

DataLoader
→ 把多条样本组织成 batch
```

### Q4
为什么 Pretrain：

```
labels ≈ input_ids
```

因为：

> 每个位置都在训练 Next Token Prediction。

PAD 除外。

### Q5

为什么 SFT 的 User token 是：

```
label=-100
```

因为：

> User 作为上下文给模型看，但 MiniMind 的 SFT 不要求模型学习预测 User 内容。

### Q6

```
label=-100
```

是不是表示模型看不到这个 token？

# 不是。

只是：

```
这个位置不算 CrossEntropy loss。
```

### Q7
为什么：

```
<BOS> 我 喜欢 AI <EOS>
```

必须有 EOS？
因为：

> 模型需要学会何时停止生成。

### Q8
为什么 tokenizer 不能随便换？
因为：

```
token ↔ ID
```

映射变了以后：

```
Embedding 第 i 行原本对应的意义
```

也就彻底错位。

# 66. 到这里，你的前两轮知识已经接成一整条了

现在你应该已经可以理解：

```
"什么是Transformer？"

↓ Chat Template

<USER>什么是Transformer？</USER>
<ASSISTANT>...

↓ Tokenizer

[1, 942, 51, 81, ...]

↓ Dataset

input_ids [T]
labels    [T]

↓ DataLoader

input_ids [B,T]
labels    [B,T]

↓ Embedding

[B,T,C]

↓ Transformer

[B,T,C]

↓ LM Head

[B,T,V]

↓ Shift

logits[:, :-1]
labels[:, 1:]

↓ CrossEntropy

只计算非 -100 的位置

↓ loss

↓ backward

↓ optimizer.step()
```

**这已经是一个真正的 LLM 训练闭环了。**
你现在欠缺的最大一块，就变成了中间那个此前被我们故意当作黑箱的：

# `[B,T,C] → Transformer → [B,T,C]`

所以下一轮正好进入整个 MiniMind 最核心的一轮：

# **第 3 轮：亲手拆 MiniMind Transformer**

我会严格按照当前 `model_minimind.py` 的真实执行顺序来讲：

```
input_ids
↓
Embedding
↓
RMSNorm
↓
Q / K / V Linear
↓
Tensor reshape
↓
QK-Norm
↓
RoPE
↓
GQA / repeat_kv
↓
Scaled Dot-Product Attention
↓
Causal Mask
↓
Output Projection
↓
Residual
↓
RMSNorm
↓
SwiGLU / FFN
↓
Residual
↓
重复 8 层
↓
Final RMSNorm
↓
LM Head
↓
6400 logits
```

而且这一轮我会特别针对你现在的水平，把**每一个关键 Tensor 的 shape 都写出来**。这样你学完后应该能真正打开 `model_minimind.py`，自己沿着 forward 从头走到尾，而不再只是“我知道 Transformer 的结构图”。

# 第三轮

考虑到你现在是“Transformer 概念已经学过，但 PyTorch 独立模型实现还比较弱”的状态，这一轮重点不是重新讲 Attention 定义，而是训练你把 **公式 ↔ Tensor Shape ↔ PyTorch 源码** 对上。

# 第三轮：真正读懂 `model_minimind.py`

这一轮结束后，我希望你看到 MiniMind 模型代码时，不再是：

> “这应该是 Attention……这应该是 FFN……”

而是能自己追踪：

> 这个 Tensor 当前是什么 shape → 为什么这样变形 → 这一层到底计算了什么 → 为什么这么设计 → 去掉会怎样。

# 1. 先把整个 MiniMind 模型压缩成 6 行
当前默认配置是：

```
vocab_size              V = 6400
hidden_size             C = 768

num_hidden_layers           = 8
num_attention_heads      H = 8
num_key_value_heads    Hkv = 4

head_dim                D = 96
```

因为：

$$
D=\frac{C}{H}=\frac{768}{8}=96
$$

当前源码就是这些默认值。
于是整个 MiniMind 可以先粗暴理解成：

```
input_ids
[B, T]

↓ Embedding

[B, T, 768]

↓ Transformer Block × 8

[B, T, 768]

↓ RMSNorm

[B, T, 768]

↓ LM Head

[B, T, 6400]

↓ CrossEntropy

loss
```

再把一个 Transformer Block 展开：

```
                   ┌────────────────────┐
                   │                    ↓
x ─→ RMSNorm ─→ Attention ─→ + ─→ RMSNorm ─→ SwiGLU ─→ +
│                               ↑                      ↑
└──────── residual ─────────────┘                      │
                                └──── residual ────────┘
```

数学上近似就是：

$$
x'=x+\operatorname{Attention}(\operatorname{RMSNorm}(x))
$$

$$
x''=x'+\operatorname{SwiGLU}(\operatorname{RMSNorm}(x'))
$$
这就是当前 `MiniMindBlock` 真正在做的事情。
所以**第三轮其实就是搞懂四个核心东西：**

1. **RMSNorm**
2. **RoPE**
3. **GQA Attention**
4. **SwiGLU FFN**
**然后把它们拼起来。**

# 2. 从 Embedding 开始：`[B,T] → [B,T,768]`

源码：

```
self.embed_tokens = nn.Embedding(
    config.vocab_size,
    config.hidden_size
)
```

也就是：

```
nn.Embedding(6400, 768)
```

它内部就是一张：

```
[6400, 768]
```

的**可训练参数表。**
假设：

```
token_id = 125
```

那么：

```
embedding(token_id)
```

本质就是：

```
取 embedding_table 的第 125 行
```

得到：

```
[768]
```

的向量。
因此：

```
input_ids
[B,T]

例如：

[32,512]

↓ Embedding

hidden_states
[32,512,768]
```

这里有个非常重要的认识：

> **Transformer 从这里开始，就再也不知道“token 125”这个整数了。**

Transformer 后面看到的全都是连续向量。

# 3. RMSNorm：为什么现代 LLM 不再大量使用 LayerNorm？

你以前 Transformer 学的是：

```
LayerNorm
```

MiniMind 当前则使用：

```
class RMSNorm(nn.Module):
```

源码核心只有：

```
x * torch.rsqrt(
    x.pow(2).mean(-1, keepdim=True) + eps
)
```

然后再乘：

```
self.weight
```

## 3.1 先回忆 LayerNorm
对于一个 token：

```
x = [x1, x2, ..., x768]
```

LayerNorm 大致：

$$
\hat x=
\frac{x-\mu}
{\sqrt{\sigma^2+\epsilon}}
$$

也就是：

```
先减均值
↓
再除标准差
```

# 4. RMSNorm 做了什么？
RMS 是：

> Root Mean Square，均方根。

计算：

$$
RMS(x)
=
\sqrt{
\frac{1}{C}
\sum_i x_i^2
}
$$

然后：

$$
RMSNorm(x)
=
\frac{x}
{\sqrt{\operatorname{mean}(x^2)+\epsilon}}
\odot w
$$

所以跟 LayerNorm 最大的区别：

### LayerNorm

```
x
↓
减 mean
↓
除 standard deviation
```

### RMSNorm

```
x
↓
不减 mean
↓
只控制整体数值尺度
```
它**更加简单**。

# 5. MiniMind 中 RMSNorm 到底沿哪个维度做？

注意代码：

```
x.pow(2).mean(-1, keepdim=True)
```

这里：

```
-1
```

表示：

> **最后一个维度。**

假设：

```
x.shape
=
[B,T,C]
=
[32,512,768]
```

那么 RMSNorm 是：

```
每一个 token
自己的 768 个 hidden features
单独做归一化
```

也就是：

```
token 1 的 768 维
→ normalize

token 2 的 768 维
→ normalize

token 3 的 768 维
→ normalize
```

不会：

```
token 1 和 token 2 混着算
```

也不会：

```
batch 1 和 batch 2 混着算
```

这一点一定要建立。

# 6. 为什么这里出现 `x.float()`？

MiniMind：

```
self.norm(x.float())
```

最后：

```
.type_as(x)
```

**假设模型使用：**

```
BF16
```

**但是 RMS：**

```
square
→ mean
→ sqrt
```

**属于比较容易受到数值精度影响的操作。**
所以：

```
BF16 x

↓ 临时转 FP32

RMSNorm

↓ 再转回

BF16
```

也就是说：

> **参数可以低精度训练，但某些数值敏感操作临时使用 FP32。**

以后你看大模型源码，这种写法非常常见。

# 7. MiniMind 其实不只有普通 RMSNorm

这里特别值得你注意。
**Attention 初始化还有：**

```
self.q_norm = RMSNorm(self.head_dim)
self.k_norm = RMSNorm(self.head_dim)
```

然后：

```
xq = self.q_norm(xq)
xk = self.k_norm(xk)
```

也就是说 MiniMind 还有：

# QK-Norm

不是只有：

```
Transformer Block 前面的 RMSNorm
```

还会**专门对：**

```
Q
K
```

**做归一化。**
而**这里维度不是：**

```
768
```

**而是：**

```
head_dim = 96
```

**因为这时候：**

```
Q.shape = [B,T,8,96]
```

**每一个 attention head 单独归一化自己的 96 维。**
**你现在先把作用理解成：**

> **控制 Q/K 数值尺度，让 Attention score 更稳定。**

**因为 Attention score 是：**
$$
QK^T
$$

如果 **Q、K 数值越来越大：**

```
QK^T
```

**也容易越来越大。**
**Softmax 就可能非常尖锐。**
QK-Norm 相当于进一步**稳定 Attention**。

# 8. 接下来真正进入 Attention

现在输入：

```
x
[B,T,768]
```

普通 **Multi-Head Attention 大家已经熟悉：**

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

然后：

$$
Attention(Q,K,V)
=
softmax\left(
\frac{QK^T}{\sqrt d}
\right)V
$$

但是 MiniMind **用的不是普通 MHA。**
而是：

# GQA：Grouped Query Attention

这是这一轮**最重要的现代 LLM 前置之一。**

# 9. 先看普通 MHA 是什么

假设：

```
8 heads
```

普通 MHA：

```
Q：8 heads
K：8 heads
V：8 heads
```

即：

```
Q0 K0 V0
Q1 K1 V1
Q2 K2 V2
...
Q7 K7 V7
```

**每一个 Q head 都有自己的 K 和 V。**

# 10. MQA 又是什么？

**Multi Query Attention 更极端：**

```
Q：8 heads

K：1 head
V：1 head
```

**即：**

```
Q0 ┐
Q1 │
Q2 │
...├─ 共用 K0/V0
Q7 ┘
```

**优势：**

> **K/V 少很多，因此 KV Cache 很小。**

但是**共享得太狠，模型能力可能受到影响。**
**于是出现折中：**

# GQA

# 11. MiniMind 的 GQA 是怎样的？
当前：

```
Q heads = 8

KV heads = 4
```

所以：

```
2 个 Q head
共享一组 K/V
```

大概：

```
Q0 ┐
   ├→ K0 V0
Q1 ┘

Q2 ┐
   ├→ K1 V1
Q3 ┘

Q4 ┐
   ├→ K2 V2
Q5 ┘

Q6 ┐
   ├→ K3 V3
Q7 ┘
```

这就是：

```
Grouped Query Attention
```

**MiniMind：**

```
self.n_rep = (
    self.n_local_heads
    //
    self.n_local_kv_heads
)
```

**这里：**

```
8 / 4 = 2
```

所以：

```
n_rep = 2
```

# 12. 现在我们开始严格追 Attention Shape
输入：

```
x
[B,T,768]
```

先：

```
xq = self.q_proj(x)
xk = self.k_proj(x)
xv = self.v_proj(x)
```

## Q Projection
源码：

```
nn.Linear(
    hidden_size,
    num_attention_heads * head_dim
)
```

也就是：

```
768 → 8 × 96
768 → 768
```

因此：

```
Q：

[B,T,768]
```

然后：

```
view(B,T,8,96)
```

得到：

```
Q
[B,T,8,96]
```

# 13. K/V 可不一样
K：

```
nn.Linear(
    768,
    4 × 96
)
```

即：

```
768 → 384
```

所以：

```
K：

[B,T,384]

→ reshape

[B,T,4,96]
```

V 同理：

```
V：

[B,T,4,96]
```

于是现在：

```
Q = [B,T,8,96]

K = [B,T,4,96]

V = [B,T,4,96]
```

这几行源码非常值得以后自己重新读一次。

# 14. 注意：这里参数量也下降了

**普通 MHA：**

```
Q projection：768 → 768
K projection：768 → 768
V projection：768 → 768
```

**GQA：**

```
Q：768 → 768

K：768 → 384
V：768 → 384
```

**所以 K/V 参数也少了。**
**但这还不是 GQA 最大的价值。**
**最大的价值在：**

# KV Cache

稍后讲。

# 15. 在算 Attention 之前，为什么还要 RoPE？

Transformer 本身：

> 不知道 token 的顺序。

比如：

```
我 喜欢 你
```

和：

```
你 喜欢 我
```

如果完全没有位置编码，仅从 Attention 的数学结构来看，顺序信息是不充分的。
你最开始学习 Transformer 时可能见过：

```
Sinusoidal Position Encoding
```

大概：

```
Embedding
+
Position Embedding
```

但是**现代 LLM 很常见的方案是：**

# **RoPE**

**Rotary Position Embedding。**
**MiniMind 当前就在用它。**

# 16. RoPE 与传统 Position Embedding 最大区别

传统方案可以粗暴理解成：

```
token embedding
+
position embedding
```

例如：

```
“我”的 embedding
+
“位置 3”的 embedding
```

然后送进 Transformer。
RoPE 不一样。
它**不是简单：**

```
位置向量 + hidden_states
```

**而是：**

> **根据 token 的位置，把 Q 和 K 向量旋转一个角度。**

注意：

```
Q：旋转
K：旋转

V：不旋转
```

# 17. “旋转”到底是什么意思？
先假设一个特别简单的二维向量：

$$
(x_1,x_2)
$$

**二维平面上旋转 $\theta$：**

$$
x'_1=x_1\cos\theta-x_2\sin\theta
$$

$$
x'_2=x_1\sin\theta+x_2\cos\theta
$$
**例如：**

```
[1,0]
```

**旋转 90°：**

```
[0,1]
```

**RoPE 做的事情本质类似。**
只不过真实的：

```
Q head = 96 维
```

**会把这些维度按照约定配对旋转。**

# 18. 为什么“旋转 Q/K”能表示位置？

这一点不要死背推导。
掌握核心直觉：
假设：

```
token A 位于 position 5
token B 位于 position 8
```

那么：

```
Q_A
```

会按照：

```
position 5
```

旋转。
而：

```
K_B
```

会按照：

```
position 8
```

旋转。
之后计算：

$$
Q_AK_B^T
$$

**这个点积就自然包含了：**

```
5 和 8 的位置关系
```

也就是：

```
relative position = 8 - 5
```

所以 RoPE 一个**非常优雅的地方在于：**

> **Attention 的 QK 相似度里面自然带入相对位置信息。**

你现在理解到这里已经足够读 MiniMind。

# 19. MiniMind 为什么提前计算 sin / cos？

模型初始化：

```
freqs_cos, freqs_sin = precompute_freqs_cis(...)
```

**然后：**

```
register_buffer("freqs_cos", ...)
register_buffer("freqs_sin", ...)
```

**因为：**

```
position 0 对应哪些 cos/sin
position 1 对应哪些 cos/sin
...
```

**这些东西不是模型每一次 forward 都需要重新算。**
所以提前算好：

```
position
×
frequency
→
cos / sin
```

**保存起来。**
**而且：**

```
cos/sin
```

**不是训练出来的参数。**
**因此不是：**

```
nn.Parameter
```

**而是：**

```
register_buffer
```

**这正好对应我们第一轮学过的：**

```
Parameter
→ optimizer 要更新

Buffer
→ 模型运行要使用
→ 但 optimizer 不更新
```

现在这个**概念已经真正和源码对应上了。**

# 20. Attention 中的执行顺序现在是

我们已经得到：

```
Q [B,T,8,96]

K [B,T,4,96]

V [B,T,4,96]
```

首先：

```
QK Norm
```

变成：

```
Q [B,T,8,96]
K [B,T,4,96]
```

shape 不变。
然后：

```
RoPE
```

还是：

```
Q [B,T,8,96]
K [B,T,4,96]
```

注意：

> **RMSNorm 和 RoPE 都不会改变 shape。**

这是读模型代码时一个非常重要的习惯：

```
这个操作改变内容？
还是改变 shape？
```

很多模型模块：

```
Norm
Activation
Dropout
RoPE
```

都：

```
改变数值
但不改变 shape
```

# 21. 然后出现 `repeat_kv`
现在有一个问题：

```
Q 有 8 heads

K/V 只有 4 heads
```

但是**最后算：**

$$
QK^T
$$

**head 数得对应。**
所以 MiniMind：

```
repeat_kv(xk, self.n_rep)
repeat_kv(xv, self.n_rep)
```

其中：

```
n_rep = 2
```

于是：

```
K：

[B,T,4,96]

↓

repeat

[B,T,8,96]
```

V：

```
[B,T,4,96]

↓

[B,T,8,96]
```

逻辑上就是：

```
K0 → K0 K0
K1 → K1 K1
K2 → K2 K2
K3 → K3 K3
```

注意一个容易误解的点：

> **GQA 并不是重新计算出了 8 个不同的 K。**

而是：

```
4 个真实 K heads
↓
逻辑共享
↓
供 8 个 Q heads 使用
```

源码的 `repeat_kv()` 就是在完成这一步。

# 22. 接下来 transpose

现在：

```
Q [B,T,8,96]
```

被：

```
transpose(1,2)
```

变成：

```
Q [B,8,T,96]
```

K：

```
[B,8,T,96]
```

V：

```
[B,8,T,96]
```

为什么？
**因为 PyTorch Attention 计算希望：**

```
Batch
Head
Sequence
HeadDim
```

**排列起来更方便。**
**所以你以后看到：**

```
.transpose(1,2)
```

不要觉得玄学。
它只是：

```
[B,T,H,D]

↓

[B,H,T,D]
```

# 23. 终于进入真正的 Attention 数学
现在：

```
Q [B,H,T,D]
K [B,H,T,D]
```

计算：

```
xq @ xk.transpose(-2, -1)
```

K：

```
[B,H,T,D]

↓

transpose

[B,H,D,T]
```

矩阵乘：

```
[B,H,T,D]
@
[B,H,D,T]

↓

[B,H,T,T]
```

得到：

# Attention Scores

也就是：

```
scores[b][h][i][j]
```

表示：

> 第 b 个样本、第 h 个 head 中，第 i 个 token 对第 j 个 token 的关注分数。

# 24. 这里终于理解为什么除 `sqrt(head_dim)`
源码：

```
scores = (
    xq @ xk.transpose(-2,-1)
) / math.sqrt(self.head_dim)
```

这里：

```
head_dim = 96
```

所以除：

$$
\sqrt{96}
$$

你以前学 Transformer 已经见过。
直觉上：

```
D 越大

Q·K
=
q1k1 + q2k2 + ... + qDkD

数值尺度容易越来越大
```

Softmax 如果输入：

```
[1,2,3]
```

还比较平滑。
如果变成：

```
[10,20,30]
```

基本**会极端集中在 30。**
**所以：**

$$
\frac1{\sqrt D}
$$

**用于控制 score 的尺度。**

# 25. Causal Mask 是什么？

这是 Decoder-Only LLM 必须掌握的东西。
假设序列：

```
<BOS> 我 喜欢 学习 AI
```

预测：

```
“喜欢”
```

的时候模型绝对不能偷看：

```
学习 AI
```

否则**训练时：**

> **答案都提前看到了。**

所以 Attention 必须：

```
position 0 只能看 0

position 1 只能看 0,1

position 2 只能看 0,1,2

...
```

Attention Matrix：

```
        K0   K1   K2   K3

Q0      ✓    ✗    ✗    ✗

Q1      ✓    ✓    ✗    ✗

Q2      ✓    ✓    ✓    ✗

Q3      ✓    ✓    ✓    ✓
```

**被禁止的位置：**

```
score = -∞
```

**经过 softmax：**
$$
e^{-\infty}=0
$$

于是**概率就是 0。**
**MiniMind fallback Attention 里就是：**

```
triu(1)
```

**产生上三角 mask，再填入：**

```
-inf
```

**这就是：**

# Causal Self-Attention

# 26. Softmax 后发生什么？
现在：

```
scores
[B,H,T,T]
```

**softmax：**

```
attention_weights
[B,H,T,T]
```

例如某个 token：

```
[0.05, 0.10, 0.70, 0.15]
```

意思大致：

```
关注 token0：5%
关注 token1：10%
关注 token2：70%
关注 token3：15%
```

然后：

$$
AttentionWeights\times V
$$

shape：

```
[B,H,T,T]
@
[B,H,T,D]

↓

[B,H,T,D]
```

即：

```
[B,8,T,96]
```

这一步的直觉就是：

> **根据 Attention 权重，把其他 token 的 Value 信息加权汇总到当前 token。**

# 27. 最后把 8 个 Head 拼回来
现在：

```
[B,8,T,96]
```

**先：**

```
transpose(1,2)
```

**变：**

```
[B,T,8,96]
```

**然后：**

```
reshape(B,T,-1)
```

**变：**

```
[B,T,768]
```

因为：

$$
8\times96=768
$$

**再经过：**

```
o_proj
```

**依然：**

```
[B,T,768]
```

**所以一个 Attention 模块：**

```
输入：

[B,T,768]

↓

各种 QKV / Attention 操作

↓

输出：

[B,T,768]
```

这很重要，因为**这样才能做：**

```
hidden_states += residual
```

# 28. 为什么 residual 两边 shape 必须一样？
假设：

```
residual

[B,T,768]
```

Attention 如果输出：

```
[B,T,384]
```

那么：

```
attention_output + residual
```

**根本没法加。**
所以 Transformer Block 的主体模块通常遵循：

```
[B,T,C]

↓

某个复杂操作

↓

[B,T,C]
```

中间可以：

```
768 → 2432 → 768
```

或者：

```
768 → Q/K/V → 768
```

但**最终必须回到：**

```
C=768
```

**方便 residual。**

# 29. 现在我们正式理解 Residual Connection

MiniMind：

```
residual = hidden_states

hidden_states = self_attn(
    RMSNorm(hidden_states)
)

hidden_states += residual
```

也就是：

$$
x'=x+Attention(Norm(x))
$$

它有两个核心意义。
第一：

```
Attention 不需要重新构造全部信息
```

而**只需要学：**

> **我要在原来的 x 上补充什么？**

第二：
**梯度有一条非常干净的路径：**

```
后面
↑
+
↑
x
```

**有利于深网络训练。**
这也是你以前学 ResNet 时：
$$
F(x)+x
$$

同一个核心思想。
所以：

> **ResNet 的残差思想和 Transformer Residual Connection 本质是同一家族的东西。**

这正是你经典 DL 知识向 LLM 的迁移。

# 30. MiniMind 用的是 Pre-Norm

观察：

```
self.self_attn(
    self.input_layernorm(hidden_states)
)
```

也就是：

```
先 Norm
↓
再 Attention
↓
再 Residual
```

因此：

$$
x+Attention(Norm(x))
$$

这叫：

# Pre-Norm

而**原始 Transformer 经典结构常被写成：**

$$
Norm(x+Attention(x))
$$

**也就是：**

# **Post-Norm**

**现在很多 LLM 更偏向 Pre-Norm。**
你现在不需要研究所有理论细节，只需要掌握：

> **MiniMind 的 Norm 在子层前面，而 residual 在子层输出后相加。**

以后你读模型结构，这个问题值得第一时间检查。

# 31. Attention 讲完，进入第二个子层：FFN

MiniMind：

```
class FeedForward(nn.Module)
```

但是它**并不是你早期 Transformer 学过的简单：**

```
Linear
↓
ReLU
↓
Linear
```

**而是：**

# SwiGLU

当前**核心：**

```
down_proj(
    silu(gate_proj(x))
    *
    up_proj(x)
)
```

这一行**非常值得真正理解。**

# 32. 先看看 Intermediate Size

MiniMind 当前默认：

```
intermediate_size =
ceil(hidden_size * pi / 64) * 64
```

`hidden_size=768`。
大约：

$$
768\pi\approx2413
$$

**向上对齐 64 的倍数：**

```
2432
```

所以 **FFN 内部：**

```
768

↓

2432

↓

768
```

# 33. 普通 FFN
如果是最简单版本：

```
x
[B,T,768]

↓

Linear

[B,T,2432]

↓

Activation

[B,T,2432]

↓

Linear

[B,T,768]
```

但 **SwiGLU 多了一条支路。**

# 34. MiniMind 的 SwiGLU

第一条：

```
gate_proj(x)
```

shape：

```
[B,T,768]
↓
[B,T,2432]
```

然后：

```
SiLU(...)
```

得到：

```
[B,T,2432]
```

第二条：

```
up_proj(x)
```

：

```
[B,T,768]
↓
[B,T,2432]
```

然后**二者：**

```
*
```

**逐元素相乘：**

```
[B,T,2432]
*
[B,T,2432]

↓

[B,T,2432]
```

最后：

```
down_proj
```

：

```
[B,T,2432]

↓

[B,T,768]
```

所以完整公式：

$$
FFN(x)
=
W_{down}
\left[
SiLU(W_{gate}x)
\odot
(W_{up}x)
\right]
$$

# 35. 为什么叫 Gate？
因为：

```
SiLU(gate_proj(x))
```

可以粗略理解成：

> 每一个 feature **应该放多少信息过去**？

而：

```
up_proj(x)
```

是真正的信息分支。
两个相乘：

```
gate
×
content
```

有点像：

> **门控多少信息通过。**

这和你以前学：

```
LSTM gate
GRU gate
```

虽然具体数学不一样，但“门控”的直觉是类似的。

# 36. `SiLU` 是什么？

SiLU：

$$
SiLU(x)=x\sigma(x)
$$

其中：

$$
\sigma(x)
$$

就是 sigmoid。
所以：

```
x 很负
→ 被压低

x 接近 0
→ 平滑变化

x 很正
→ 接近线性通过
```

MiniMind 配置：

```
hidden_act = 'silu'
```

然后：

```
ACT2FN[config.hidden_act]
```

获得对应激活函数。

# 37. Attention 和 FFN 到底分别负责什么？

这是非常值得形成的直觉。

## Attention

主要做：

> **不同 token 之间的信息交换。**

例如：

```
小明把书给了小红，因为她喜欢阅读
```

“她”需要关联：

```
小红
```

Attention 可以从其他 token 获取信息。

## FFN / MLP

主要是在：

> **每一个 token 自己的 hidden representation 内进行非线性特征变换。**

它不会像 Attention 那样直接做：

```
token i ↔ token j
```

可以粗略记：

```
Attention
→ token 与 token 沟通

FFN
→ 每个 token 自己加工信息
```

两者交替：

```
沟通
→ 加工
→ 沟通
→ 加工
...
```

8 层不断重复。
这个直觉非常有用。

# 38. 现在一个完整 MiniMindBlock 你应该看懂了

源码逻辑：

```
residual = hidden_states

hidden_states = Attention(
    RMSNorm(hidden_states)
)

hidden_states += residual

hidden_states = hidden_states + MLP(
    RMSNorm(hidden_states)
)
```

展开：

```
x
[B,T,768]

│
├──────────── residual ───────────────┐
│                                     │
↓                                     │
RMSNorm                               │
[B,T,768]                             │
↓                                     │
GQA + RoPE                            │
[B,T,768]                             │
↓                                     │
+ ←───────────────────────────────────┘
↓
x'
[B,T,768]

│
├──────────── residual ───────────────┐
│                                     │
↓                                     │
RMSNorm                               │
[B,T,768]                             │
↓                                     │
SwiGLU                                │
768 → 2432 → 768                      │
↓                                     │
+ ←───────────────────────────────────┘
↓
x''
[B,T,768]
```

这个图建议你真正记住。

# 39. 然后重复 8 次

MiniMind：

```
self.layers = nn.ModuleList([
    MiniMindBlock(...)
    for ...
])
```

当前默认：

```
8 layers
```

forward：

```
for layer in self.layers:
    hidden_states = layer(hidden_states)
```

所以：

```
Embedding

[B,T,768]

↓

Block 0

[B,T,768]

↓

Block 1

[B,T,768]

↓

...

↓

Block 7

[B,T,768]
```

注意：

> Transformer 越堆越深，但 **hidden shape 根本不用改变。**

**变化的是：**

```
向量里面装的“信息”
```

而不是 Tensor 大小。

# 40. 最后还有一次 RMSNorm

所有 block 结束：

```
hidden_states = self.norm(hidden_states)
```

所以：

```
[B,T,768]

↓ final RMSNorm

[B,T,768]
```

然后 `MiniMindModel` 返回：

```
hidden_states
```

注意：

> 此时还没有得到“每个词的概率”。

它只是模型对每个 token 得到的：

```
768维 hidden representation
```

# 41. 真正变成 logits 的是 LM Head
外层：

```
self.lm_head = nn.Linear(
    hidden_size,
    vocab_size,
    bias=False
)
```

也就是：

```
768 → 6400
```

因此：

```
hidden_states
[B,T,768]

↓

lm_head

↓

logits
[B,T,6400]
```

你**第一轮已经知道：**

```
6400
```

**并不是：**

> **一个 token 有 6400 个 logits。**

准确说应该是：

> **每一个位置，都有针对词表中 6400 个候选 token 的 6400 个 logits。**

所以：

```
logits[b,t,v]
```

含义：

> 第 `b` 个样本，在第 `t` 个位置，对词表第 `v` 个 token 给出的分数。

# 42. 一个非常漂亮的细节：Weight Tying
MiniMind 默认：

```
tie_word_embeddings = True
```

然后：

```
self.model.embed_tokens.weight
=
self.lm_head.weight
```

这个设计值得你理解。
**Embedding 参数：**

```
[V,C]
=
[6400,768]
```

而：

```
nn.Linear(768,6400)
```

**它的 weight 在 PyTorch 内部也是：**

```
[6400,768]
```

**发现了吗？**
**shape 完全一致。**
于是 **MiniMind 让：**

```
输入 Embedding
```

**和：**

```
输出 LM Head
```

**共用同一组权重。**
直觉上：

```
Embedding：

token
→ hidden vector
```

而 **LM Head：**

```
hidden vector
→ 哪个 token
```

**一个像：**

```
词 → 向量
```

**一个像：**

```
向量 → 词
```

**所以让它们共享权重是很自然的设计。**
同时还能：

> **减少参数量。**

对于 MiniMind 这种小模型尤其有价值。

# 43. 现在把整个 Forward 彻底串起来

假设：

```
B = 2
T = 5
```

输入：

```
input_ids
[2,5]
```

## Step 1：Embedding

```
[2,5]

↓

[2,5,768]
```

## Step 2：Block 0 Attention
Q：

```
[2,5,768]

↓

[2,5,8,96]
```

K：

```
[2,5,768]

↓

[2,5,4,96]
```

V：

```
[2,5,4,96]
```

## Step 3：QK Norm + RoPE

```
Q [2,5,8,96]

K [2,5,4,96]

V [2,5,4,96]
```

shape 不变。

## Step 4：GQA repeat

```
K/V

[2,5,4,96]

↓

[2,5,8,96]
```

## Step 5：transpose

```
Q/K/V

[2,8,5,96]
```

## Step 6：Attention Score

```
Q @ K^T

[2,8,5,96]
@
[2,8,96,5]

↓

[2,8,5,5]
```

## Step 7：softmax × V

```
[2,8,5,5]
@
[2,8,5,96]

↓

[2,8,5,96]
```

## Step 8：拼 Heads

```
[2,8,5,96]

↓

[2,5,768]
```

## Step 9：Residual

```
[2,5,768]
+
[2,5,768]

↓

[2,5,768]
```

## Step 10：SwiGLU

两路：

```
[2,5,768]
→
[2,5,2432]
```

乘起来，再：

```
[2,5,2432]

↓

[2,5,768]
```

Residual：

```
[2,5,768]
```

然后重复：

```
8 Blocks
```

shape 始终：

```
[2,5,768]
```

最后：

```
lm_head

[2,5,768]

↓

[2,5,6400]
```

这就是：

```
logits
```

如果你现在已经能自己顺下来这整个过程，那么你对 Transformer 的理解已经开始从：

> “概念理解”

进入：

> **“实现级理解”。**

# 44. 现在再补一个非常重要的东西：KV Cache
这是你以后做 LLM inference 必须理解的。
假设已经生成：

```
我 喜欢 学习
```

现在要生成下一个 token。
如果没有 KV Cache：
生成：

```
我
```

算一次：

```
Q K V
```

生成：

```
喜欢
```

又把：

```
我 喜欢
```

全部重新算一遍 QKV。
生成：

```
学习
```

又把：

```
我 喜欢 学习
```

全部重新算。
大量重复计算。

# 45. 哪些东西其实可以保存？

对于**之前 token：**

```
我 喜欢 学习
```

**它们的：**

```
K
V
```

**已经算完了。**
**下一次生成：**

```
AI
```

**时根本没必要重新算旧 token 的 K/V。**
所以：

```
过去 token 的 K
过去 token 的 V
```

保存起来。
这就是：

# KV Cache

# 46. 为什么不叫 QKV Cache？
这是个很好的理解检查题。
当前 token 要计算 Attention：

$$
Q_{current}
K_{all}
V_{all}
$$

**旧 token 的：**

```
Q_old
```

**以后根本不会再作为“当前 query”使用。**
**但是旧 token 的：**

```
K_old
V_old
```

**必须不断被以后新 token 查询。**
**所以：**

```
Q
→ 用一次就够了

K/V
→ 未来每个 token 都可能需要
```

因此叫：

# KV Cache

而不是 QKV Cache。

# 47. MiniMind 当前 KV Cache 在哪里？

Attention：

```
if past_key_value is not None:
    xk = torch.cat([past_key_value[0], xk], dim=1)
    xv = torch.cat([past_key_value[1], xv], dim=1)
```

**然后：**

```
past_kv = (xk, xv)
```

**也就是：**

```
旧 K
+
新 K

↓

完整 K
```

**V 同理。**
**注意 MiniMind 保存的是 GQA 尚未 `repeat_kv` 的：**

```
K/V：
[B,past_length,4,96]
```

而**不是：**

```
[B,past_length,8,96]
```

**这点很漂亮。**

# 48. 现在终于真正理解 GQA 为什么重要

如果普通 MHA：

```
K heads = 8
V heads = 8
```

KV Cache 要存：

```
8 组 K
+
8 组 V
```

MiniMind GQA：

```
K heads = 4
V heads = 4
```

因此这里 **KV Cache 大约只有普通 8-KV-head MHA 的：**

$$
\frac48=\frac12
$$

**也就是：**

> **减少约一半的 K/V Cache head 存储量。**

**对于上下文很长的 LLM：**

```
32K
64K
128K
...
```

**KV Cache 可以非常巨大。**
**所以现代 LLM 非常在意：**

```
MHA
→ GQA / MQA
```

这已经不只是理论结构问题，而是真实推理成本问题。

# 49. `generate()` 里就真的用了这件事

当前 MiniMind：

```
past_len = ...

self.forward(
    input_ids[:, past_len:],
    past_key_values,
    use_cache=True
)
```

例如已经：

```
input_ids：

我 喜欢 学习 AI
```

而：

```
past_len = 3
```

那么**下一次只送：**

```
AI
```

**进去。**
**以前：**

```
我 喜欢 学习
```

**的 K/V：**

```
直接从 cache 取。
```

**这就是为什么：**

> **LLM 的 prefill 和 decode 是两个非常不同的推理阶段。**

你现在先记：

```
Prefill
→ 第一次处理整个 Prompt
→ 一次性建立大量 KV Cache

Decode
→ 之后通常一次只处理一个新 token
→ 使用旧 KV Cache
```

这个概念以后做模型优化会非常重要。

# 50. Flash Attention 当前可以先怎么理解？

MiniMind：

```
F.scaled_dot_product_attention(...)
```

满足条件时**直接调用 PyTorch 优化实现。**
**你现在不要提前陷入 FlashAttention 算法。**
暂时理解：
**普通写法逻辑上：**

```
QK^T
↓
完整 Attention Matrix
↓
Softmax
↓
乘 V
```

Flash Attention 的核心目标是：

> **数学结果仍然是 Attention，但通过重新组织计算与显存访问，不需要像朴素实现那样把巨大中间矩阵完整地反复写入显存，从而更快、更省显存。**

算法细节以**后 CS336 会非常适合深入。**
**现在 MiniMind 阶段知道：**

```
scaled_dot_product_attention
```

不是新的 Attention 数学原理。
而主要是：

> **更高效的实现方式。**

够了。

# 51. 到这里把每个模块“删掉会怎样”总结一下

| 模块                           | 主要作用                   | 删掉会怎样                                  |
| ------------------------------ | -------------------------- | ------------------------------------------- |
| Embedding                      | token ID → 连续向量        | Transformer 没有**可处理的连续表示**        |
| RMSNorm                        | 稳定 hidden 数值尺度       | **深层训练稳定性可能变差**                  |
| QK-Norm                        | 控制 Q/K 尺度              | **Attention score 更容易出现数值不稳定**    |
| RoPE                           | 注入位置信息               | 模型**难以正确区分顺序/相对位置**           |
| GQA                            | 减少 KV heads              | **换回 MHA 可运行，但参数和 KV Cache 更大** |
| Causal Mask                    | 禁止偷看未来 token         | **训练目标直接泄漏**                        |
| Attention                      | token 间交换信息           | token 之间**失去动态上下文交互**            |
| SwiGLU                         | **token 内非线性特征变换** | **模型表达能力大幅下降**                    |
| Residual                       | 保留信息与梯度通道         | 深层网络**明显更难训练**                    |
| Final RMSNorm                  | 输出前稳定 hidden state    | 训练/输出数值稳定性可能下降                 |
| LM Head                        | hidden → vocab logits      | 根本不能预测 token                          |
| KV Cache                       | 保存历史 K/V               | 结果仍可生成，但自回归推理重复计算严重      |
| 这个表比死背模型名词重要得多。 |                            |                                             |

# 52. 你现在应该怎样重新阅读 `model_minimind.py`

不要从第一行开始逐字符啃。
建议按这个顺序：

```
① MiniMindForCausalLM.forward

↓ 找到整个输出

② MiniMindModel.forward

↓ 找到 8 个 Blocks

③ MiniMindBlock.forward

↓ 找到 Attention + FFN + Residual

④ Attention.forward

↓ 追 Q/K/V shape

⑤ FeedForward.forward

↓ 看 SwiGLU

⑥ RMSNorm

⑦ RoPE

⑧ 最后再看 KV Cache / generate
```

为什么**倒着看？**
**因为你先知道：**

> **“整个系统到底在干嘛？”**

**再钻细节。**
否则很容易一上来就陷在：

```
rope_scaling
precompute_freqs_cis
rotate_half
```

里，然后完全忘了模型主干。

# 53. 这一轮现阶段不要深挖的三块

当前源码**里还有：**

### **① YaRN**

```
长上下文 RoPE scaling
```
**暂时跳过。**

### ② MoE

```
MOEFeedForward
```
暂时跳过。

### ③ HuggingFace 的各种继承接口

```
PreTrainedModel
GenerationMixin
MoeCausalLMOutputWithPast
```
暂时把**它们当：**

> **HuggingFace 兼容层。**

**这三块如果现在同时学，会把主线搞乱。**
等基础模型真正看懂后再回来。

# 54. 第三轮你必须形成的“源码阅读肌肉”

以后你看到：

```
xq = self.q_proj(x)
```

脑袋里应该自动出现：

```
x
[B,T,768]

↓

Linear(768,768)

↓

Q
[B,T,768]

↓

view

[B,T,8,96]
```

看到：

```
xk = self.k_proj(x)
```

自动：

```
[B,T,768]

↓

Linear(768,384)

↓

[B,T,4,96]
```

看到：

```
xq @ xk.transpose(-2,-1)
```

自动：

```
[B,8,T,96]
@
[B,8,96,T]

↓

[B,8,T,T]
```

看到：

```
down_proj(
    silu(gate_proj(x))
    *
    up_proj(x)
)
```

自动：

```
768
↓
2432 × 两路
↓
element-wise gate
↓
768
```

这才是我们这轮真正想培养的能力。

# 55. 给你一个第三轮验收题

先不要查资料，看看现在是否能自己回答。

### 题 1

为什么：

```
hidden_size = 768
attention_heads = 8
```

会得到：

```
head_dim = 96
```

### 题 2
为什么 MiniMind：

```
Q = 8 heads
K = 4 heads
V = 4 heads
```

最后仍然能够做 Attention？

### 题 3

下面 shape 怎么变？

```
Q = [32,512,8,96]
```

执行：

```
Q.transpose(1,2)
```

以后是什么？

### 题 4

下面矩阵乘得到什么 shape？

```
Q   [32,8,512,96]

K^T [32,8,96,512]
```

### 题 5
为什么 RoPE：

```
只改 Q / K
```

而不是简单给：

```
hidden_states
```

加 position embedding？
不要求你数学证明，但应该能讲出它是在干什么。

### 题 6

为什么 KV Cache：

```
只缓存 K/V
```

而不缓存 Q？

### 题 7

MiniMind 使用 GQA 后，如果原来 MHA 有：

```
8 KV heads
```

现在只有：

```
4 KV heads
```

为什么这会降低推理显存？

### 题 8

为什么：

```
Attention
```

与：

```
SwiGLU FFN
```

最终都必须重新回到：

```
[B,T,768]
```

### 题 9
这两行有**什么本质区别？**

```
RMSNorm(hidden_states)
```

和：

```
RMSNorm(Q)
```

提示：

```
最后一个维度分别是多少？
```

### 题 10
完整补完：

```
input_ids
[B,T]

→ __________
[B,T,768]

→ Transformer × 8
[B,T,768]

→ __________
[B,T,6400]

→ Shift + CrossEntropy
loss
```

如果这 10 题你大部分已经能靠理解回答出来，**你已经拥有足够前置去真正开始逐行阅读 MiniMind 的核心模型源码了**，而不是只停留在“我学过 Transformer”。
下一步我建议不要继续无限补模型理论，而是进入**第四轮：真正的大模型训练工程**——拿 `train_pretrain.py` 把 `AdamW → LR schedule → gradient accumulation → mixed precision → grad clipping → checkpoint/resume → DDP` 串起来。到那一步，你就会把这一轮的“模型 forward”与第一轮的“backward / optimizer”真正闭环起来。