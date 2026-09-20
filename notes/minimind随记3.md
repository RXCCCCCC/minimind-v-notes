[TOC]



可以，新的 `learnminimind` 就从这里正式接上。你前面的学习进度我按你给出的衔接说明继承：**后续目标切换为 MiniMind-V，MiniMind 只作为已经掌握的 LLM backbone 前置**。

这一轮我们只完成：

> `VLMConfig → MiniMindVLM.__init__ → Vision Encoder → Vision Projector`

先不进入 `forward()` 里“视觉 token 怎样替换文本 embedding”。那部分下一轮单独拆开。

---

# 0. 先定位：我们现在到底在看哪里

当前仓库：

```text
jingyaogong/minimind-v
└── model/
    ├── model_minimind.py
    └── model_vlm.py        ← 本轮
```

当前 `master` 的 `model_vlm.py` 大约 172 行。它直接：

```text
import model_minimind
↓
复用 MiniMind
↓
在 MiniMindForCausalLM 外面增加视觉能力
```

也就是说你之前形成的这个认识完全正确：

```text
MiniMind-V
=
MiniMind LLM Backbone
+
Vision Encoder
+
Vision Projector
+
视觉 token 注入逻辑
```

它**没有重新写一套 Transformer**。当前源码中 `MiniMindVLM` 直接继承 `MiniMindForCausalLM`。([GitHub][1])

而且当前 2026-04-20 之后的 `master` 已经和旧教程有一个很重要的区别：

```text
旧版本：
SigLIP2 P16
→ 更多视觉 token
→ projector 内再压缩 token

当前版本：
SigLIP2 P32
256×256
→ 8×8 patch
→ 原生 64 visual tokens
→ 不再额外 reshape / merge
```

README 的更新日志也明确说明了这一变化。([GitHub][2])

---

# 1. 第一块：`VLMConfig`

位置：

```text
model/model_vlm.py
└── class VLMConfig(MiniMindConfig)
```

源码在 GitHub 页面大约对应文件第 17～25 行；网页展开位置在这里。([GitHub][1])

源码里最关键的真实定义包括：

```python
class VLMConfig(MiniMindConfig):
```

以及视觉维度默认值 `768`、视觉 token 数默认值 `64`。([GitHub][1])

把它结构化展开，其实就是：

```text
VLMConfig
│
├── 继承 MiniMindConfig
│
├── image_special_token = <|image_pad|>
├── image_ids = [12]
├── image_hidden_size = 768
├── image_token_len = 64
│
└── 继续初始化 MiniMindConfig
    ├── hidden_size = 768
    ├── num_hidden_layers = 8
    ├── vocab_size = 6400
    ├── num_attention_heads = 8
    ├── num_key_value_heads = 4
    └── ...
```

当前 MiniMind backbone 的**默认 `hidden_size=768`、8 层、8 个 attention heads、4 个 KV heads 仍来自 `MiniMindConfig`。([GitHub][3])**

---

## 1.1 为什么 `VLMConfig` 要继承 `MiniMindConfig`

这一步特别重要。

你**原来的 MiniMind：**

```text
MiniMindConfig
↓
描述 LLM

hidden_size
num_layers
heads
vocab_size
RoPE
MoE
...
```

现在 **MiniMind-V 并没有：**

```text
重新设计一个 VLMConfig
把所有 LLM 参数复制一遍
```

而是：

```text
MiniMindConfig
        ↑
        │ 继承
    VLMConfig
```

所以可以认为：

```text
VLMConfig
=
MiniMindConfig
+
几项视觉相关配置
```

这是非常典型的工程思路。

---

# 2. 四个 VLM 特有配置分别是什么

## 2.1 `image_special_token`

默认：

```text
<|image_pad|>
```

注意，现在**不要急着把它理解成“真正的图像 embedding”**。

它首先只是 **tokenizer 序列里的一个特殊 token。**

例如未来**一条多模态输入，概念上可能形成：**

```text
<|image_pad|>
<|image_pad|>
<|image_pad|>
...
64 个
...
请描述这张图片
```

**Tokenizer 之后：**

```text
[12, 12, 12, ... 64 个 ..., text_ids...]
```

**后面的 `forward()` 才会做一件非常关键的事：**

```text
这些位置原来的文本 embedding
↓
替换成真正的 visual embedding
```

所以你现在**先记一句：**

> **`<|image_pad|>` 是给视觉特征预留位置的占位 token，并不是 Vision Encoder 的输出本身。**

**真正的替换逻辑就是后面我们要重点看的 `count_vision_proj()`。当前源码的确会根据 `image_ids[0]` 找这些位置。([GitHub][1])**

---

# 3. `image_ids = [12]` 是什么

这里很容易混淆。

```text
image_special_token
=
"<|image_pad|>"
```

这是**字符串。**

**Tokenizer 会把它映射成：**

```text
token id = 12
```

**于是：**

```text
image_ids = [12]
```

**当前代码后面会拿：**

```text
marker = image_ids[0]
```

**然后在：**

```text
input_ids
[B,T]
```

**里面寻找：**

```text
12 12 12 12 ...
```

再**把这些位置对应的 embedding 替换掉**。当前 `count_vision_proj()` 正是这么做的。([GitHub][1])

因此：

```text
<|image_pad|>
        ↓ tokenizer
       12
        ↓ Embedding
某个普通 token embedding

```

但**真正多模态时**：

```text
token id 12 对应位置
        ↓
原 embedding 被替换
        ↓
SigLIP visual embedding
```

你现在只需要先知道这个机制，替换代码下一轮细看。

---

# 4. `image_hidden_size = 768`

这个就是：

> Vision Encoder 输出的**每一个 visual token 的 feature dimension**。

当前 **MiniMind-V 使用：**

```text
SigLIP2
siglip2-base-p32-256-ve
```

README 明确描述当前 **Vision Encoder 是基于 ViT-B/32，输入 256×256，每个 patch 为 32×32，因此形成：**

```text
256 / 32 = 8

8 × 8 = 64 patches
```

**Vision backbone 给出：**

```text
64 个视觉 token
×
每个 768 维
```

**也就是单张图片：**

```text
[64, 768]
```

**batch 化：**

```text
[B, 64, 768]
```

当前项目**只取 Vision Encoder backbone 的 encoder hidden states，不使用最后为 SigLIP 图文对齐任务准备的 pooled/final embedding。([GitHub][2])**

---

# 5. `image_token_len = 64`

**这个东西和上面的 `image_hidden_size` 完全不是一回事。**

你一定要分开：

| 参数                      | 含义                           |
| ------------------------- | ------------------------------ |
| `image_hidden_size = 768` | **每个视觉 token 有多少维**    |
| `image_token_len = 64`    | **一张图片有多少个视觉 token** |

所以：

```text
Visual Feature

[B, 64, 768]
 ↑   ↑   ↑
 │   │   └─ 每个 token 768 维
 │   └──── 64 个 visual tokens
 └──────── batch
```

这两个 768/64 很容易混。

---

# 6. 为什么刚好是 64？

当前 **Vision Encoder：**

```text
input image
[3,256,256]

patch_size = 32
```

**于是图片被分：**

```text
纵向：256 / 32 = 8
横向：256 / 32 = 8
```

得到：

```text
8 × 8 = 64 patch tokens
```

所以整个过程**先粗略记成：**

```text
Image
[B,3,256,256]

↓ SigLIP2 ViT-B/32

[B,64,768]
```

README 当前就是**按这个结构解释 MiniMind-V 的**。([GitHub][2])

---

# 7. 第二块：`MMVisionProjector`

位置：

```text
model/model_vlm.py
└── class MMVisionProjector(nn.Module)
```

当前 **projector 的结构是：**

```text
LayerNorm(in_dim)
↓
Linear(in_dim → out_dim)
↓
GELU
↓
Linear(out_dim → out_dim)
```

源**码对应 `model_vlm.py` 的 projector 定义。([GitHub][1])**

在默认 MiniMind-V 里：

```text
in_dim
=
image_hidden_size
=
768

out_dim
=
MiniMind hidden_size
=
768
```

因此**默认 shape：**

```text
[B,64,768]
    ↓ LayerNorm
[B,64,768]
    ↓ Linear(768→768)
[B,64,768]
    ↓ GELU
[B,64,768]
    ↓ Linear(768→768)
[B,64,768]
```

看到这里你可能**马上产生一个很好的问题：**

> **“输入是 768，输出也是 768，那 projector 到底投影了什么？”**

这个问题非常关键。

---

# 8. `768 → 768` 为什么还需要 Projector？

因为：

> **维度一样 ≠ 表示空间一样。**

这是**理解 VLM 最重要的一点之一。**

**SigLIP：**

```text
visual token
v ∈ R^768
```

**MiniMind：**

```text
text token embedding
t ∈ R^768
```

**虽然数学上都是：**

```text
768-dimensional vector
```

但**坐标系完全不同。**

可以把它想成：

```text
SigLIP 的 768 维
=
“视觉语言”

MiniMind 的 768 维
=
“LLM 语言”
```

**维度数量刚好一样，不代表：**

```text
SigLIP feature
可以直接喂给 MiniMind
```

---

# 9. 一个非常直观的类比

假设两个国家都用：

```text
26 个字母
```

但**一个说英语：**

```text
cat
```

**另一个用了完全不同的编码规则。**

你不能因为：

```text
都是 26 个符号
```

就认为：

```text
表示空间相同
```

Vision Projector **的任务就是学：**

```text
SigLIP representation
        ↓
      翻译
        ↓
MiniMind representation
```

**所以：**

```text
R^768
↓
R^768
```

**也完全有意义。**

---

# 10. Projector 到底学的是什么

可以把**它写成：**

$$
z_v
=
P(h_v)
$$

**其中：**

```text
h_v
=
SigLIP visual feature

Shape:
[B,64,768]
```

而：

```text
P
=
Vision Projector
```

**输出：**

```text
z_v
=
LLM-compatible visual embedding

Shape:
[B,64,768]
```

**因此 projector 不是主要干：**

> **“改 shape”。**

**而是主要干：**

> **representation alignment / cross-modal alignment**

**即：**

```text
Vision Semantic Space
↓
LLM Semantic Space
```

**当前官方 README 也明确将这一层描述为把原始视觉 feature 投到 LLM hidden space，使视觉 token 能和文本 token 在同一表示空间交互。**([GitHub][2])

---

# 11. 为什么是两层 MLP 而不是一层 Linear？

当前 MiniMind-V README 将**方案类比到 LLaVA-1.5：**

```text
LayerNorm
+
2-layer MLP
```

相比单个：

```text
Linear
```

增加：

```text
GELU
```

就增加了非线性映射能力。([GitHub][2])

概念上：

```text
Linear:
x → Wx+b
```

**只能做线性变换。**

而：

```text
Linear
↓
GELU
↓
Linear
```

可以**学习更复杂的：**

```text
视觉表示 → LLM 表示
```

**关系。**

---

# 12. `LayerNorm` 为什么放在最前面

当前 `master` 的一个明确**更新就是给 projector 加上了 `LayerNorm`。**([GitHub][2])

输入：

```text
x
[B,64,768]
```

LayerNorm：

```text
沿最后一个维度 768 做归一化
```

所以：

```text
[B,64,768]
→
[B,64,768]
```

**shape 不变。**

其主要意义不是改变维度，而是让进入 projector 的 SigLIP feature 的**数值分布更稳定。**

从你的**科研视角看，这已经是第一个非常典型的：**

```text
可消融变量
```

**未来完全可以实验：**

```text
Baseline:
LN → Linear → GELU → Linear

Ablation:
Linear → GELU → Linear

或者:
single Linear
```

但**现在先别改**。

我们只是知道：

> P**rojector 本身以后就是一个很天然的实验切入点。**

---

# 13. 有个很有意思的源码细节：`source_tokens`

构造函数目前**还保留了类似：**

```text
source_tokens = 64
target_tokens = 64
```

**这样的参数接口，但是当前 projector 的 `forward()` 实际只运行 MLP。**

换句话说，现在的：

```text
source_tokens
target_tokens
```

已经**不参与 token reshape / merge**。

这是当前版本相较 4 月初版本非常值得注意的历史遗留痕迹。

因为 4 月初版本：

```text
P16
256/16 = 16

16×16 = 256 visual tokens

↓
reshape / merge

64 tokens
```

而现在 P32：

```text
256/32 = 8

8×8 = 64 tokens
```

已经**天然是 64。**

所以：

```text
无需：
256 → 64

直接：
64 → 64
```

官方 2026-04-20 更新日志明确写了**“P32 原生 64 token，无需下采样”。([GitHub][2])**

这是**读源码非常好的一个反应**：

> **不要看到函数参数存在，就默认它一定参与实际计算，要继续看 `forward()`。**

---

# 14. 第三块：`MiniMindVLM.__init__`

现在来到真正的核心：

```text
class MiniMindVLM(MiniMindForCausalLM)
```

注意这个**继承关系。**

不是：

```text
MiniMindVLM(nn.Module)
```

而是：

```text
MiniMindVLM
    ↓ inherits
MiniMindForCausalLM
```

当前源码明确如此。([GitHub][1])

这句话几乎可以解释整个项目设计。

---

# 15. 原来的 MiniMind 有什么

你已经学过：

```text
MiniMindForCausalLM
│
├── self.model
│   ├── embed_tokens
│   ├── Transformer Block × N
│   └── final norm
│
└── self.lm_head
```

输入：

```text
input_ids
[B,T]
```

经过：

```text
Embedding
[B,T,768]

↓ Transformer

hidden_states
[B,T,768]

↓ LM Head

logits
[B,T,6400]
```

当前 `model_minimind.py` 的 `MiniMindForCausalLM` 仍就是这套结构。([GitHub][3])

---

# 16. MiniMind-V 初始化时做了什么？

逻辑可以准确概括成：

```text
MiniMindVLM.__init__()

① 建 VLMConfig

② 调 super().__init__()
   ↓
   先完整建一个 MiniMind LLM

③ 加载 Vision Encoder

④ 加载 Image Processor

⑤ 新建 Vision Projector
```

因此最终模型：

```text
MiniMindVLM
│
├── MiniMind LLM
│   ├── Embedding
│   ├── Decoder Blocks
│   └── LM Head
│
├── vision_encoder
│   └── SigLIP2
│
├── processor
│   └── SiglipImageProcessor
│
└── vision_proj
    └── LN + MLP
```

当前初始化代码就是按这一顺序做的。([GitHub][1])

---

# 17. 为什么必须先 `super().__init__()`

这一行你作为 PyTorch 学习重点一定要看懂。

因为：

```text
MiniMindVLM
继承
MiniMindForCausalLM
```

所以：

```text
super().__init__(config)
```

其实就是调用：

```text
MiniMindForCausalLM.__init__()
```

于是**帮你创建：**

```text
self.model = MiniMindModel(...)
self.lm_head = Linear(...)
```

**包括：**

```text
Embedding
Transformer
RMSNorm
LM Head
```

**全部已经建完。**

**这就是：**

```text
哪些 MiniMind 代码完全复用？
```

**答案：**

> **几乎整个 LLM backbone。**

---

# 18. MiniMind → MiniMind-V 到这里到底新增了多少？

原来：

```text
MiniMind

input_ids
↓
Embedding
↓
Transformer
↓
LM Head
```

现在：

```text
MiniMind-V

                Image
                  ↓
              Processor
                  ↓
            Vision Encoder
                  ↓
           Vision Projector
                  ↓
              Visual Tokens
                  ↓
                  ┐
                  │
input_ids         │
    ↓             │
Text Embedding ───┤
                  ↓
           Multimodal Embedding
                  ↓
       原 MiniMind Transformer
                  ↓
               LM Head
```

真正**新增的核心其实只有左上角这一条：**

```text
Image
↓
Processor
↓
Vision Encoder
↓
Projector
↓
Visual Embeddings
```

以及下一轮会看的：

```text
怎么把 Visual Embeddings 塞进 Text Embeddings
```

---

# 19. 第四块：`get_vision_model()`

现在继续看初始化过程中：

```text
vision_encoder
processor
```

到底怎么来的。

位置：

```text
model/model_vlm.py
└── MiniMindVLM
    └── get_vision_model()
```

当前逻辑为：

```text
输入：
vision_model_path

↓

检查路径存在

↓

SiglipVisionModel.from_pretrained(...)

↓

SiglipImageProcessor.from_pretrained(...)

↓

遍历 Vision Encoder 参数
requires_grad = False

↓

model.eval()

↓

返回：
vision_encoder, processor
```

源码对应约第 36～55 行附近。([GitHub][1])

---

# 20. 当前到底用的是什么 Vision Encoder？

**不是 CLIP。**

**也不是你可能从旧 MiniMind-V 资料里看到的：**

```text
CLIPVisionModel
```

**当前是：**

```text
SiglipVisionModel
```

**具体资源目录：**

```text
./model/siglip2-base-p32-256-ve
```

**README 的下载步骤也明确要求下载这一视觉编码器。([GitHub][2])**

---

# 21. Vision Encoder 在这里负责什么

输入最终会是：

```text
pixel_values

[B,3,256,256]
```

然后：

```text
SigLIP2 Vision Transformer
```

把**图片切成 patch：**

```text
256×256
↓
32×32 patch
↓
8×8
↓
64 patches
```

**每个 patch 变成：**

```text
768 dimensional feature
```

**最终核心 backbone 输出：**

```text
last_hidden_state

[B,64,768]
```

MiniMind-V 后面正是使用 `outputs.last_hidden_state`。([GitHub][1])

---

# 22. 这里先帮你纠正一个特别容易误会的地方

SigLIP **本身原来可以拿：**

```text
整张图片 embedding
```

**去跟：**

```text
text embedding
```

**做视觉-语言对齐。**

但是 MiniMind-V **不要那个最终全局向量**。

**因为如果你只拿：**

```text
[B,768]
```

**相当于：**

```text
一整张图片
→
一个视觉 token
```

**空间信息压缩得太厉害。**

MiniMind-V 要的是：

```text
[B,64,768]
```

也就是：

```text
64 个 patch
↓
64 个 visual token
```

这样 LLM 后面才能看到：

```text
这里可能是猫
这里可能是桌子
这里可能是背景
...
```

而**不是整张图片只有一个向量。**

当前代码因此返回的是 `last_hidden_state`。([GitHub][1])

---

# 23. 为什么 Vision Encoder 全被冻结？

当前源码：

```text
for every vision parameter
    requires_grad = False
```

然后：

```text
vision_encoder.eval()
```

所以：

```text
SigLIP2
❄ Frozen
```

当前项目 README 也说明**视觉编码器约 95M 参数、全程冻结，只作为图像 feature extractor。([GitHub][4])**

这是整个项目设计的核心之一。

如果**不冻：**

```text
LLM
+
95M Vision Encoder
+
Projector
```

**全部一起训练。**

**对于 MiniMind-V 这种：**

```text
低成本
单卡
入门型 VLM
```

**明显会更重**，而且也**容易把已经训练好的 SigLIP 视觉能力破坏掉。**

---

# 24. `requires_grad=False` 与 `eval()` 是一回事吗？

不是。

这点很适合顺手补 PyTorch。

### `requires_grad=False`

控制：

```text
是否计算这个参数的 gradient
```

于是：

```text
parameter.grad
```

**不会被正常训练更新。**

也就是说：

```text
optimizer
不会更新 Vision Encoder
```

---

### `.eval()`

控制：

```text
module 的运行模式
```

主要影响：

```text
Dropout
BatchNorm
```

等 train/eval 行为。

所以：

```text
requires_grad=False
```

和：

```text
eval()
```

**承担的是两件不同的事情。**

这里两个都做，是很标准的冻结 feature extractor 写法。

---

# 25. 更彻底的是：后面还有 `torch.no_grad()`

后面的：

```text
get_image_embeddings()
```

里面当前还使用：

```text
with torch.no_grad():
    vision_model(...)
```

因此整个视觉 backbone 路径实际上是：

```text
requires_grad = False
+
eval()
+
no_grad()
```

**三层明确表示：**

> **SigLIP2 就是一个固定 feature extractor。**

([GitHub][1])

---

# 26. 于是到这一轮结束，我们已经能画出真实 shape

当前默认情况下：

```text
原始图片
PIL.Image

↓ SiglipImageProcessor

pixel_values

[B,3,256,256]

↓ SigLIP2 Vision Encoder
   ViT-B/32

[B,64,768]

↓ MMVisionProjector

LayerNorm
[B,64,768]

↓ Linear 768→768

[B,64,768]

↓ GELU

[B,64,768]

↓ Linear 768→768

visual embedding

[B,64,768]
```

与此同时文本侧：

```text
input_ids
[B,T]

↓ MiniMind Embedding

text embedding
[B,T,768]
```

现在你应该已经看见了 projector 的真正目的：

```text
视觉：
[B,64,768]
       ↓
     Projector
       ↓
[B,64,768]
       │
       │ 同样的 hidden_size=768
       ↓

文本：
[B,T,768]
```

**下一步就终于可以把两者放进同一个序列。**

---

# 27. 到目前为止 MiniMind 与 MiniMind-V 精确对照

| 环节                  | MiniMind         | MiniMind-V                          |
| --------------------- | ---------------- | ----------------------------------- |
| Config                | `MiniMindConfig` | `VLMConfig(MiniMindConfig)`         |
| 文本 tokenizer        | 有               | 完全保留                            |
| Text Embedding        | 有               | 完全保留                            |
| Transformer           | 有               | 完全保留                            |
| LM Head               | 有               | 完全保留                            |
| **Vision Encoder**    | **无**           | **SigLIP2 P32**                     |
| **Image Processor**   | **无**           | **SiglipImageProcessor**            |
| **Projector**         | **无**           | **LN + 2-layer MLP**                |
| **image placeholder** | **无**           | **`<\|image_pad\|>`**               |
| **visual tokens**     | **无**           | **64**                              |
| **hidden dim**        | **768**          | **文本、projected vision 都是 768** |

所以不要把 VLM 想成：

```text
重新学另一种模型
```

你现在更应该形成：

```text
我已经有 MiniMind

现在只是：

在它的 embedding 输入之前
多接了一条 image → embedding 支路
```

---

# 28. 这一轮你最应该形成的“源码阅读反应”

以后你**读任何 VLM，看到：**

```python
VisionModel
```

**第一反应：**

> **图片是谁编码的？输出 `[B,N,Cv]` 是多少？**

**看到：**

```python
Projector
```

**第一反应：**

> **它是在做 `Cv → C_llm` 的跨模态表示对齐，还是还顺便压 visual token 数？**

**看到：**

```python
image_token_len
```

**第一反应：**

> **这是 visual token 数量 N，不是 hidden dimension。**

看到：

```python
image_hidden_size
```

**第一反应：**

> **这是 Vision Encoder 每个 token 的 feature dimension `Cv`。**

**看到：**

```python
requires_grad=False
```

**第一反应：**

> **这个 backbone 被当 feature extractor 使用。**

**看到：**

```python
class VLM(MiniMindForCausalLM)
```

**第一反应：**

> **先找它到底复用了父类什么，别把父类已有 LLM 再学一遍。**

---

# 29. 你现在先把整个结构压缩成这一张脑图

```text
                    MiniMind-V
                         │
        ┌────────────────┴────────────────┐
        │                                 │
      Image                              Text
        │                                 │
        ↓                                 ↓
SiglipImageProcessor                  Tokenizer
        │                                 │
        ↓                                 ↓
pixel_values                         input_ids
[B,3,256,256]                         [B,T]
        │                                 │
        ↓                                 ↓
SigLIP2 P32                       Token Embedding
  FROZEN                             [B,T,768]
        │
        ↓
last_hidden_state
[B,64,768]
        │
        ↓
Vision Projector
LN
Linear
GELU
Linear
        │
        ↓
Visual Embedding
[B,64,768]
        │
        └────────── ??? ──────────┐
                                  ↓
                         Multimodal Embedding
                              [B,T,768]
                                  ↓
                        MiniMind Transformer
                                  ↓
                               logits
```

现在**真正唯一还没解释的核心就是图里的：**

```text
???
```

**即：**

> **那 64 个 `[768]` 的视觉向量，究竟怎样塞进 `[B,T,768]` 的文本 embedding？**

**这正是下一部分源码：**

```text
get_image_embeddings()
        ↓
count_vision_proj()
        ↓
MiniMindVLM.forward() 前半段
```

下一轮我们**就只啃这一件事，而且会具体追：**

```text
<|image_pad|>
↓
token id = 12
↓
input_ids 中连续 64 个 12
↓
原 text embedding
↓
找到位置
↓
替换成 [64,768] visual embedding
↓
最终仍然得到 [B,T,768]
```

这里会是你真正理解“**图片到底怎样进入 Decoder-only LLM**”的关键一轮。

可以，这一轮就把 **“图片到底怎么混进 Decoder-only LLM”** 真正吃透。我们继续严格沿用你给的新 `learnminimind` 学习路线。



# MiniMind-V 源码带读（二）

## `get_image_embeddings → count_vision_proj → forward` 前半段

当前仓库仍然是：

```text
jingyaogong/minimind-v
└── model/
    └── model_vlm.py
        └── MiniMindVLM
            ├── get_image_embeddings()
            ├── count_vision_proj()
            └── forward()
```

这一轮结束时，你应该能准确回答：

> 一张图片**经过 SigLIP 和 Projector 变成 `[64,768]` 后，这 64 个向量到底放到 LLM 哪儿？**

答案**先压缩成一句：**

```text
不是追加
不是 Cross-Attention
不是额外视觉分支一直并行

而是：

先构造包含 64 个 image placeholder 的文本序列
↓
照常做 token embedding
↓
找到这 64 个 placeholder 的 embedding 位置
↓
原位换成 64 个 visual embedding
↓
得到普通的 [B,T,768]
↓
直接送进原 MiniMind Decoder
```

当前源码就是这样实现的。([GitHub][1])

---

# 1. 先建立一个具体例子

先假设 batch size：

```text
B = 1
```

**最大序列长度：**

```text
T = 512
```

**一张图片对应：**

```text
N_visual = 64
```

**hidden size：**

```text
C = 768
```

数据集最终会把原始：

```text
<image>
```

替换成：

```text
<|image_pad|><|image_pad|>...<|image_pad|>
```

一共 64 个。

当前 `VLMDataset` 中实际就是：

```python
self.image_special_token = image_special_token * image_token_len
```

而：

```python
content = turn['content'].replace(
    '<image>',
    self.image_special_token
)
```

默认 `image_token_len=64`。([GitHub][2])

于是一个概念上的输入：

```text
<image>
请描述这张图片
```

变成：

```text
<|image_pad|> × 64
请描述这张图片
```

Tokenizer 再**把每个：**

```text
<|image_pad|>
```

**变成 token id：**

```text
12
```

**所以 `input_ids` 某一段会长得像：**

```text
...,
12,12,12,12,12,... 共64个 ...,
“请”,
“描述”,
“这”,
“张”,
“图片”,
...
```

注意：

> **此时图片的信息本身还完全没有进入 `input_ids`。**

这些 `12` 仅仅是在告诉模型：

```text
这里未来留 64 个位置给图片
```

---

# 2. 为什么一定要提前留 64 个位置？

因为后面视觉编码器产生的是：

```text
visual embeddings
[64,768]
```

所以**文本序列里必须给它：**

```text
64 个位置
```

这样才可以**做到一一对应：**

```text
image_pad #1  ← visual token #1
image_pad #2  ← visual token #2
image_pad #3  ← visual token #3
...
image_pad #64 ← visual token #64
```

因此：

```text
image_token_len = 64
```

实际上同时连接了两个世界：

```text
Dataset 世界
64 个 placeholder

       ↕ 必须对应

Vision 世界
64 个 visual tokens
```

这是个非常重要的 VLM 设计约束。

---

# 3. 第一段：`get_image_embeddings()`

当前真实源码核心是：

```python
@staticmethod
def get_image_embeddings(image_inputs, vision_model):
    if hasattr(image_inputs, 'keys'):
        image_inputs = {
            k: v.squeeze(1) if v.ndim > 2 and v.shape[1] == 1 else v
            for k, v in image_inputs.items()
        }

    with torch.no_grad():
        outputs = vision_model(**image_inputs)

    return outputs.last_hidden_state
```

位置就是 `model/model_vlm.py → MiniMindVLM.get_image_embeddings()`。([GitHub][1])

---

# 4. 第一行：为什么检查 `.keys`？

```python
if hasattr(image_inputs, 'keys'):
```

因为 Hugging Face 的：

```text
SiglipImageProcessor
```

通常返回的是一个**类似字典的 `BatchFeature`：**

```python
{
    "pixel_values": tensor(...)
}
```

**而不是裸 Tensor。**

所以代码不能简单：

```python
vision_model(image_inputs)
```

而要：

```python
vision_model(**image_inputs)
```

等价于概念上的：

```python
vision_model(
    pixel_values=image_inputs["pixel_values"]
)
```

---

# 5. 这一段 `squeeze(1)` 是干什么？

```python
v.squeeze(1)
if v.ndim > 2 and v.shape[1] == 1
else v
```

主要是在**处理多出来的“图片数量维”。**

**比如 batch 之后可能出现：**

```text
pixel_values

[B,1,3,256,256]
```

这里：

```text
dim 0 = batch
dim 1 = 每个样本有几张图片
dim 2 = RGB
dim 3 = H
dim 4 = W
```

如果：

```text
num_images = 1
```

那么：

```python
squeeze(1)
```

**把：**

```text
[B,1,3,256,256]
```

**变：**

```text
[B,3,256,256]
```

**这才是普通 Vision Encoder 希望看到的 batch shape。**

注意：

```python
squeeze(1)
```

**只在：**

```text
shape[1] == 1
```

**时删除。**

**因此不是随便消掉一个维度。**

---

# 6. 然后真正进入 Vision Encoder

```python
with torch.no_grad():
    outputs = vision_model(**image_inputs)
```

前一轮已经讲过：

```text
vision_model
=
SiglipVisionModel
```

并且**参数被冻结。**

**所以单图 batch：**

```text
pixel_values
[B,3,256,256]

↓

SigLIP2 ViT-B/32

↓

outputs.last_hidden_state
[B,64,768]
```

源码明确返回的就是：

```python
outputs.last_hidden_state
```

而不是 pooled image embedding。([GitHub][1])

---

# 7. 注意一个 PyTorch 细节：这里还没有 Projector

`get_image_embeddings()` 返回的是：

```text
SigLIP 原生 feature

[B,64,768]
```

不是：

```text
MiniMind-compatible feature
```

**Projector 在它外面调用：**

```text
get_image_embeddings(...)
        ↓
vision_proj(...)
```

**所以逻辑是：**

```text
pixel_values
[B,3,256,256]

↓ SigLIP

vision features
[B,64,768]

↓ vision_proj

visual embeddings
[B,64,768]
```

虽然**两个 shape 一样，但前一轮你已经知道：**

```text
shape 相同
≠
representation space 相同
```

---

# 8. 现在进入 `forward()`

当前 `MiniMindVLM.forward()` 一开始：

```python
batch_size, seq_length = input_ids.shape
```

所以：

```text
input_ids
[B,T]
```

例如：

```text
[4,512]
```

这里完全**和 MiniMind 一样**。([GitHub][1])

---

# 9. 然后遇到这一行——本轮最关键的第一行

```python
hidden_states = self.model.dropout(
    self.model.embed_tokens(input_ids)
)
```

这其实和**原 MiniMind 的：**

```python
hidden_states = self.dropout(
    self.embed_tokens(input_ids)
)
```

**是同一个逻辑。原 MiniMind 当前 `MiniMindModel.forward()` 也这么做。([GitHub][3])**

Shape：

```text
input_ids
[B,T]

↓ nn.Embedding(vocab_size,768)

hidden_states
[B,T,768]
```

---

# 10. 一个非常重要的细节：image token 也先正常做 Embedding

假设：

```text
input_ids[0][20:84]
=
[12,12,... 64个]
```

那么：

```python
self.model.embed_tokens(input_ids)
```

会**真的去查：**

```text
Embedding Table[12]
```

**因此此时：**

```text
position 20
=
Embedding(12)

position 21
=
Embedding(12)

...

position 83
=
Embedding(12)
```

**所以暂时得到：**

```text
[普通文字 embedding]
[Embedding(12)]
[Embedding(12)]
...
[Embedding(12)]
[普通文字 embedding]
```

这 **64 个甚至因为 token id 都是 `12`：**

```text
最初 embedding value 也是一样的
```

**至少在加入位置编码、Transformer 之前如此。**

---

# 11. 但这些 `Embedding(12)` 根本不是最终视觉信息

这正是你现在必须建立的认识。

`<|image_pad|>` 的 embedding 只是：

```text
临时占位
```

真正马上会发生：

```text
Embedding(12)
Embedding(12)
Embedding(12)
...
      ↓
全部原位替换
      ↓
visual token 1
visual token 2
visual token 3
...
```

因此 `<|image_pad|>` 本身不是负责：

> “学习图片长什么样”。

它主要负责：

> **告诉模型在哪些 sequence positions 注入 visual features。**

---

# 12. 为什么 MiniMind-V 不直接跳过这一步？

你可能会想到：

> 既然**最后都要替换，那为什么还要先 `embed_tokens(input_ids)`？**

**因为绝大部分 sequence position 仍然是文本。**

比如：

```text
位置：

0      BOS
1~10   system
11~74  image placeholders
75~90  用户文字
91~... assistant
```

**文本位置仍然必须：**

```text
token id
↓
text embedding
```

**所以最方便的办法就是：**

```text
整个序列先统一 Embed

[B,T]
↓
[B,T,768]

然后只修改图片位置
```

而**不是自己重新拼各种 text embedding。**

**这也是当前实现很简洁的地方。**

---

# 13. `pixel_values is not None and start_pos == 0`

接着代码：

```python
if pixel_values is not None and start_pos == 0:
```

这里有两个条件。([GitHub][1])

第一：

```text
pixel_values is not None
```

说明**当前确实是图文输入。**

**否则就：**

```text
纯文本
```

**直接走 MiniMind。**

---

# 14. `start_pos == 0` 为什么非常重要？

你已经学过 KV Cache，所以这里应该很容易连接起来。

**生成第一轮：**

```text
Prefill
```

**输入可能：**

```text
[图片 placeholder ×64] + [用户 prompt]
```

**这时：**

```text
past_key_values = None
```

**所以：**

```text
start_pos = 0
```

**于是：**

```text
处理图片
→ SigLIP
→ Projector
→ 替换 placeholder
```

---

之后 **autoregressive decode：**

```text
第一个新 token
第二个新 token
第三个新 token
...
```

**这时已有 KV Cache：**

```text
start_pos > 0
```

**于是条件：**

```python
start_pos == 0
```

**不成立。**

所以：

> **图片不会在每生成一个 token 时重新跑一遍 SigLIP。**

这非常重要。

---

# 15. 如果没有这个判断会怎样？

假设回答要生成 100 个 token。

没有：

```python
start_pos == 0
```

可能变成：

```text
Prefill
跑 SigLIP

Decode token 1
又跑 SigLIP

Decode token 2
又跑 SigLIP

...

Decode token 100
又跑 SigLIP
```

这显然是**巨大的浪费。**

现在：

```text
Prefill:
Image → SigLIP → visual tokens
                   ↓
             加进序列
                   ↓
Transformer 计算 K/V
                   ↓
              KV Cache

Decode:
只继续利用已有 KV Cache
```

这就跟你之前学的 Prefill / Decode 正好接起来了。

---

# 16. 接着产生 `vision_tensors`

对于普通单图情况，最核心的路径其实就是：

```python
vision_tensors = self.vision_proj(
    MiniMindVLM.get_image_embeddings(
        pixel_values,
        self.vision_encoder
    )
)
```

当前源码还兼容多图及不同 tensor 组织方式，所以实际代码有几个分支。([GitHub][1])

普通情况可以先压缩理解成：

```text
pixel_values
[B,3,256,256]

↓ get_image_embeddings

[B,64,768]   SigLIP space

↓ vision_proj

vision_tensors
[B,64,768]   LLM-compatible space
```

---

# 17. 但当前代码其实支持多张图片

这一点非常值得你看。

如果 batch 内**每个样本有 `num` 张图片：**

```text
pixel_values
[B,num,3,256,256]
```

**当前代码会把前两维暂时 flatten：**

```python
v.flatten(0, 1)
```

即：

```text
[B,num,3,H,W]

↓

[B*num,3,H,W]
```

这是为什么？

因为 **Vision Encoder 不需要知道：**

```text
哪些图片属于同一个聊天样本
```

它**只需要：**

```text
给我一批图片
```

---

# 18. 经过 Vision Encoder

```text
[B*num,3,256,256]

↓

SigLIP

[B*num,64,768]

↓

Projector

[B*num,64,768]
```

然后：

```python
.view(
    bs,
    num,
    image_token_len,
    -1
)
```

变回：

```text
[B,num,64,768]
```

所以对多图情况：

```text
vision_tensors
[B,N_img,64,768]
```

其中：

```text
B     = batch
N_img = 每个样本图片数量
64    = 每张图片 visual token 数
768   = hidden size
```

这也是为什么后面的融合代码要有一个：

```text
k = 当前第几张图
```

---

# 19. 现在进入全文件最关键函数：`count_vision_proj()`

源码路径：

```text
model/model_vlm.py
└── MiniMindVLM
    └── count_vision_proj()
```

核心源码：

```python
def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=512):
    if vision_tensors is None or not self.config.image_ids:
        return h

    marker, vf = self.config.image_ids[0], vision_tensors

    if vf.dim() == 3:
        vf = vf.unsqueeze(1)

    out = []

    for b in range(h.size(0)):
        hb, seq, k, i = h[b], tokens[b].tolist(), 0, 0

        while i < len(seq):
            if seq[i] == marker:
                start = i

                while i < len(seq) and seq[i] == marker:
                    i += 1

                if k < vf.size(1):
                    hb = torch.cat(
                        (hb[:start],
                         vf[b][k][:i-start],
                         hb[i:]),
                        dim=0
                    )[:seqlen]

                    k += 1
            else:
                i += 1

        out.append(hb)

    return torch.stack(out)
```

这就是当前 `master` 真正做 image-text fusion 的地方。([GitHub][1])

---

# 20. 函数的四个输入先完全搞清楚

## `tokens`

```text
input_ids

Shape:
[B,T]
```

里面存在：

```text
12,12,12,...连续64个
```

作用：

> 用**来找图片应该放在哪里。**

---

## `h`

就是**刚才：**

```python
embed_tokens(input_ids)
```

**得到的：**

```text
text embeddings
[B,T,768]
```

**作用：**

> 是真正**准备修改的 embedding sequence。**

---

## `vision_tensors`

单图：

```text
[B,64,768]
```

**多图：**

```text
[B,N_img,64,768]
```

作用：

> **真正的 visual embeddings。**

---

## `seqlen`

当前**传进来的：**

```python
input_ids.shape[1]
```

**所以通常：**

```text
T
```

**作用：**

> **保证最后序列不要超过原 sequence length。**

---

# 21. 第一行保护逻辑

```python
if vision_tensors is None or not self.config.image_ids:
    return h
```

含义：

```text
没有视觉输入
或者
没有 image marker 定义

↓

什么都不改
↓

原文本 embedding 直接返回
```

这也是为什么同**一个 `MiniMindVLM` 可以基本兼容纯文本输入。**

---

# 22. 这里：

```python
marker, vf = self.config.image_ids[0], vision_tensors
```

默认：

```text
marker = 12
```

而：

```text
vf = visual features
```

**单图可能：**

```text
vf
[B,64,768]
```

---

# 23. 为什么又 `unsqueeze(1)`？

```python
if vf.dim() == 3:
    vf = vf.unsqueeze(1)
```

**单图：**

```text
[B,64,768]
```

**只有三维。**

**但下面的逻辑想统一成：**

```text
[B,N_img,64,768]
```

**因此：**

```text
[B,64,768]

↓ unsqueeze(1)

[B,1,64,768]
```

**这样无论：**

```text
1 张图片
```

**还是：**

```text
3 张图片
```

**都统一表示：**

```text
[B,N_img,64,768]
```

**这是一种非常常见的 PyTorch 工程技巧。**

---

# 24. 然后逐个 batch sample 处理

```python
for b in range(h.size(0)):
```

如果：

```text
h
[B,T,768]
```

那么：

```text
h.size(0)=B
```

每次处理一个样本。

---

# 25. 这一行变量很多，但都非常简单

```python
hb, seq, k, i = h[b], tokens[b].tolist(), 0, 0
```

分别拆：

### `hb`

```text
h[b]

Shape:
[T,768]
```

当前样本的**完整 embedding sequence。**

---

### `seq`

```text
tokens[b]

Shape 原本:
[T]

↓ .tolist()

Python list，长度 T
```

里面类似：

```text
[1, 233, 542, 12, 12, ... 64个..., 298, 315, ...]
```

作用纯粹是**方便扫描 token ID。**

---

### `k`

```text
当前应该使用第几张图片

k=0
```

如果**文本中有第二段 64 个 `12`：**

```text
k=1
```

**代表用第二张图片。**

---

### `i`

```text
扫描 sequence 的位置
```

相当于 C++ 里的：

```cpp
for(int i=0;i<n;)
```

---

# 26. 开始扫描 token

```python
while i < len(seq):
```

一直从左往右找：

```text
token id == 12
```

---

# 27. 找到 image placeholder

```python
if seq[i] == marker:
```

也就是：

```text
seq[i] == 12
```

代表：

> 找到了一段图片占位区域的开头。

然后：

```python
start = i
```

记下开始位置。

---

# 28. 接下来不是只吃一个 `12`

而是：

```python
while i < len(seq) and seq[i] == marker:
    i += 1
```

**一直扫完连续的一整段：**

```text
12 12 12 12 ... 12
```

假设：

```text
start = 20
```

共**有 64 个。**

**那么结束时：**

```text
i = 84
```

所以：

```text
i - start
=
84 - 20
=
64
```

这就得到了：

```text
当前这个 image placeholder block 长度
```

---

# 29. 为什么不是直接假设一定是 64？

**这是一个不错的防御式写法。**

**虽然正常 Dataset 确实生成 64 个：**

```text
<|image_pad|>
```

但序列存在：

```python
[:self.max_length]
```

截断。当前数据集会**在 tokenize 后截断到 `max_length`。([GitHub][2])**

**假如 image block 靠近截断边界，就可能只留下：**

```text
47 个 image placeholder
```

**所以源码没有写死：**

```python
[:64]
```

而是：

```python
[:i-start]
```

这是**更稳妥的。**

---

# 30. 最关键的一行终于来了

```python
hb = torch.cat(
    (
        hb[:start],
        vf[b][k][:i-start],
        hb[i:]
    ),
    dim=0
)[:seqlen]
```

这行要完全拆开。

---

# 31. 原来的 `hb`

假设：

```text
hb
[T,768]

T=512
```

**图片占据：**

```text
position 20 ~ 83
```

---

## 第一段

```python
hb[:start]
```

即：

```text
hb[:20]
```

Shape：

```text
[20,768]
```

对应：

```text
图片之前的文字 embedding
```

这些完全保留。

---

# 32. 第二段就是图片

```python
vf[b][k][:i-start]
```

先看：

```python
vf[b]
```

如果：

```text
vf
[B,N_img,64,768]
```

那么：

```text
vf[b]
[N_img,64,768]
```

然后：

```python
vf[b][k]
```

取当前第 `k` 张图片：

```text
[64,768]
```

最后：

```python
[:i-start]
```

正常：

```text
i-start = 64
```

所以：

```text
[64,768]
```

---

# 33. 第三段

```python
hb[i:]
```

即：

```text
hb[84:]
```

Shape：

```text
[512-84,768]
=
[428,768]
```

对应：

```text
图片 placeholder 后面的文字 embedding
```

也完全保留。

---

# 34. 然后三段拼起来

```text
文字 before
[20,768]

+

图片 visual embeddings
[64,768]

+

文字 after
[428,768]

=

[512,768]
```

所以：

```text
sequence length
完全没有变化
```

这点极其重要。

它不是：

```text
原 512 token
+
额外 64 visual tokens
=
576
```

而是：

```text
原来的 64 个 placeholder
被 64 个 visual token 一比一替换
```

所以仍然：

```text
512
```

---

# 35. 用一个更直观的图

原来 `input_ids`：

```text
位置
0     1     2     ...   20     21     ...   83    84   ...
│     │     │            │                   │      │
BOS   文本  文本         IMG    IMG ...     IMG    文本
                       id=12              id=12
```

第一次 embedding：

```text
E(BOS)
E(text)
E(text)
...
E(12)
E(12)
...
E(12)
E(text)
```

**然后：**

```text
                 ↓ count_vision_proj
```

**变成：**

```text
E(BOS)
E(text)
E(text)
...
V1
V2
V3
...
V64
E(text)
```

其中：

```text
Vi ∈ R^768
```

---

# 36. 所以所谓“多模态融合”在 MiniMind-V 里其实发生得非常早

它属于：

> **embedding-level fusion / input-level fusion**

**不是在第 4 层 Transformer 才融合。**

**也不是：**

```text
Text Decoder
    ↑
Cross Attention
    ↑
Vision Encoder
```

**而是：**

```text
Text Embedding ─┐
                ├→ 一个统一的 embedding sequence
Visual Embedding┘
                         ↓
                 Decoder-only LLM
```

这就是 MiniMind-V 的核心架构。

---

# 37. 这是和很多 Encoder-Decoder / Cross-Attention VLM 最大的区别之一

一种模型可以做：

```text
Vision Encoder
      ↓
visual features
      ↓
Cross Attention ← Text Decoder
```

那意味着 **decoder 内部专门增加：**

```text
Cross Attention Layer
```

**MiniMind-V 没这么干。**

它采用的逻辑更加接近：

```text
LLaVA-style projector fusion
```

即：

```text
Vision Encoder
↓
Projector
↓
变成“像词 embedding 一样”的向量
↓
直接塞进 token sequence
```

因此原来的 Decoder-only architecture **改动非常少。**

---

# 38. 这也解释了为什么 Projector 输出必须是 `C_llm`

原**文本 embedding：**

```text
[T_text,768]
```

**visual embedding 如果还是：**

```text
[64,1024]
```

**你根本没办法做：**

```python
torch.cat(..., dim=0)
```

**因为最后一维必须一样。**

也就**是说必须：**

```text
C_visual
↓ Projector
C_llm
```

**最终：**

```text
Text:
[...,768]

Vision:
[...,768]
```

**才能拼在同一 sequence 维度。**

**所以你现在应该对前一轮：**

```text
Vision Projector 为什么存在？
```

**有了更深一层理解：**

**它不只是抽象意义上的“对齐”。**

**它在工程上也必须做到：**

```text
让 visual token 和 text token
拥有同一个 hidden dimension
```

---

# 39. `k += 1` 又是什么？

**替换完第一段图片：**

```python
k += 1
```

意味着：

```text
下一组连续的 image placeholders
↓
对应下一张图片
```

例如：

```text
用户：
看看 <image> 和 <image> 有什么区别？
```

Dataset 会得到：

```text
[64个 image_pad]
和
[64个 image_pad]
```

扫描时：

```text
第一组
→ k=0
→ vf[b][0]

第二组
→ k=1
→ vf[b][1]
```

所以：

```text
placeholder block
```

和：

```text
image
```

**按照出现顺序对应。**

---

# 40. 多图完整 Shape 现在你应该能看懂

假设：

```text
B = 4
每个样本 2 张图片
```

**Processor 后：**

```text
pixel_values
[4,2,3,256,256]
```

**flatten：**

```text
[8,3,256,256]
```

**SigLIP：**

```text
[8,64,768]
```

**Projector：**

```text
[8,64,768]
```

**reshape：**

```text
[4,2,64,768]
```

然后**对第 `b` 个样本：**

```text
vf[b]
[2,64,768]
```

**第一段 image placeholder：**

```text
vf[b][0]
[64,768]
```

**第二段：**

```text
vf[b][1]
[64,768]
```

最后 **`hb` 仍然：**

```text
[T,768]
```

**batch stack 后：**

```text
[B,T,768]
```

---

# 41. `return torch.stack(out)`

每个：

```text
hb
[T,768]
```

放入：

```python
out
```

最后：

```python
torch.stack(out)
```

得到：

```text
[B,T,768]
```

这就是最终的：

# Multimodal Embedding

---

# 42. 回到 `forward()` 最关键的一行

当前调用：

```python
hidden_states = self.count_vision_proj(
    tokens=input_ids,
    h=hidden_states,
    vision_tensors=vision_tensors,
    seqlen=input_ids.shape[1]
)
```

([GitHub][1])

调用之前：

```text
hidden_states
=
纯 tokenizer embedding

[B,T,768]
```

调用之后：

```text
hidden_states
=
text embedding
+
visual embedding 原位替换后的多模态序列

[B,T,768]
```

所以：

```text
Shape 没变

但内容变了
```

这个概念你要特别熟悉：

> **Tensor shape 没变，不代表计算没有发生重大变化。**

---

# 43. 此时原 MiniMind Transformer 根本不需要改输入接口

接下来的源码就是：

```python
for layer_idx, (layer, past_key_value) in enumerate(...):
    hidden_states, present = layer(
        hidden_states,
        position_embeddings,
        ...
    )
```

([GitHub][1])

也就是你已经学过的：

```text
MiniMindBlock × N
```

它看到的只是：

```text
hidden_states
[B,T,768]
```

它不知道：

```text
这个 position 是汉字
那个 position 是英文
那 64 个 position 是图片
```

对 Transformer 来说：

> 全部只是 `[768]` 向量。

---

# 44. 所以 Transformer 怎么“看图”？

现在你可能会自然问：

> 那 Transformer 又不知道哪些是图，怎么理解图片？

因为：

```text
视觉 token
已经被 projector 投进 LLM hidden space
```

然后 self-attention 可以直接让：

```text
文字 Q
```

去关注：

```text
视觉 K/V
```

例如序列：

```text
[V1][V2]...[V64][请][问][图][中][是][什][么]
```

某层 Attention 中：

```text
“图”
↓ Q

可以 attend 到
V3 / V8 / V17 / V42 ...
```

而：

```text
V17
```

本质来自图片某个 patch。

于是：

```text
Text token
↔
Visual token
```

通过原来完全普通的：

```text
Self-Attention
```

发生交互。

---

# 45. 这一步跟你之前学 Q/K/V 正好完全连接

以前纯 MiniMind：

```text
token sequence

我 爱 猫
↓
Q/K/V
↓
各文字之间 Attention
```

现在：

```text
visual token 1
visual token 2
...
visual token 64
请
描述
图片
```

整体一起：

```text
[B,T,768]
```

进入同样：

```text
Q = Wq h
K = Wk h
V = Wv h
```

所以：

```text
视觉 token
也会产生 Q/K/V

文本 token
也会产生 Q/K/V
```

然后整个 sequence 在 causal self-attention 里统一计算。

这就是：

> **图片真正进入语言模型内部计算的时刻。**

---

# 46. 一个重要问题：视觉 token 有自己的 tokenizer id 吗？

回答需要稍微精确一点。

### 占位阶段

有：

```text
<|image_pad|>
→ token id 12
```

但：

### 真正进入 Transformer 时

Transformer 接收到的已经不是：

```text
token id 12
```

而是：

```text
projected visual embedding
```

所以真正的图片内容：

> **不是被 tokenizer 离散成 64 个 vocabulary tokens。**

而是：

```text
Image
↓
Vision Encoder
↓
连续浮点 vector
↓
Projector
↓
连续浮点 vector
↓
LLM
```

这叫连续视觉表示。

---

# 47. 因此视觉 token 和普通文字 token 有一个本质区别

文字：

```text
"猫"
↓
tokenizer
↓
id = xxx
↓
Embedding Table
↓
768-D vector
```

图片：

```text
patch
↓
SigLIP Vision Transformer
↓
768-D feature
↓
Projector
↓
768-D vector
```

最后两条路线汇合：

```text
                     [768]
                      │
文字 → Embedding ─────┤
                      │
图片 → Projector ─────┘
                      ↓
               Self-Attention
```

这张图应该成为你现在脑中的 MiniMind-V 核心结构。

---

# 48. 那 placeholder 的 embedding 参数还重要吗？

对真正有图片且正常替换的位置：

```text
Embedding(12)
```

会被直接替换掉。

因此这些位置实际进入 Transformer 的不是它。

所以你不能误以为：

```text
id=12 的 embedding
=
图片 embedding
```

不是。

它更像：

```text
地址标签
```

负责告诉代码：

> “**这一段内存位置请换成视觉 feature**。”

---

# 49. 为什么不用直接把 visual embeddings append 到前面？

理论上当然也可以设计：

```text
visual tokens
+
text tokens
```

但那样 **Dataset 里的文本 sequence 和 multimodal sequence length 就不一致：**

```text
input_ids T

但是 embeddings:
T+64
```

随后：

```text
labels
attention mask
position
```

**都要额外调整。**

**MiniMind-V 当前通过：**

```text
提前在 token sequence 中放 64 个 placeholders
```

**让：**

```text
input_ids 长度
labels 长度
embedding 长度
position 长度
```

**天然统一：**

```text
T
```

**工程上非常简单。**

---

# 50. 这也解释了为什么 Dataset 极其重要

现在你已经能发现：

`model_vlm.py` 并**不能单独成立。**

**它依赖 Dataset 保证：**

```text
每张图片
↔
一段连续 image marker
```

**而 Dataset 当前确实把每个：**

```text
<image>
```

**换成默认 64 个 `<|image_pad|>`。([GitHub][2])**

所以真正完整链条已经露出来：

```text
Dataset
创建 64 placeholders

↓

Tokenizer
变成 64 个 id=12

↓

Embedding
暂时变成 64 个 Embedding(12)

↓

Vision Encoder
图片 → 64 visual features

↓

Projector
64 visual features
→ 64 LLM-space embeddings

↓

count_vision_proj
找到 64 个 id=12 的位置

↓

替换

↓

Multimodal Embedding
[B,T,768]
```

---

# 51. 为什么 Labels 不需要跟着 visual embeddings 改？

这个问题非常关键，我们下一轮 Dataset 会详细讲。

先给核心：

```text
input_ids 的位置没变
```

因为是**原位替换 embedding**。

也就是说：

```text
位置 20~83
```

替换前是：

```text
image placeholder
```

替换后仍是：

```text
position 20~83
```

只是 embedding value 不一样。

所以：

```text
labels
[T]
```

无需因为融合视觉向量而改变长度。

而视觉 placeholder 区域正常不会要求模型去预测真实“图片 token 内容”；SFT 的 label masking 是在 Dataset 阶段处理的，我们下轮会逐行看 `generate_labels()`。

---

# 52. 再理解 `start_pos == 0`

现在有了完整结构后再回看它，你应该更容易理解。

### Prefill

输入：

```text
64 visual placeholders + question
```

执行：

```text
Image
→ Vision Encoder
→ Projector
→ 替换
→ Transformer
```

Transformer 对 visual tokens 也生成：

```text
K/V
```

并缓存。

---

### Decode

下一 token：

```text
只有新的 token embedding
```

以前图片位置的信息已经存在：

```text
KV Cache
```

所以不需要：

```text
再次 Vision Encoder
```

因此源码：

```python
if pixel_values is not None and start_pos == 0:
```

就是：

> **视觉编码只在 Prefill 阶段做。**

这和你之前问过的：

```text
KV Cache 到底缓存什么？
```

现在终于和多模态连接起来了。

---

# 53. MiniMind → MiniMind-V 对照到这里非常清楚了

## MiniMind

```text
input_ids
[B,T]

↓ embed_tokens

[B,T,768]

↓ Transformer

[B,T,768]

↓ lm_head

[B,T,V]
```

---

## MiniMind-V

```text
input_ids
[B,T]

↓ embed_tokens

text embeddings
[B,T,768]
        │
        │
Image   │
↓       │
Processor
↓
pixel_values
↓
SigLIP
[B,64,768]
↓
Projector
[B,64,768]
        │
        ↓
count_vision_proj()
        │
        ↓
multimodal embeddings
[B,T,768]

↓ 完全继续原 MiniMind Transformer

[B,T,768]

↓ lm_head

[B,T,V]
```

所以真正发生变化的核心入口只有：

```text
Embedding 与 Transformer 之间
```

插了一次：

# visual embedding replacement

---

# 54. 你现在应该形成的源码阅读反应

以后你看到：

```python
embed_tokens(input_ids)
```

不要只想到：

> Token → Embedding。

在 VLM 里还要马上问：

> **有没有在 embedding 后做 visual feature injection？**

---

看到：

```python
image_ids
```

第一反应：

> 这是拿来**定位视觉占位位置**的，**不是 Vision Encoder 的分类 ID。**

---

看到：

```python
count_vision_proj
```

第一反应：

> 找**连续 image-marker block → 对应第 k 张图 → 用 visual embeddings 原位替换。**

---

看到：

```text
[B,64,768]
→
[B,T,768]
```

**不要误解成 reshape。**

**真正是：**

```text
[64,768]
被放进
[T,768]
的某 64 个位置
```

---

**看到：**

```python
start_pos == 0
```

第一反应：

```text
Prefill 时处理视觉
Decode 时依靠 KV Cache
```

---

# 55. 一个你现在应该能自己回答的问题

假设：

```text
T = 512

图片占用 64 visual tokens
```

图片**会不会额外让 Transformer 实际处理：**

```text
512 + 64 = 576
```

**？**

**不会。**

**因为 Dataset 已经提前把：**

```text
64 个 placeholder
```

**包含在这 512 个 token 里。**

所以**实际仍然：**

```text
[B,512,768]
```

**但这也意味着：**

> **图片实际上吃掉了 64 个 context positions。**

**例如 `max_length=512` 时：**

```text
1 张图片：
约占 64 positions

2 张图片：
约占 128 positions
```

**剩给文本的 context budget 就更少。**

**这件事以后你研究：**

```text
Visual Token Compression / VisionZip
```

**时会非常重要。**

**你未来研究 VisionZip，本质上很可能就在优化这一矛盾：**

```text
视觉 token 越多
→ 视觉信息越丰富
→ 计算越贵
→ 占 context 越多

视觉 token 越少
→ 更省算力/上下文
→ 可能丢失视觉信息
```

所以你现在学的 MiniMind-V **64 visual tokens**，以后会直接成为理解 visual token compression 的前置。

---

# 56. 本轮你只需要牢牢记住这 6 件事

1. `<|image_pad|>` 是**位置占位符**，不是实际图片内容。

2. Dataset 默认给每张图片预留 **64 个连续 placeholder**。([GitHub][2])

3. 图片经过：

```text
SigLIP → Projector
```

得到：

```text
[64,768]
```

4. 整条文本先正常：

```text
input_ids
[B,T]
→
embed_tokens
[B,T,768]
```

5. `count_vision_proj()` 找到那 64 个 placeholder，对应 embedding **原位换成 visual embeddings**。([GitHub][1])

6. 替换后仍然：

```text
[B,T,768]
```

所以后面的 MiniMind Decoder 基本照搬；视觉与文字靠原来的 **Self-Attention** 发生交互。

---

## 下一轮

下一轮最合适的不是继续往 `logits/loss` 冲，而是按照你最开始要求的路线，正式进入：

```text
dataset/lm_dataset.py
└── VLMDataset
```

把刚才模型端的另一半完全接起来：

```text
parquet 一条数据
↓
conversations
↓
image_bytes
↓
<image>
↓
64 × <|image_pad|>
↓
chat template
↓
tokenizer
↓
input_ids
↓
generate_labels()
↓
image2tensor()
↓
SiglipImageProcessor
↓
image_data
↓
vlm_collate_fn
↓
[B,T]
[B,T]
[B,N,3,256,256]
↓
MiniMindVLM.forward()
```

而且这里会重点解释一个现在已经必须搞懂的问题：

> **既然输入里有 64 个 image placeholder，那么这些位置的 `labels` 到底是什么？为什么不会让模型学着“输出 64 个 `<|image_pad|>`”？**

这正好把 **Dataset → multimodal fusion → loss** 完整闭环起来。

可以。我们这一轮正式把 **Dataset 端接到上一轮的 `MiniMindVLM.forward()`**。

先纠正一个当前 `master` 的版本点：你原来的衔接说明里写的是

```text
dataset/lm_dataset.py
├── VLMDataset
└── vlm_collate_fn
```

但我刚重新核对了当前源码：

```text
dataset/lm_dataset.py
└── VLMDataset

trainer/trainer_utils.py
└── vlm_collate_fn
```

也就是说 `vlm_collate_fn` **现在已经移动到 `trainer/trainer_utils.py`**；`train_pretrain_vlm.py` 和 `train_sft_vlm.py` 都从那里 import。当前 Pretrain 与 SFT 也确实共用同一个 `VLMDataset` 和同一个 `vlm_collate_fn`。仓库当前训练数据已经统一为 Parquet 图文一体化格式。([GitHub][1])

---

# MiniMind-V 源码带读（三）

## `VLMDataset → labels → image_data → vlm_collate_fn`

我们这轮最终要把这条链路真正打通：

```text
Parquet 一条样本
│
├── conversations
└── image_bytes
       ↓
VLMDataset.__getitem__()
       │
       ├── <image>
       │      ↓
       │  64 × <|image_pad|>
       │
       ├── Chat Template
       │      ↓
       │  input_ids [T]
       │
       ├── generate_labels()
       │      ↓
       │  labels [T]
       │
       └── image_bytes
              ↓
          PIL.Image
              ↓
       SiglipImageProcessor
              ↓
     pixel_values [N_img,3,256,256]
              ↓
         vlm_collate_fn
              ↓
input_ids    [B,T]
labels       [B,T]
pixel_values [B,N_img,3,256,256]
              ↓
       MiniMindVLM.forward()
```

---

# 1. 先打开源码

当前文件：

```text
jingyaogong/minimind-v
└── dataset/
    └── lm_dataset.py
        └── class VLMDataset(Dataset)
```

类定义当前是：

```python
class VLMDataset(Dataset):
    def __init__(
        self,
        parquet_path,
        tokenizer,
        preprocess=None,
        max_length=512,
        image_special_token='<|image_pad|>',
        image_token_len=64
    ):
        super().__init__()
        self.dataset = HFDataset.from_parquet(parquet_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess = preprocess
        self.image_special_token = image_special_token * image_token_len

        self.bos_id = tokenizer(
            f'{tokenizer.bos_token}assistant\n',
            add_special_tokens=False
        ).input_ids

        self.eos_id = tokenizer(
            f'{tokenizer.eos_token}\n',
            add_special_tokens=False
        ).input_ids
```

这一小段其实已经把：

```text
文本
图片占位符
监督区域
```

三件事情准备好了。

---

# 2. `HFDataset.from_parquet(parquet_path)`

```python
self.dataset = HFDataset.from_parquet(parquet_path)
```

这里用的**不是：**

```text
torch.utils.data.Dataset
```

**去直接解析 parquet。**

而是先：

```text
HuggingFace datasets.Dataset
```

**读取：**

```text
pretrain_i2t.parquet
```

**或者：**

```text
sft_i2t.parquet
```

**README 当前也明确说明数据已经改成 Parquet，把图像和文本存放在同一份文件里，不再需要以前几十万张图片文件单独解压。([GitHub][1])**

从当前代码能确定，**一行样本至少需要：**

```text
row
├── conversations
└── image_bytes
```

**因为后面直接：**

```python
row['conversations']
row['image_bytes']
```

---

# 3. 一条样本概念上是什么样

**不要把它理解成传统：**

```text
image.jpg
caption.txt
```

**现在更接近：**

```python
{
    "conversations": [
        {
            "role": "user",
            "content": "<image>\n请描述这张图片"
        },
        {
            "role": "assistant",
            "content": "这是一只站在草地上的鹿。"
        }
    ],

    "image_bytes": ...
}
```

注意这个例子只是**为了理解字段结构；具体真实数据文本由 Parquet 数据集决定。**

**最重要的是：**

```text
conversations
```

**里面本身只是：**

```text
<image>
```

**还不是：**

```text
64 个 <|image_pad|>
```

**转换是在 Dataset 里发生的。**

---

# 4. `self.image_special_token`

关键代码：

```python
self.image_special_token = image_special_token * image_token_len
```

默认：

```text
image_special_token
=
<|image_pad|>

image_token_len
=
64
```

因此实际上**得到一个字符串：**

```text
<|image_pad|><|image_pad|>...<|image_pad|>
```

**连续 64 个。**

**不是 Python list：**

```python
["<|image_pad|>", ...]
```

**而是：**

```text
一个很长的字符串
```

**里面把特殊 token 连续重复了 64 次。**

**这就和我们上一轮模型端：**

```text
64 visual tokens
```

第一次真正接上了：

```text
Vision Encoder
产生 64 个 visual tokens

            ↕

Dataset
提前留下 64 个 placeholder positions
```

---

# 5. 下面两个变量名字其实有点容易误导

```python
self.bos_id = tokenizer(
    f'{tokenizer.bos_token}assistant\n',
    add_special_tokens=False
).input_ids
```

以及：

```python
self.eos_id = tokenizer(
    f'{tokenizer.eos_token}\n',
    add_special_tokens=False
).input_ids
```

看到：

```text
bos_id
eos_id
```

你**不要立即理解成：**

```text
一个 BOS token id
一个 EOS token id
```

**因为这里返回的是：**

```text
list[int]
```

**例如概念上：**

```text
bos_id
=
tokenize("<BOS>assistant\n")
```

所以它**实际上是：**

> **“assistant 回复开始标记”的 token 序列。**

**同理：**

```text
eos_id
```

是：

> **assistant 回复结束附近的 token 序列。**

---

# 6. 为什么 Dataset 要知道 assistant 从哪里开始？

因为 **SFT 不能简单地：**

```text
整段聊天全部计算 loss
```

**例如：**

```text
User:
这张图片是什么？

Assistant:
这是一只猫。
```

**训练真正想监督的是：**

```text
Assistant:
这是一只猫。
```

**而通常不需要让模型学习：**

```text
预测用户输入本身
```

**所以最终希望 labels 是：**

```text
User 部分
-100 -100 -100 ...

Image 部分
-100 -100 ... -100

Assistant answer
真实 token ids
```

这**正是后面的：**

```text
generate_labels()
```

**要做的事情。**

---

# 7. 先看 `create_chat_prompt()`

当前源码：

```python
def create_chat_prompt(self, conversations):
    messages = []

    for turn in conversations:
        content = (
            turn['content'].replace(
                '<image>',
                self.image_special_token
            )
            if turn.get('role') != 'system'
            else turn['content']
        )

        messages.append({
            "role": turn['role'],
            "content": content
        })

    tools = (
        conversations[0]["functions"]
        if (
            conversations
            and conversations[0]["role"] == "system"
            and conversations[0].get("functions")
        )
        else None
    )

    return self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=tools
    )
```

这部分非常重要。

---

# 8. `<image>` 到底在哪里变 64 个 placeholder？

就是：

```python
turn['content'].replace(
    '<image>',
    self.image_special_token
)
```

**假设：**

```text
原始：

<image>
这是什么动物？
```

**经过这里：**

```text
<|image_pad|>
<|image_pad|>
...
共64个
...
<|image_pad|>
这是什么动物？
```

**所以：**

```text
Parquet
```

**本身不需要储存 64 个特殊 token。**

**只需要储存：**

```text
<image>
```

**Dataset 动态展开。**

---

# 9. 为什么 `system` 不做 replace？

代码：

```python
if turn.get('role') != 'system'
```

**所以：**

```text
user
assistant
其他非-system role
```

**里的 `<image>` 会被替换。**

**但：**

```text
system
```

**不替换。**

**这反映了当前数据设计默认：**

> 图片**主要作为实际会话内容，而不是 system prompt 中的视觉占位。**

---

# 10. `apply_chat_template()` 又干了什么？

现在：

```python
messages
```

还是**结构化数据：**

```python
[
    {"role": "user", ...},
    {"role": "assistant", ...}
]
```

然后：

```python
self.tokenizer.apply_chat_template(...)
```

将其**变成 tokenizer 规定的聊天格式字符串。**

**概念上相当于：**

```text
BOS / role markers / user
...
assistant
...
EOS
```

但这里**不要凭通用经验猜具体模板格式，因为：**

> **实际格式由 MiniMind 当前 tokenizer 的 chat template 决定。**

源码这里明确：

```python
tokenize=False
```

所以：

```text
apply_chat_template
```

此时还只是：

```text
结构化 messages
↓
字符串 prompt
```

还**没变 token ids。**

---

# 11. 为什么 `add_generation_prompt=False`

因为**这里不是：**

```text
我要让模型开始推理生成答案
```

**而是：**

```text
训练样本中答案本来就已经存在
```

**所以完整：**

```text
user + assistant response
```

**都应该放进 prompt。**

如果**是推理时，才经常需要额外添加：**

```text
assistant:
```

**提示模型从这里继续生成。**

---

# 12. Dataset 前面还有两个预处理函数

当前文件上面还有：

```python
pre_processing_chat()
```

和：

```python
post_processing_chat()
```

我们不展开成单独理论，但要**知道真实训练数据经过它们。**

---

## `pre_processing_chat()`

其中：

```python
if any(conv.get('tools') for conv in conversations):
    return conversations
```

如果**存在 tool use 数据：**

```text
保持原样
```

**否则，如果第一条不是 system：**

```python
if random.random() < add_system_ratio:
```

**默认：**

```text
add_system_ratio = 0.2
```

**即大约 20% 概率给训练样本前面补一个随机 system prompt。**

例如：

```text
你是minimind，一个小巧但有用的语言模型。
```

---

# 13. 为什么做这个 augmentation？

本质上相当于一种：

```text
conversation formatting augmentation
```

让模型**不要过度依赖：**

```text
“永远没有 system prompt”
```

**这种固定模式。**

不过这**属于训练数据策略，不是 VLM 核心机制。**

**未来它也可以成为实验变量，但现在先不碰。**

---

# 14. `post_processing_chat()`

当前代码：

```python
if '<think>\n\n</think>\n\n' in prompt_content \
        and random.random() > empty_think_ratio:

    prompt_content = prompt_content.replace(
        '<think>\n\n</think>\n\n',
        ''
    )
```

默认：

```text
empty_think_ratio = 0.2
```

所以**约 80% 情况下：**

```text
空的 <think></think>
```

**会被移除。**

**跟 VLM 输入机制本身关系不大。**

---

# 15. 现在进入 `__getitem__()`

当前最核心部分：

```python
def __getitem__(self, index: int):
    row = self.dataset[index]

    conversations = (
        json.loads(row['conversations'])
        if isinstance(row['conversations'], str)
        else row['conversations']
    )

    image_bytes = row['image_bytes']

    if not isinstance(image_bytes, list):
        image_bytes = [image_bytes]
```

这里开始**真正产生：**

```text
一个训练 sample
```

---

# 16. `row = self.dataset[index]`

取出一行 Parquet：

```text
row
├── conversations
└── image_bytes
```

其中 `conversations` 有可能**本身存成 JSON 字符串：**

```python
'[
  {"role":"user", ...},
  ...
]'
```

所以：

```python
json.loads(...)
```

**转成 Python list/dict。**

**如果本来已经是 list：**

```text
直接用
```

---

# 17. 为什么强制把 `image_bytes` 变成 list？

```python
if not isinstance(image_bytes, list):
    image_bytes = [image_bytes]
```

**假设单图：**

```text
image_bytes
=
bytes
```

**转：**

```text
[bytes]
```

**所以后续无论：**

```text
1 张图
```

**还是：**

```text
3 张图
```

**统一表示：**

```text
List[image_bytes]
```

**这是为了多图兼容。**

---

# 18. 接下来是文本完整流程

源码：

```python
conversations = pre_processing_chat(conversations)

prompt = self.create_chat_prompt(conversations)

prompt = post_processing_chat(prompt)

input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
```

完整过程：

```text
conversations
↓
随机 system augmentation
↓
<image> → 64 image placeholders
↓
Chat Template
↓
完整聊天字符串
↓
Tokenizer
↓
input_ids
```

此时：

```text
input_ids
长度 ≤ max_length
```

---

# 19. 注意这里已经截断

```python
[:self.max_length]
```

比如 **SFT 当前训练脚本：**

```text
max_seq_len = 768
```

**而 Pretrain 当前：**

```text
max_seq_len = 450
```

**这是当前 `master` 训练脚本里的默认值**；README 的近期更新也记录了 max sequence length 的调整。([GitHub][1])

因此：

```text
input_ids
最多：
[768]
```

**或 Pretrain：**

```text
[450]
```

---

# 20. 然后右侧 Padding

```python
input_ids += [self.tokenizer.pad_token_id] * (
    self.max_length - len(input_ids)
)
```

**所以不管原始聊天多长：**

```text
100 tokens
350 tokens
700 tokens
```

最终**每个 sample 都变：**

```text
input_ids
[max_length]
```

**例如 SFT：**

```text
[768]
```

**这一点极大简化了后面的 batch。**

---

# 21. 所以这里跟普通 MiniMind 一个明显不同

你以前学 `PretrainDataset` 时主要是：

```text
纯文本
→ Tokenizer
→ fixed sequence
```

MiniMind-V 现在：

```text
conversation
+
image placeholders
↓
Tokenizer
↓
fixed sequence
```

视觉信息**还没有真正进 `input_ids`：**

```text
这里依然只是 id=12。
```

**真正图像内容仍然走另一条：**

```text
image_bytes
↓
pixel_values
```

---

# 22. 现在是这轮最关键的一部分：`generate_labels()`

源码：

```python
def generate_labels(self, input_ids):
    labels = [-100] * len(input_ids)

    i = 0

    while i < len(input_ids):
        if input_ids[i:i + len(self.bos_id)] == self.bos_id:
            start = i + len(self.bos_id)
            end = start

            while end < len(input_ids):
                if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                    break
                end += 1

            for j in range(
                start,
                min(end + len(self.eos_id), self.max_length)
            ):
                labels[j] = input_ids[j]

            i = (
                end + len(self.eos_id)
                if end < len(input_ids)
                else len(input_ids)
            )
        else:
            i += 1

    return labels
```

这一段必须彻底看懂。

---

# 23. 第一步：全部设成 `-100`

```python
labels = [-100] * len(input_ids)
```

假设：

```text
T = 20
```

一开始：

```text
labels =

[-100, -100, -100, ... -100]
```

也就是**说默认：**

> **所有 token 都不参与 CrossEntropy。**

你已经**知道模型端：**

```python
F.cross_entropy(
    ...,
    ignore_index=-100
)
```

**所以 Dataset 现在是在做：**

```text
Loss Mask
```

---

# 24. 然后扫描哪里出现 assistant

代码：

```python
if input_ids[i:i + len(self.bos_id)] == self.bos_id:
```

**还记得：**

```text
self.bos_id
```

**实际上代表：**

```text
<BOS>assistant\n
```

**对应的一串 token IDs。**

所以它在找：

> **assistant 回复开始的位置。**

---

# 25. 找到以后

```python
start = i + len(self.bos_id)
```

这**意味着：**

```text
assistant role marker
```

**本身也不会参与监督。**

监督从：

```text
assistant 真正回答的第一个 token
```

开始。

例如：

```text
assistant
这 是 一 只 猫
```

**那么：**

```text
start
```

**指向：**

```text
“这”
```

---

# 26. 再寻找 assistant 回复结束

```python
while end < len(input_ids):
    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
        break
    end += 1
```

也就是**一直找：**

```text
EOS
```

**于是得到：**

```text
assistant answer 的范围
```

---

# 27. 然后只把这部分 labels 恢复成真实 token id

```python
for j in range(
    start,
    min(end + len(self.eos_id), self.max_length)
):
    labels[j] = input_ids[j]
```

结果概念上：

```text
System:
-100 -100 -100

User:
-100 -100 -100

Image placeholders:
-100 × 64

Question:
-100 -100 -100

Assistant marker:
-100 ...

Assistant answer:
token_id token_id token_id ...

EOS:
真实 token id
```

所以模型**只对：**

```text
assistant response
```

**计算主要语言建模 loss。**

---

# 28. 这就回答了上轮留下的问题

> 既然 input 中有 64 个 `<|image_pad|>`，为什么模型不会训练成输出 64 个 `<|image_pad|>`？

因为**正常图文问答数据里，图片 placeholder 位于：**

```text
user content
```

**而 user 区域：**

```text
labels = -100
```

**所以：**

```text
64 个 image placeholder
```

**虽然存在于：**

```text
input_ids
```

**里，但：**

```text
不作为 target
```

---

# 29. 这是一个非常关键的区分

图片 **placeholder 有两个作用：**

### 输入侧

```text
负责给 visual embeddings 留位置
```

所以**必须存在：**

```text
input_ids
```

### 输出监督侧

通常：

```text
不需要模型预测它
```

所以：

```text
labels = -100
```

---

# 30. 严谨补一个源码细节

当前：

```python
create_chat_prompt()
```

**对所有：**

```text
role != system
```

**的 `<image>` 都会替换。**

所以理论上**如果数据本身把：**

```text
<image>
```

**写到了 assistant answer 中：**

```text
generate_labels()
```

**是可能把那些位置也设成真实 target 的。**

因**此严格来说：**

> **“image placeholder 一定永远是 -100”**

并**不是由这份代码强制保证的。**

**更准确是：**

> **当前典型 VLM 数据把图像放在 user 输入侧，因此 image placeholder 正常位于非监督区域。**

这个区别以后你做数据清洗时非常重要。

---

# 31. 再把 `shift` 接起来

你已经知道模型最后：

```python
shift_logits = logits[..., :-1, :]
shift_labels = labels[..., 1:]
```

假设：

```text
位置 j
=
assistant answer 的“猫”
```

那么：

```text
labels[j]
=
猫的 token id
```

经过 shift：

```text
position j-1 的 logits
```

去预测：

```text
labels[j]
=
猫
```

所以仍然完全是：

# Next Token Prediction

VLM 没有改变语言模型训练本质。

---

# 32. 视觉 token 虽然自己没有 loss，但仍然能训练 Projector

这是非常关键、也很容易第一次困惑的地方：

> 图片位置都是 `labels=-100`，那 Projector 怎么得到梯度？

因为 Loss 虽然发生在：

```text
assistant answer tokens
```

但 assistant token 的 hidden state 会通过 Self-Attention 依赖：

```text
前面的 visual embeddings
```

例如：

```text
图片 visual tokens
      ↓
Self-Attention
      ↓
“这是一只___”
      ↓
模型预测“猫”
      ↓
CrossEntropy
```

反向传播：

```text
Loss
↓
预测“猫”的 hidden state
↓
Attention
↓
visual embeddings
↓
Vision Projector
```

所以 Projector 仍然拿到 gradient。

这就是 MiniMind-V Pretrain 为什么可以：

```text
冻结 Vision Encoder
冻结 LLM
只训练 Projector
```

但 Projector 依然学会视觉-语言对齐。

---

# 33. 这一点值得你真正理解成计算图

假设：

```text
visual feature
v
```

经过：

```text
z = Projector(v)
```

然后：

```text
h_answer
=
Transformer(
    z,
    text embeddings
)
```

最终：

```text
loss
=
CE(
    LMHead(h_answer),
    target
)
```

所以：

$$
\frac{\partial L}{\partial W_{proj}}
=
\frac{\partial L}{\partial h_{answer}}
\frac{\partial h_{answer}}{\partial z}
\frac{\partial z}{\partial W_{proj}}
$$

即使：

```text
visual position 本身没有 label
```

只要答案依赖它：

```text
Projector 就有梯度。
```

这是你理解 VLM training 必须建立的认知。

---

# 34. 现在处理图片

回到 `__getitem__()`：

```python
image_inputs_list = [
    MiniMindVLM.image2tensor(
        Image.open(io.BytesIO(img)),
        self.preprocess
    )
    for img in image_bytes
]
```

完整过程：

```text
image_bytes
↓
io.BytesIO(img)
↓
PIL.Image.open()
↓
PIL.Image
↓
MiniMindVLM.image2tensor()
```

---

# 35. 回忆当前 `image2tensor()`

`model/model_vlm.py` 当前真实代码：

```python
@staticmethod
def image2tensor(image, processor):
    if image.mode in ['RGBA', 'LA']:
        image = image.convert('RGB')

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    return inputs
```

当前 processor 是：

```text
SiglipImageProcessor
```

由：

```python
SiglipImageProcessor.from_pretrained(...)
```

得到。当前 Vision Encoder 是 P32、固定 256×256 的 SigLIP2 路线。([GitHub][2])

---

# 36. `RGBA → RGB`

如果图片：

```text
RGBA
```

就是：

```text
Red
Green
Blue
Alpha
```

4 通道。

Vision Encoder 需要普通 RGB：

```text
3 channels
```

所以：

```text
RGBA
↓
RGB
```

---

# 37. Processor 做的事情

这里不要把 Processor 理解成 Vision Encoder。

它只是输入预处理：

```text
PIL Image
↓
resize / normalization / tensorization
↓
pixel_values
```

输出是一个 Hugging Face `BatchFeature`，概念上类似：

```python
{
    "pixel_values": Tensor
}
```

当前单张图：

```text
pixel_values
[1,3,256,256]
```

注意前面那个：

```text
1
```

不是训练 batch size。

只是：

> 这次 processor 调用里处理了一张图片。

---

# 38. 多张图时这里发生什么？

假设一个训练样本有：

```text
N_img = 2
```

那么：

```python
image_inputs_list
```

大概：

```text
[
    {"pixel_values": [1,3,256,256]},
    {"pixel_values": [1,3,256,256]}
]
```

然后：

```python
if hasattr(image_inputs_list[0], 'keys'):
```

当前 SigLIP processor 返回字典式 BatchFeature，所以走这里：

```python
image_data = {
    k: torch.cat(
        [inp[k] for inp in image_inputs_list],
        dim=0
    )
    for k in image_inputs_list[0].keys()
}
```

---

# 39. `torch.cat(..., dim=0)`

两张图：

```text
[1,3,256,256]
+
[1,3,256,256]
```

沿 dim 0 拼：

```text
[2,3,256,256]
```

所以单个 Dataset sample 返回的图片部分是：

```text
image_data["pixel_values"]

[N_img,3,256,256]
```

单图时：

```text
[1,3,256,256]
```

---

# 40. 到这里 `__getitem__()` 最终返回什么？

当前源码：

```python
return (
    torch.tensor(input_ids, dtype=torch.long),
    torch.tensor(labels, dtype=torch.long),
    image_data
)
```

所以**一条样本**：

```text
input_ids
[max_length]

labels
[max_length]

image_data["pixel_values"]
[N_img,3,256,256]
```

注意：

> 这里还没有 B。

因为：

```text
Dataset[index]
```

只负责：

```text
一个 sample
```

---

# 41. 接下来 B 从哪里来？

这就是当前已经移到：

```text
trainer/trainer_utils.py
```

里的：

```python
vlm_collate_fn()
```

真实源码：

```python
def vlm_collate_fn(batch):
    input_ids = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    pixel_data = [b[2] for b in batch]

    if hasattr(pixel_data[0], 'keys'):
        pixel_values = {
            k: torch.stack(
                [d[k] for d in pixel_data]
            )
            for k in pixel_data[0].keys()
        }
    else:
        pixel_values = torch.stack(pixel_data)

    return input_ids, labels, pixel_values
```

---

# 42. `batch` 到底是什么？

DataLoader 先调用：

```text
Dataset[index_1]
Dataset[index_2]
Dataset[index_3]
Dataset[index_4]
```

如果：

```text
batch_size = 4
```

那么：

```python
batch
=
[
    sample1,
    sample2,
    sample3,
    sample4
]
```

每个：

```text
sample
=
(input_ids, labels, image_data)
```

---

# 43. 文本 stack

```python
input_ids = torch.stack(
    [b[0] for b in batch]
)
```

每条：

```text
[T]
```

4 条：

```text
[T]
[T]
[T]
[T]
```

stack：

```text
[B,T]
```

例如 SFT：

```text
[4,768]
```

---

# 44. Labels 同样

```python
labels = torch.stack(
    [b[1] for b in batch]
)
```

得到：

```text
[B,T]
```

例如：

```text
[4,768]
```

---

# 45. 图片怎么 stack？

单图情况下，每条 sample：

```text
pixel_values

[1,3,256,256]
```

4 条 stack：

```text
[B,1,3,256,256]
```

即：

```text
[4,1,3,256,256]
```

每个维度：

```text
B
│
├── N_img
│
├── C = 3
│
├── H = 256
│
└── W = 256
```

---

# 46. 所以训练脚本真正拿到的 batch

当前：

```python
for step, (
    input_ids,
    labels,
    pixel_values
) in enumerate(loader):
```

得到：

```text
input_ids
[B,T]

labels
[B,T]

pixel_values["pixel_values"]
[B,N_img,3,256,256]
```

然后：

```python
model(
    input_ids,
    labels=labels,
    pixel_values=pixel_values
)
```

正式进入我们上一轮已经学过的：

```text
MiniMindVLM.forward()
```

---

# 47. 现在你应该能看懂上一轮那个 `ndim == 5`

模型里：

```python
sample_val = next(iter(pixel_values.values()))

if sample_val.ndim == 5:
```

为什么恰好是 5 维？

因为 Dataset + collate 给出来的正是：

```text
[B,N_img,3,H,W]
```

五维。

例如：

```text
[4,1,3,256,256]
```

---

# 48. 然后模型里为什么 `flatten(0,1)`

上一轮：

```python
{k: v.flatten(0, 1) for k, v in pixel_values.items()}
```

现在完全能看懂了。

原来：

```text
[B,N_img,3,256,256]
```

flatten：

```text
[B × N_img,3,256,256]
```

因为 SigLIP 不关心：

```text
“这张图片属于 batch 里的哪个聊天”
```

它只负责：

```text
给我图片
→ 我编码
```

之后再：

```python
.view(
    bs,
    num,
    image_token_len,
    -1
)
```

恢复：

```text
[B,N_img,64,768]
```

---

# 49. 现在完整 shape 已经闭环了

假设：

```text
B = 4
T = 768
N_img = 1
```

Dataset：

```text
单条：

input_ids
[768]

labels
[768]

pixel_values
[1,3,256,256]
```

Collate：

```text
input_ids
[4,768]

labels
[4,768]

pixel_values
[4,1,3,256,256]
```

模型：

```text
pixel_values
[4,1,3,256,256]

↓ flatten

[4,3,256,256]

↓ SigLIP

[4,64,768]

↓ Projector

[4,64,768]

↓ view

[4,1,64,768]
```

文本：

```text
input_ids
[4,768]

↓ Embedding

[4,768,768]
```

然后：

```text
count_vision_proj()
```

把每条样本：

```text
64 × id=12 的 positions
```

对应 embedding 换成：

```text
[64,768]
```

最终仍：

```text
hidden_states
[4,768,768]
```

---

# 50. 终于把整个 MiniMind-V 数据输入真正闭环了

现在从磁盘开始，一条完整真实主线是：

```text
pretrain_i2t.parquet / sft_i2t.parquet
│
├──────────────────────────────┐
│                              │
conversations                image_bytes
│                              │
↓                              ↓
json/list                    BytesIO
│                              ↓
↓                           PIL.Image
pre_processing_chat            │
│                              ↓
↓                       SiglipImageProcessor
<image>                          │
↓                                ↓
64 × <|image_pad|>          pixel_values
│                         [N_img,3,256,256]
↓                                │
apply_chat_template              │
│                                │
↓                                │
prompt                           │
│                                │
↓                                │
Tokenizer                        │
│                                │
↓                                │
input_ids [T]                    │
│                                │
├── generate_labels             │
│      ↓                         │
│   labels [T]                   │
│                                │
└────────────┬───────────────────┘
             ↓
        Dataset sample
             ↓
       vlm_collate_fn
             ↓
┌─────────────────────────────────────┐
│ input_ids     [B,T]                 │
│ labels        [B,T]                 │
│ pixel_values  [B,N_img,3,256,256]   │
└─────────────────────────────────────┘
             ↓
       MiniMindVLM.forward
             │
       ┌─────┴─────┐
       │           │
 input_ids     pixel_values
       │           │
 Embedding       SigLIP
       │           ↓
       │        Projector
       │           │
       └─────┬─────┘
             ↓
 count_vision_proj
             ↓
 [B,T,768]
             ↓
 MiniMind Transformer
             ↓
 logits [B,T,V]
             ↓
 shift logits / labels
             ↓
 CrossEntropy
 ignore_index=-100
             ↓
           LOSS
```

这张图现在应该是你目前对 MiniMind-V 最重要的一张脑图。

---

# 51. 一个当前源码里值得你注意的工程点

现在的：

```python
vlm_collate_fn
```

只是简单：

```python
torch.stack(...)
```

图片没有额外做：

```text
num_images padding
```

所以 batch 中各个 sample 的图片 Tensor shape 必须可 stack。

也就是说如果：

```text
sample 1
[1,3,256,256]

sample 2
[3,3,256,256]
```

直接：

```python
torch.stack()
```

会 shape 不一致。

因此当前实现虽然模型层面支持：

```text
N_img > 1
```

但当前简单 collate 对“同一 batch 中每条样本图片数量不同”的情况没有复杂的动态 padding 机制。

这是一个非常典型的：

> **模型能力 ≠ 数据流水线完整支持能力**

以后你读科研仓库时一定要形成这种检查习惯。

---

# 52. 另一个值得注意的地方：没有返回 `attention_mask`

当前 Dataset 返回：

```text
input_ids
labels
image_data
```

没有：

```text
attention_mask
```

训练脚本调用：

```python
model(
    input_ids,
    labels=labels,
    pixel_values=pixel_values
)
```

也没有显式传 attention mask。

而文本采用：

```text
右侧 PAD 到固定 max_length
```

PAD 位置的：

```text
labels
```

保持：

```text
-100
```

所以：

```text
PAD 不产生 loss
```

但是固定 padding 仍然会消耗一定前向计算。

这也是以后真正做：

```text
训练效率优化
```

时可以关注的地方，但目前不要改。

---

# 53. MiniMind 和 MiniMind-V Dataset 对照

你以前的 MiniMind 可以想成：

```text
Text
↓
Tokenizer
↓
input_ids
↓
labels
```

MiniMind-V 现在只是新增了两件核心事情。

### 新增 1：文本里留视觉位置

```text
<image>
↓
64 × <|image_pad|>
↓
Tokenizer
↓
64 × id 12
```

### 新增 2：图片本身另外编码

```text
image_bytes
↓
PIL
↓
Processor
↓
pixel_values
```

所以：

```text
MiniMind-V Dataset
=
MiniMind 文本数据流
+
image placeholder
+
pixel_values
```

这正符合你一开始希望建立的：

```text
MiniMind-V
不是一个完全陌生模型
```

---

# 54. 你现在应该形成的源码阅读反应

以后看到：

```python
image_special_token * image_token_len
```

第一反应：

> **在文本 sequence 中提前购买 N 个视觉 token 的位置。**

看到：

```python
generate_labels()
```

第一反应：

> **哪些 token 真正贡献 supervised loss？**

看到：

```text
labels = -100
```

第一反应：

> **这个 position 可以参与 forward / attention，但不直接成为 CE target。**

这个区别极其重要：

```text
不计算自身 loss
≠
这个 token 对训练没作用
```

视觉 token 正是最典型的例子。

看到：

```python
processor(images=image)
```

第一反应：

> **这只是 image preprocessing，不是 Vision Encoder forward。**

看到：

```python
torch.stack
```

第一反应：

> **被 stack 的每个 sample shape 是否完全一致？**

看到：

```text
[B,N_img,3,H,W]
```

第一反应：

> 模型后面大概率要把：

```text
B × N_img
```

flatten 后统一送 Vision Encoder。

---

# 55. 到这里你应该真正回答得出的三个问题

### Q1：图片本身有没有经过 tokenizer？

**没有。**

Tokenizer 只处理：

```text
<|image_pad|>
```

这种文本占位符。

真正图片走：

```text
PIL
→ SiglipImageProcessor
→ SigLIP
```

---

### Q2：那为什么 input_ids 里还需要 image token？

因为它用来规定：

```text
visual embedding
在 LLM sequence 的哪个位置出现。
```

---

### Q3：图片位置没有语言模型标签，Projector 为什么还能训练？

因为：

```text
答案 token 的预测
依赖 visual embedding
```

Loss 可以通过：

```text
答案
→ Self-Attention
→ Visual Embedding
→ Projector
```

反向传播。

这三个问题如果你现在都完全能解释，说明 **MiniMind-V 最核心的数据与融合链路你已经真正接通了。**

---

## 下一轮最合适进入哪里

到目前为止，我们已经把：

```text
Dataset
→ pixel_values
→ Vision Encoder
→ Projector
→ visual replacement
```

全部接通了。

下一轮就应该回到 `model/model_vlm.py` 的后半段，把：

```text
Multimodal Hidden States
↓
RoPE
↓
原 MiniMind Blocks
↓
Final Norm
↓
LM Head
↓
logits
↓
shift
↓
CrossEntropy
↓
梯度到底更新谁
```

完整闭环。

其中最值得重点讲的是：

```text
为什么 visual position 没有 label
但 Projector 仍得到梯度；

freeze_llm=2 时
Loss → LLM → visual embedding → Projector
的梯度到底怎么走；

被冻结的 LLM 虽然不更新参数，
为什么仍然必须参与 backward 的计算图。
```

这一轮学完，我们就基本完成 **MiniMind-V 单次训练 step 的完整模型链路**，之后才适合正式进入 `train_pretrain_vlm.py → train_sft_vlm.py` 两阶段训练的区别。
