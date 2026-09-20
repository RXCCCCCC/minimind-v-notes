[TOC]

# 第四轮

下一轮我建议直接进入 **第四轮：MiniMind 大模型训练工程**。第三轮你已经解决了“**模型 forward 到底怎么算**”，第四轮解决：

> **这些 forward 结果究竟怎么变成梯度、怎么更新模型、怎么稳定训练、怎么保存和恢复。**

这轮对你尤其重要，因为它正好补你目前“深度学习概念覆盖不错，但独立 PyTorch 训练能力偏弱”的缺口。

# 第四轮的核心主线

我们会拿当前真实的 `trainer/train_pretrain.py` 从头拆，但不会按源码从第 1 行机械解释，而是围绕一次真正的 **optimizer update** 来看。

MiniMind 当前一次训练的核心实际上就是：

```text
DataLoader 给出一个 batch
        ↓
input_ids / labels → GPU
        ↓
设置当前 learning rate
        ↓
autocast(BF16/FP16)
        ↓
model(input_ids, labels)
        ↓
loss
        ↓
loss / accumulation_steps
        ↓
backward()
        ↓
积累若干个 batch 的梯度
        ↓
unscale
        ↓
gradient clipping
        ↓
optimizer.step()
        ↓
zero_grad()
```

这正是当前源码实际执行的流程。

---

# 我会把第四轮拆成 7 个部分

### 1. 彻底搞懂 `step` 到底是什么

这是第一件事。

你会发现大模型训练里至少有三种“步”容易混：

```text
一个 batch
一个 backward
一个 optimizer.step()
```

MiniMind 默认：

```text
batch_size = 32
accumulation_steps = 8
```

意味着**单卡情况下：**

```text
batch 1 → backward
batch 2 → backward
...
batch 8 → backward
         ↓
optimizer.step()
```

**所以：**

```text
1 次 optimizer update
=
8 个 micro-batch
=
32 × 8
=
256 条训练样本（理论满 batch 情况）
```

**如果 `max_seq_len=340`，那么一次参数更新最多涉及约：**

$$
32\times 8\times 340=87040
$$

**个 token 位置。**

**多卡时再乘 GPU 数。**

这会把你之前学的 Gradient Accumulation 真正对应到训练脚本里。当前默认参数确实是 `batch_size=32`、`accumulation_steps=8`、`max_seq_len=340`。

---

### 2. AdamW + Learning Rate Schedule

我们会真正回答：

```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=args.learning_rate
)
```

里的：

```text
model.parameters()
到底是什么？

Adam 为什么需要 m、v？

Adam 和 AdamW 为什么不是完全一回事？

weight decay 到底在做什么？

learning_rate=5e-4 到底控制什么？
```

然后看 MiniMind：

```python
lr = get_lr(...)
```

为什么**每一个 step 都重新调整学习率**。

这里会把：

```text
warmup
cosine decay
learning rate
optimizer
gradient
```

**真正连起来。**

你不需要死背 Adam 推导，但要**达到：**

> **给我一张 loss/LR 曲线，我知道训练可能发生了什么。**

---

### 3. BF16 / FP16 / FP32 + AMP

第三部分会**把我们前面只是“知道含义”的混合精度真正讲透。**

当前代码：

```python
autocast_ctx = ...
```

然后：

```python
with autocast_ctx:
    res = model(...)
```

同时：

```python
GradScaler(
    enabled=(args.dtype == 'float16')
)
```

也就是说当前**默认 `bfloat16` 时，GradScaler 实际不会启用；使用 `float16` 时才启用。**

我们会解决：

```text
为什么模型参数不是所有计算都必须 FP32？

autocast 到底自动做了什么？

为什么 FP16 更需要 GradScaler？

为什么 BF16 通常不需要？

loss scaling 到底为什么能救 FP16 下的梯度？
```

这个东西以后你跑任何模型训练都绕不开。

---

### 4. Gradient Accumulation + Gradient Clipping

这里我们不只是讲定义。

直接追：

```python
loss = loss / args.accumulation_steps
```

为什么**必须除 8？**

然后：

```python
scaler.scale(loss).backward()
```

为什么这里**连续调用 8 次，梯度不会被覆盖？**

再看：

```python
scaler.unscale_(optimizer)
```

为什么**一定要先 unscale，才能：**

```python
clip_grad_norm_
```

最后：

```python
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad()
```

你会把这一整段真正看成**一个完整系统，而不是五个陌生 API。**

---

# 5. Checkpoint / Resume

这个部分对你以后做科研项目特别重要。

MiniMind 当前保存的不只是：

```text
model weights
```

还会**处理：**

```text
model
optimizer
GradScaler
epoch
step
```

**等训练状态**，从而支持 `--from_resume 1`。

我们会专门区分：

```text
仅加载模型权重继续训练
```

和：

```text
恢复完整训练现场
```

完全不是同一件事。

例如如果：

```text
Adam 的 momentum 状态丢了
learning-rate 所在 step 丢了
GradScaler 状态丢了
```

虽然**模型参数还在，但严格意义上已经不是原来的训练轨迹**。

这跟以后科研里的：

> reproducibility

直接相关。

---

# 6. 单 GPU → DDP

这一部分放后面讲。

因为你现在最忌讳的是：

> 单 GPU Training Loop 还没完全吃透，就被 DDP 的 rank/world_size/sampler/process group 搞乱。

MiniMind 当前已经包含：

```text
init_distributed_mode
DistributedSampler
DistributedDataParallel
local_rank
dist.get_rank()
```

等**分布式训练逻辑。**

我们的学习顺序是：

```text
单 GPU Training Loop
        ↓
先完全理解
        ↓
假设有 2 张 GPU
        ↓
数据怎么拆？
模型怎么复制？
梯度怎么同步？
        ↓
再理解 DDP
```

最后你应该理解：

```text
GPU0：模型副本 + batch A
GPU1：模型副本 + batch B

各自 forward/backward
        ↓
梯度同步
        ↓
两边做相同 optimizer update
```

而不是**死记 DDP API。**

---

# 7. `torch.compile`、WandB/SwanLab 暂时降级

源码里还有：

```text
torch.compile
wandb / swanlab
日志
ETA
```

这些会讲，但只讲够用。

优先级明显低于：

```text
Loss
Backward
Optimizer
LR
AMP
Gradient
Checkpoint
DDP
```

否则又容易被工程外围细节带跑偏。

---

# 第四轮结束后，我要求你达到什么程度

不是“看过 `train_pretrain.py`”。

而是让你能够拿一张白纸自己写出这种骨架：

```python
model = ...
optimizer = ...

for epoch in range(epochs):
    for x, y in loader:

        optimizer.zero_grad()

        output = model(x)
        loss = ...

        loss.backward()

        torch.nn.utils.clip_grad_norm_(...)

        optimizer.step()
```

然后你能**自己解释怎么一步步升级成**：

```text
基础 Train Loop
↓
AMP
↓
Gradient Accumulation
↓
LR Schedule
↓
Checkpoint
↓
Resume
↓
DDP
```

到了这里，你的 PyTorch 就不再只是：

> “AI 给我代码，我大概能看懂。”

而开始进入：

> **“我知道完整训练程序为什么应该这样组织。”**

这才是 MiniMind 对你最大的价值。

---

## 而且第四轮结束后，我们不会继续无限补理论

我建议后面直接开始**从“学习”切到“项目”。**

路线改成：

```text
第一～第四轮
MiniMind 必需前置
        ↓
第五轮
真正跑通 MiniMind baseline
        ↓
保存 loss 曲线 / checkpoint / inference 样例
        ↓
第六轮
Pretrain → SFT → LoRA / 对齐链路
        ↓
第七轮
选一个适合你的模型修改点
        ↓
Baseline
→ Hypothesis
→ Modification
→ Controlled Experiment
→ Ablation
        ↓
第八轮
整理 GitHub + 实验表 + README + 技术报告
        ↓
形成简历项目
        ↓
拿去套导师
```

所以你不会陷入“前置知识永远学不完”。

**下一条我们就可以正式开始第四轮第一部分。**我会直接拿当前 `train_pretrain.py`，先把**一次 optimizer update 从 DataLoader 到参数改变完整追一遍**，然后再逐个深入 AdamW、LR、AMP、梯度累积等。当前源码默认训练参数和完整主循环都已经核过，可以直接按真实实现学。

你说得对。前一版更像“根据源码讲训练知识”，而你真正需要的是：

> **打开 MiniMind 仓库 → 我告诉你现在看哪个文件、哪个类/函数、哪几行代码 → 贴出这段真实代码 → 逐行带你看。**

从这一轮开始都按这个方式来。我们先重做第四轮，而且**一部分一部分学，不一次把整个 `train_pretrain.py` 灌完。**

---

# 第四轮 · 第 1 部分：一条训练数据到底怎样走到 `backward()`

今天只追这一条：

```text
JSONL 一条文本
↓
PretrainDataset.__getitem__()
↓
input_ids / labels
↓
DataLoader 拼成 batch
↓
train_epoch()
↓
model(input_ids, labels)
↓
CrossEntropy Loss
↓
loss.backward()
```

先不要管 AdamW、AMP、DDP、checkpoint。

---

# 0. 先打开这三个源码位置

## 位置 A：数据集

仓库：

```text
dataset/
└── lm_dataset.py
```

找到：

```python
class PretrainDataset(Dataset):
```

当前 `master` 就是这个实现。

---

## 位置 B：预训练主程序

仓库：

```text
trainer/
└── train_pretrain.py
```

我们今天主要看：

```python
# ========== 5. 定义模型、数据、优化器 ==========
```

然后：

```python
# ========== 8. 开始训练 ==========
```

最后回到文件上方：

```python
def train_epoch(...):
```

这是当前 `master` 的真实训练脚本。

---

## 位置 C：模型怎样算 loss

仓库：

```text
model/
└── model_minimind.py
```

找到：

```python
class MiniMindForCausalLM
```

里面的：

```python
def forward(...):
```

我们第三轮已经读过模型主体，这次只看它最后怎样从 logits 得到 loss。

---

# 1. 从程序入口开始，而不是先看 `train_epoch`

打开：

```text
trainer/train_pretrain.py
```

先找到参数：

```python
parser.add_argument("--batch_size", type=int, default=32, help="batch size")

parser.add_argument(
    '--max_seq_len',
    default=340,
    type=int,
    help="训练的最大截断长度（中文1token≈1.5~1.7字符）"
)

parser.add_argument(
    '--accumulation_steps',
    default=8,
    type=int,
    help="梯度累积步数"
)
```

当前默认我们先记三个数字：

```text
batch_size = 32
max_seq_len = 340
accumulation_steps = 8
```

今天**先用前两个：**

```text
一条样本 → 最长 340 token

一个 batch → 最多 32 条样本
```

**所以后面你要始终脑补：**

```text
input_ids.shape ≈ [32, 340]
labels.shape    ≈ [32, 340]
```

**最后一个不足 32 的 batch 除外。**

---

# 2. 模型和 Dataset 在哪里创建？

仍在：

```text
trainer/train_pretrain.py
```

找到源码注释：

```python
# ========== 5. 定义模型、数据、优化器 ==========
```

当前代码：

```python
model, tokenizer = init_model(
    lm_config,
    args.from_weight,
    device=args.device
)

train_ds = PretrainDataset(
    args.data_path,
    tokenizer,
    max_length=args.max_seq_len
)

train_sampler = (
    DistributedSampler(train_ds)
    if dist.is_initialized()
    else None
)

scaler = torch.cuda.amp.GradScaler(
    enabled=(args.dtype == 'float16')
)

optimizer = optim.AdamW(
    model.parameters(),
    lr=args.learning_rate
)
```

今天只看前两句：

```python
model, tokenizer = init_model(...)
```

和：

```python
train_ds = PretrainDataset(...)
```

也就是说：

```text
tokenizer
        ↓
PretrainDataset

model
        ↓
后面吃 Dataset 生成的 token ids
```

先形成这个关系。

---

# 3. 现在跳到 `PretrainDataset`

打开：

```text
dataset/lm_dataset.py
```

找到：

```python
class PretrainDataset(Dataset):
```

完整核心代码只有这么一点：

```python
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset(
            'json',
            data_files=data_path,
            split='train'
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        tokens = self.tokenizer(
            str(sample['text']),
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True
        ).input_ids

        tokens = (
            [self.tokenizer.bos_token_id]
            + tokens
            + [self.tokenizer.eos_token_id]
        )

        input_ids = (
            tokens
            + [self.tokenizer.pad_token_id]
            * (self.max_length - len(tokens))
        )

        input_ids = torch.tensor(
            input_ids,
            dtype=torch.long
        )

        labels = input_ids.clone()

        labels[
            input_ids == self.tokenizer.pad_token_id
        ] = -100

        return input_ids, labels
```

这就是当前仓库真实实现。

我们一行一行来。

---

# 4. `self.samples` 是什么？

源码：

```python
self.samples = load_dataset(
    'json',
    data_files=data_path,
    split='train'
)
```

而 `train_pretrain.py` **默认传进来的路径是**：

```python
parser.add_argument(
    "--data_path",
    type=str,
    default="../dataset/pretrain_t2t_mini.jsonl"
)
```

所以：

```text
pretrain_t2t_mini.jsonl
        ↓
HuggingFace datasets.load_dataset
        ↓
self.samples
```

当：

```python
sample = self.samples[index]
```

时，就**拿出一条预训练文本数据。**

**后面代码明确访问：**

```python
sample['text']
```

**所以对 `PretrainDataset` 来说，它关心的核心字段就是：**

```text
text
```

---

# 5. 第一处真正重要的代码：Tokenizer

源码：

```python
tokens = self.tokenizer(
    str(sample['text']),
    add_special_tokens=False,
    max_length=self.max_length - 2,
    truncation=True
).input_ids
```

这里我们把它拆开。

假设**原始：**

```text
sample['text']

=
"人工智能正在改变我们的生活"
```

**Tokenizer 可能变成：**

```text
tokens =

[174, 2851, 92, 617,  ...]
```

**注意这里只是示意，具体 token ID 必须由 MiniMind tokenizer 决定。**

---

## 为什么 `add_special_tokens=False`？

**因为下一行作者自己添加：**

```python
tokens = (
    [self.tokenizer.bos_token_id]
    + tokens
    + [self.tokenizer.eos_token_id]
)
```

**所以不希望 tokenizer 再自动加一遍。**

否则**可能出现重复：**

```text
BOS BOS ... EOS EOS
```

---

# 6. 为什么是 `max_length - 2`？

参数传入：

```text
max_length = 340
```

但源码：

```python
max_length=self.max_length - 2
```

所以文本本身最多：

```text
338 tokens
```

为什么留两个？

就是下一行的：

```text
BOS
+
正文
+
EOS
```

占掉两个位置。

因此：

```text
338 + BOS + EOS = 340
```

这不是通用理论上的随便写法，而是 **MiniMind 这份 Dataset 当前明确这样实现的**。

---

# 7. BOS / EOS 加完以后是什么？

源码：

```python
tokens = (
    [self.tokenizer.bos_token_id]
    + tokens
    + [self.tokenizer.eos_token_id]
)
```

例如原来：

```text
人工智能正在发展
```

Tokenize：

```text
[31, 72, 185, 90]
```

加完：

```text
[BOS, 31, 72, 185, 90, EOS]
```

其中：

```text
BOS = Begin Of Sequence
EOS = End Of Sequence
```

模型**因此知道：**

```text
这里开始
...
这里结束
```

---

# 8. 接下来 Padding

源码：

```python
input_ids = (
    tokens
    + [self.tokenizer.pad_token_id]
    * (self.max_length - len(tokens))
)
```

假设：

```text
max_length = 10
```

当前只有：

```text
[BOS, A, B, C, EOS]
```

长度 5。

**那么：**

```text
10 - 5 = 5
```

**于是补：**

```text
[BOS, A, B, C, EOS, PAD, PAD, PAD, PAD, PAD]
```

**最终每一条：**

```text
长度固定 = 340
```

---

# 9. 为什么一定要变成一样长？

因为**后面 DataLoader 要把：**

```text
sample 1
sample 2
...
sample 32
```

**叠成一个 Tensor。**

如果：

```text
sample1 = 167 token
sample2 = 291 token
sample3 = 83 token
```

没法直接形成规则矩阵。

MiniMind 当前 Dataset 直接把每条都 pad 成：

```text
340
```

于是**才能：**

```text
32 × 340
```

**组成 batch。**

---

# 10. 然后转 Tensor

源码：

```python
input_ids = torch.tensor(
    input_ids,
    dtype=torch.long
)
```

所以**单条样本：**

```text
input_ids.shape = [340]
```

**为什么是：**

```python
torch.long
```

**？**

**因为这些东西是：**

```text
token ID
```

**比如：**

```text
27
193
584
```

**是整数索引。**

还记得第三轮：

```python
nn.Embedding(vocab_size, hidden_size)
```

吗？

Embedding 吃的就是这种整数 token index。

---

# 11. `labels = input_ids.clone()` 是整个预训练目标的关键

源码：

```python
labels = input_ids.clone()
```

这句话初看特别奇怪：

> 输入和答案怎么完全一样？

假设：

```text
input_ids:

[BOS, 我, 喜欢, AI, EOS]
```

labels 也先复制成：

```text
[BOS, 我, 喜欢, AI, EOS]
```

但真正的错位并不是 Dataset 这里做。

**错位发生在模型的 loss 代码里。**

这个等会马上跳过去看。

---

# 12. 为什么 PAD 改成 `-100`

源码：

```python
labels[
    input_ids == self.tokenizer.pad_token_id
] = -100
```

所以：

```text
input_ids:

[BOS, 我, 喜欢, AI, EOS, PAD, PAD]
```

**labels：**

```text
[BOS, 我, 喜欢, AI, EOS, -100, -100]
```

这**不是随便定义一个神奇标签。**

等会模型中：

```python
F.cross_entropy(
    ...,
    ignore_index=-100
)
```

会**明确告诉 PyTorch：**

> **label 是 -100 的位置，不计算 loss。**

**MiniMind Dataset 与 MiniMind Model 正好对应。**

---

# 13. 所以 `__getitem__()` 最终返回什么？

最后：

```python
return input_ids, labels
```

也就是说：

```text
train_ds[i]
```

返回：

```text
input_ids.shape = [340]
labels.shape    = [340]
```

---

# 14. 现在回到 `train_pretrain.py`

找到：

```python
# ========== 8. 开始训练 ==========
```

当前源码：

```python
for epoch in range(start_epoch, args.epochs):
    train_sampler and train_sampler.set_epoch(epoch)

    setup_seed(args.seed + epoch)
    indices = torch.randperm(len(train_ds)).tolist()

    skip = (
        start_step
        if (epoch == start_epoch and start_step > 0)
        else 0
    )

    batch_sampler = SkipBatchSampler(
        train_sampler or indices,
        args.batch_size,
        skip
    )

    loader = DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    if skip > 0:
        ...
    else:
        train_epoch(
            epoch,
            loader,
            len(loader),
            0,
            wandb
        )
```

这里第一次**出现真正的数据批处理。**

---

# 15. 这一行特别值得看

```python
indices = torch.randperm(
    len(train_ds)
).tolist()
```

假设 **Dataset 只有 10 条：**

```text
0 1 2 3 4 5 6 7 8 9
```

**`randperm()` 可能得到：**

```text
7 2 5 0 9 3 1 8 6 4
```

也就是：

> **每个 epoch 打乱训练样本顺序。**

然后**交给：**

```python
SkipBatchSampler(...)
```

---

# 16. `SkipBatchSampler` 又在哪里？

不是 PyTorch 自带。

仓库：

```text
trainer/trainer_utils.py
```

找到：

```python
class SkipBatchSampler(Sampler):
```

源码**核心：**

```python
class SkipBatchSampler(Sampler):
    def __init__(
        self,
        sampler,
        batch_size,
        skip_batches=0
    ):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0

        for idx in self.sampler:
            batch.append(idx)

            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue

                yield batch
                batch = []

        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch
```

现在**先忽略 resume 的：**

```python
skip_batches
```

**只看正常情况 `skip=0`。**

**如果：**

```text
batch_size = 32
```

**它就不断收集：**

```text
32 个 index
```

**然后：**

```python
yield batch
```

**例如：**

```text
[581, 23, 991, ..., 76]
```

**总共 32 个索引。**

---

# 17. 注意 MiniMind 这里不是普通的

你可能以后常看到：

```python
DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True
)
```

但 MiniMind 当前不是这么写。

它是：

```python
DataLoader(
    train_ds,
    batch_sampler=batch_sampler,
    ...
)
```

也就是说：

> **哪些数据组成一个 batch，作者自己通过 `SkipBatchSampler` 控制。**

这是为了**后面的：**

```text
resume 后跳过已经训练的 batch
```

**服务的。**

这一点**必须按仓库源码理解，而不能拿最普通的 PyTorch 教程写法替代**。

---

# 18. DataLoader 最终做了什么？

`batch_sampler` 给：

```text
32 个 index
```

比如：

```text
[3, 8, 15, ..., 912]
```

**DataLoader 就逐个调用：**

```python
train_ds[3]
train_ds[8]
train_ds[15]
...
```

**而刚刚我们知道：**

```python
train_ds[i]
```

**返回：**

```text
input_ids [340]
labels    [340]
```

**32 条 stack 起来：**

```text
input_ids
[32,340]

labels
[32,340]
```

**这就是最终进入模型的 batch。**

---

# 19. 现在终于进入 `train_epoch()`

还是：

```text
trainer/train_pretrain.py
```

直接跳到文件上面：

```python
def train_epoch(
    epoch,
    loader,
    iters,
    start_step=0,
    wandb=None
):
```

核心循环：

```python
for step, (input_ids, labels) in enumerate(
    loader,
    start=start_step + 1
):
```

现在你应该能够真正理解：

```python
input_ids
labels
```

到底从哪来的。

不是凭空出现。

完整来源：

```text
pretrain_t2t_mini.jsonl

↓ load_dataset

sample['text']

↓ tokenizer

tokens

↓ BOS/EOS/PAD

input_ids [340]

↓ clone + PAD→-100

labels [340]

↓ SkipBatchSampler + DataLoader

input_ids [32,340]
labels    [32,340]

↓ train_epoch
```

这个链你必须建立起来。

---

# 20. 第一行：把数据送到 GPU

源码：

```python
input_ids = input_ids.to(args.device)
labels = labels.to(args.device)
```

假设：

```text
args.device = cuda:0
```

那么：

```text
CPU RAM
input_ids [32,340]

↓

GPU VRAM
input_ids [32,340]
```

**shape 没变。**

**只是设备改变。**

---

# 21. 接下来这一块我们暂时跳过

源码马上是：

```python
lr = get_lr(
    epoch * iters + step,
    args.epochs * iters,
    args.learning_rate
)

for param_group in optimizer.param_groups:
    param_group['lr'] = lr
```

这就是 Learning Rate Schedule。

**先不要展开。**

它是**第四轮第 2 部分的内容。**

今天继续追数据。

---

# 22. 关键位置来了：真正调用模型

源码：

```python
with autocast_ctx:
    res = model(
        input_ids,
        labels=labels
    )

    loss = res.loss + res.aux_loss

    loss = loss / args.accumulation_steps
```

**暂时把：**

```python
with autocast_ctx:
```

**当成：**

> **混合精度环境。**

**AMP 下一部分再深入。**

**现在只看：**

```python
res = model(
    input_ids,
    labels=labels
)
```

**这里我们必须真的跳到：**

```text
model/model_minimind.py
```

**不能凭印象讲。**

---

# 23. 跳到 `MiniMindForCausalLM.forward()`

当前位置：

```text
model/model_minimind.py
→ class MiniMindForCausalLM
→ def forward(...)
```

当前代码：

```python
def forward(
    self,
    input_ids,
    attention_mask=None,
    past_key_values=None,
    use_cache=False,
    logits_to_keep=0,
    labels=None,
    **kwargs
):
    hidden_states, past_key_values, aux_loss = self.model(
        input_ids,
        attention_mask,
        past_key_values,
        use_cache,
        **kwargs
    )

    slice_indices = (
        slice(-logits_to_keep, None)
        if isinstance(logits_to_keep, int)
        else logits_to_keep
    )

    logits = self.lm_head(
        hidden_states[:, slice_indices, :]
    )

    loss = None

    if labels is not None:
        x, y = (
            logits[..., :-1, :].contiguous(),
            labels[..., 1:].contiguous()
        )

        loss = F.cross_entropy(
            x.view(-1, x.size(-1)),
            y.view(-1),
            ignore_index=-100
        )

    return MoeCausalLMOutputWithPast(
        loss=loss,
        aux_loss=aux_loss,
        logits=logits,
        ...
    )
```

这是**当前 `master` 实现。**

**现在你前面学的东西全部串起来了。**

---

# 24. 输入 shape 再走一次

训练脚本送进来：

```text
input_ids
[32,340]
```

经过：

```python
self.model(input_ids, ...)
```

也就是第三轮读过的：

```text
Embedding
↓
8 × Transformer Block
↓
Final RMSNorm
```

**得到：**

```text
hidden_states
[32,340,768]
```

**然后：**

```python
logits = self.lm_head(...)
```

**LM Head：**

```text
768 → 6400
```

**所以：**

```text
logits
[32,340,6400]
```

---

# 25. 然后终于出现 Next Token Prediction 的错位

**源码：**

```python
x = logits[..., :-1, :]

y = labels[..., 1:]
```

**假设：**

```text
input_ids / labels:

[BOS, 我, 喜欢, AI, EOS]
```

模型各位置输出：

```text
位置 0 logits
位置 1 logits
位置 2 logits
位置 3 logits
位置 4 logits
```

**但是位置 0 应该预测：**

```text
我
```

**位置 1 应该预测：**

```text
喜欢
```

**所以：**

```text
x:

logits[0]
logits[1]
logits[2]
logits[3]
```

**对应：**

```text
y:

我
喜欢
AI
EOS
```

**因此：**

```python
logits[..., :-1, :]
labels[..., 1:]
```

正好**对齐。**

---

# 26. Shape 也要跟上

**原：**

```text
logits
[B,T,V]

=
[32,340,6400]
```

**执行：**

```python
logits[..., :-1, :]
```

**得到：**

```text
x
[32,339,6400]
```

**labels：**

```text
[32,340]
```

**执行：**

```python
labels[..., 1:]
```

**得到：**

```text
y
[32,339]
```

**于是现在：**

```text
每一个 x[b,t,:]
```

**是一组：**

```text
6400 个 token logits
```

**对应：**

```text
y[b,t]
```

**这个位置唯一正确的：**

```text
token id
```

---

# 27. 为什么还要 `view(-1, ...)`？

**源码：**

```python
x.view(-1, x.size(-1))
```

**原来：**

```text
x
[32,339,6400]
```

**变成：**

```text
[32×339, 6400]
```

**即：**

```text
[10848,6400]
```

**而：**

```python
y.view(-1)
```

**：**

```text
[32,339]

↓

[10848]
```

**所以 CrossEntropy 看到的就是：**

```text
10848 道 6400 分类问题
```

**每一道问题：**

> **在 6400 个 token 中，正确的下一个 token 是哪个？**

**这就是语言模型预训练。**

---

# 28. `ignore_index=-100` 终于接上 Dataset

还记得 Dataset：

```python
labels[
    input_ids == pad_token_id
] = -100
```

模型这里：

```python
F.cross_entropy(
    ...,
    ignore_index=-100
)
```

正好对应。

于是 padding：

```text
PAD PAD PAD
```

不会贡献训练 loss。

这个设计是：

```text
Dataset 写 -100
+
Model CrossEntropy 忽略 -100
```

两边共同完成的。

---

# 29. `res.loss` 到底是什么，现在就清楚了

回到：

```text
trainer/train_pretrain.py
```

刚才：

```python
res = model(
    input_ids,
    labels=labels
)
```

**返回的是：**

```python
MoeCausalLMOutputWithPast(...)
```

**其中：**

```text
res.loss
```

**就是刚刚的：**

```python
F.cross_entropy(...)
```

**也就是：**

> **当前这个 batch 的 next-token prediction loss。**

---

# 30. 那 `res.aux_loss` 是什么？

训练脚本：

```python
loss = res.loss + res.aux_loss
```

模型代码中：

```python
aux_loss = sum([
    l.mlp.aux_loss
    for l in self.layers
    if isinstance(l.mlp, MOEFeedForward)
], ...)
```

也就是**说它是：**

> **MoE 模式下的额外辅助损失。**

**但 MiniMind 当前训练参数默认：**

```python
--use_moe 0
```

**所以我们当前学习普通 64M Dense MiniMind 时，可以先把主线理解为：**

```text
loss ≈ CrossEntropy next-token loss
```

**MoE 后面单独补。**

---

# 31. 接下来这一句非常重要

训练脚本：

```python
loss = loss / args.accumulation_steps
```

默认：

```text
accumulation_steps = 8
```

于是：

```text
真实 batch loss
÷
8
```

为什么**这么干，涉及：**

# **Gradient Accumulation**

我们下一部分会拿下面完整源码一起讲：

```python
loss = loss / args.accumulation_steps

scaler.scale(loss).backward()

if step % args.accumulation_steps == 0:
    scaler.unscale_(optimizer)

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        args.grad_clip
    )

    scaler.step(optimizer)
    scaler.update()

    optimizer.zero_grad(set_to_none=True)
```

今天暂时只知道：

> 作者准备连续累计 8 个 batch 的 gradient，所以先把每个 batch 的 loss 除以 8。

---

# 32. 最后到今天的终点：`backward()`

源码下一行：

```python
scaler.scale(loss).backward()
```

暂时先**忽略：**

```python
scaler.scale(...)
```

**那这句话核心就是：**

```python
loss.backward()
```

**即：**

```text
loss
↑
LM Head
↑
8 × Transformer
↑
Embedding
```

   

真正**修改参数的是后面的：**

```python
optimizer.step()
```

**我们下一部分再进入它。**

---

# 33. 到这里你应该能直接在源码里画出这张图

现在不要离开 MiniMind 源码，只根据刚刚三个文件：

```text
dataset/pretrain_t2t_mini.jsonl
             │
             ▼
dataset/lm_dataset.py
PretrainDataset.__getitem__()
             │
             ├─ tokenizer
             ├─ BOS / EOS
             ├─ PAD
             └─ labels PAD → -100
             │
             ▼
单条：
input_ids [340]
labels    [340]
             │
             ▼
trainer/train_pretrain.py
SkipBatchSampler + DataLoader
             │
             ▼
一个 batch：
input_ids [32,340]
labels    [32,340]
             │
             ▼
train_epoch()
             │
             ▼
model(input_ids, labels)
             │
             ▼
model/model_minimind.py
MiniMindForCausalLM.forward()
             │
             ├─ MiniMindModel
             │      ↓
             │  [32,340,768]
             │
             ├─ lm_head
             │      ↓
             │  [32,340,6400]
             │
             ├─ logits[..., :-1, :]
             │  [32,339,6400]
             │
             ├─ labels[..., 1:]
             │  [32,339]
             │
             └─ CrossEntropy
                    ↓
                   loss
                    │
                    ▼
trainer/train_pretrain.py
loss / 8
                    │
                    ▼
backward()
                    │
                    ▼
Parameter.grad
```

**这张图就是今天这一小节。**

---

# 34. 你现在回仓库时具体怎么操作

不要把三个文件通篇读一遍。

就按这个顺序定位：

1. `dataset/lm_dataset.py`
   → 搜 `class PretrainDataset`
   → 只读 `__init__ / __getitem__`

2. `trainer/train_pretrain.py`
   → 搜 `# ========== 5.`
   → 找 `train_ds = PretrainDataset(...)`
   → 再搜 `# ========== 8.`
   → 找 `SkipBatchSampler + DataLoader`
   → 再搜 `def train_epoch`

3. `model/model_minimind.py`
   → 搜 `class MiniMindForCausalLM`
   → 搜它下面的 `def forward`
   → 只看 `logits → x/y → cross_entropy`

现在其余代码都可以先折叠。

---

# 35. 这一部分的验收，不是背概念

你现在最好能直接对着代码回答：

**① 为什么 `PretrainDataset` tokenizer 的 `max_length` 是 `self.max_length - 2`？**

因为源码随后手动增加：

```text
BOS + EOS
```

---

**② 为什么单条数据是 `[340]`，到了 `train_epoch()` 变成 `[32,340]`？**

因为：

```text
SkipBatchSampler 选择 32 个索引
↓
DataLoader stack 32 条 Dataset 返回值
```

---

**③ `labels = input_ids.clone()` 为什么不是模型复制答案作弊？**

因为真正训练时：

```python
logits[..., :-1, :]
```

对应：

```python
labels[..., 1:]
```

模型当前位置预测的是**下一个 token**。

---

**④ `-100` 是谁写进去、谁读取？**

写：

```text
dataset/lm_dataset.py
PretrainDataset.__getitem__
```

读：

```text
model/model_minimind.py
F.cross_entropy(ignore_index=-100)
```

---

**⑤ `backward()` 后模型参数有没有改变？**

没有。

只是：

```text
parameter.grad
```

有了梯度。

---

## 下一小节就接着当前位置往下读

下一部分不换主线，直接继续：

```text
trainer/train_pretrain.py
→ train_epoch()
```

从这一句：

```python
loss = loss / args.accumulation_steps
```

开始，逐行读到：

```python
scaler.scale(loss).backward()

if step % args.accumulation_steps == 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(...)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

也就是专门把 **梯度累积 → AMP/GradScaler → 梯度裁剪 → AdamW 参数更新 → zero_grad** 按当前 MiniMind 源码彻底讲透。

以后整个 MiniMind 学习我都按今天这种格式：**先报文件路径和函数位置，再贴仓库真实代码，再顺着代码讲，不拿抽象教程代替仓库。**



继续，而且这次严格沿着 **MiniMind 当前 `master` 源码**往下走。

这一小节只看一个位置：

```text
jingyaogong/minimind
└── trainer/
    └── train_pretrain.py
        └── train_epoch()
```

我们从上一节停下来的：

```python
loss = loss / args.accumulation_steps
```

一直读到：

```python
optimizer.zero_grad(set_to_none=True)
```

当前 `master` 的**原始核心代码就是：**

```python
with autocast_ctx:
    res = model(input_ids, labels=labels)
    loss = res.loss + res.aux_loss
    loss = loss / args.accumulation_steps

scaler.scale(loss).backward()

if step % args.accumulation_steps == 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

    scaler.step(optimizer)
    scaler.update()

    optimizer.zero_grad(set_to_none=True)
```

默认参数则是：

```python
--batch_size 32
--dtype bfloat16
--accumulation_steps 8
--grad_clip 1.0
--learning_rate 5e-4
```

这些都是当前仓库真实代码。

---

# 第四轮 · 第 2 部分：一次 MiniMind 参数更新到底怎么发生

先把今天这一段压缩成：

```text
一个 batch
↓
forward
↓
loss / 8
↓
backward
↓
只积梯度，不改参数
↓
连续重复 8 个 batch
↓
unscale
↓
gradient clipping
↓
AdamW.step()
↓
参数真正改变
↓
zero_grad
↓
开始下一组 8 batch
```

今天最重要的是区分两个概念：

> **training step 和 optimizer update 在这份代码里不是一回事。**

---

# 1. 先找 `accumulation_steps` 在哪里定义

位置还是：

```text
trainer/train_pretrain.py
→ if __name__ == "__main__":
→ argparse 参数
```

源码：

```python
parser.add_argument(
    '--accumulation_steps',
    default=8,
    type=int,
    help="梯度累积步数"
)
```

所以当前默认：

```text
accumulation_steps = 8
```

上一节我们知道：

```text
batch_size = 32
```

所以 MiniMind 并不是：

```text
32 条数据
→ backward
→ 马上更新一次参数
```

而是：

```text
32 条 → backward
32 条 → backward
32 条 → backward
32 条 → backward
32 条 → backward
32 条 → backward
32 条 → backward
32 条 → backward
             ↓
          update
```

也就是 8 个 **micro-batch 合起来更新一次。**

**单 GPU 下相当于：**
$$
32\times8=256
$$

所以**可以粗略理解：**

```text
一个 optimizer update
≈ 看完 256 条训练样本
```

**如果每条最多 340 个 token 槽位：**

$$
256\times340=87040
$$

**所以一次参数更新最多经过约：**

```text
87040 个 token position
```

**当然这里包含 padding，真正参与 CrossEntropy 的有效 token 会更少，因为 PAD label 是 `-100`。**

---

# 2. 为什么不直接把 `batch_size` 设置成 256？

这是梯度累积存在的核心理由。

如果：

```python
batch_size = 256
```

GPU 一次 forward 要同时保存 256 条样本所有 Transformer 中间激活。

显存可能爆掉。

于是改成：

```text
真实 GPU batch = 32

连续算 8 次
↓
梯度先不清除
↓
最后一起更新
```

相当于：

> **用更多计算步骤换显存。**

注意，不是让一次 forward 真的变成 256。

GPU 每次仍只看到：

```text
[32, 340]
```

---

# 3. 第一行关键源码：为什么 `loss / 8`

看原代码：

```python
loss = res.loss + res.aux_loss
loss = loss / args.accumulation_steps
```

也就是默认：

```python
loss = loss / 8
```

为什么？

因为 PyTorch 的：

```python
loss.backward()
```

默认会：

# 累加梯度

而**不是覆盖。**

---

# 4. PyTorch 的 `.grad` 默认是累加的

假设某一个模型参数：

```python
w
```

第一个 batch 算出来：

$$
\nabla_wL_1=4
$$

执行：

```python
loss1.backward()
```

于是：

```text
w.grad = 4
```

第二个 batch 又算：

$$
\nabla_wL_2=6
$$

如果中间**没有**：

```python
optimizer.zero_grad()
```

那么执行第二次：

```python
loss2.backward()
```

之后：

```text
w.grad = 4 + 6
       = 10
```

不是：

```text
w.grad = 6
```

这就是 MiniMind 梯度累积能够成立的基础。

---

# 5. 那为什么必须先 `/ 8`？

假设 8 个 batch 的 loss：

```text
L1
L2
L3
...
L8
```

如果直**接全部 backward：**

$$
\nabla L_1+
\nabla L_2+
\cdots+
\nabla L_8
$$

**得到的是：**

> **8 个 batch 梯度的和。**

**但一般希望模拟一个大 batch 的平均 loss：**

$$
L=
\frac{
L_1+L_2+\cdots+L_8
}{8}
$$

**那么：**

$$
\nabla L
=
\frac1{8}
\left(
\nabla L_1+
\nabla L_2+
\cdots+
\nabla L_8
\right)
$$

**所以代码先做：**

```python
loss = loss / 8
```

**再连续 backward。**

---

# 6. 用一个具体数字看最清楚

假设某个参数在 8 个 batch 上梯度分别：

```text
8
16
24
32
40
48
56
64
```

如果直接累加：

```text
8+16+24+32+40+48+56+64
= 288
```

但平均梯度应该：

$$
288/8=36
$$

MiniMind 每次先除 8。

所以进入 backward 的梯度相当于：

```text
1
2
3
4
5
6
7
8
```

最后累加：

```text
36
```

**正好就是 8 个 batch 的平均梯度。**

---

# 7. 接下来真正执行 backward

源码：

```python
scaler.scale(loss).backward()
```

先暂时把 `scaler.scale` 遮住：

```python
loss.backward()
```

你上一节已经知道：

```text
loss.backward()

≠ 修改参数
```

它只是沿计算图：

```text
CrossEntropy
↑
LM Head
↑
Transformer Block 8
↑
Embedding
```

计算所有 Parameter 的：

$$
\frac{\partial L}{\partial\theta}
$$

然后放进：

```python
parameter.grad
```

所以：

```text
backward
→ 计算/累加 grad

optimizer.step
→ 根据 grad 真正修改 parameter
```

这两个必须彻底分开。

---

# 8. 现在直接模拟 MiniMind 的 8 个 step

注意源码：

```python
for step, (input_ids, labels) in enumerate(...):
```

这里的：

```text
step
```

是：

> **DataLoader 拿了多少个 batch。**

不是 optimizer 已经更新了多少次。

---

## step = 1

```python
loss /= 8
scaler.scale(loss).backward()
```

然后判断：

```python
if step % 8 == 0:
```

：

```text
1 % 8 != 0
```

所以不进去。

状态：

```text
Parameter:
    weight = 没变

Parameter.grad:
    保存第 1 个 batch / 8 的梯度
```

---

# 9. step = 2

再次：

```python
scaler.scale(loss).backward()
```

因为没有 `zero_grad()`：

```text
.grad =
第1批梯度/8
+
第2批梯度/8
```

参数仍然：

```text
没变
```

---

# 10. 一直到 step = 7

```text
.grad =
(g1 + g2 + ... + g7) / 8
```

但：

```text
Parameter 本身一次都没有更新。
```

---

# 11. step = 8

先正常：

```python
loss /= 8
scaler.scale(loss).backward()
```

所以：

```text
.grad =
(g1 + g2 + ... + g8) / 8
```

然后：

```python
if step % args.accumulation_steps == 0:
```

即：

```text
8 % 8 == 0
```

成立。

**这一次终于进入：**

```python
scaler.unscale_(optimizer)

torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    args.grad_clip
)

scaler.step(optimizer)
scaler.update()

optimizer.zero_grad(set_to_none=True)
```

所以：

> **MiniMind 第 8 个 DataLoader step 才发生第一次 optimizer update。**

---

# 12. 画成时间线

现在看这个图：

```text
micro-step 1
forward
backward
grad = g1/8
parameter 不变

↓


micro-step 2
forward
backward
grad = (g1+g2)/8
parameter 不变

↓

...

micro-step 7
grad = (g1+...+g7)/8
parameter 不变

↓

micro-step 8
grad = (g1+...+g8)/8

↓
clip
↓
AdamW.step()

★★★★★ 参数第一次改变 ★★★★★

↓
zero_grad()
↓
.grad 清掉

↓

micro-step 9
开始下一组
```

所以今后你**看到训练日志里的：**

```text
step=800
```

**不能立刻理解成：**

> **参数已经更新 800 次。**

在当前 `accumulation_steps=8` 下，正常情况下大约是：

$$
800/8=100
$$

次 optimizer update。

---

# 13. 现在正式讲 `scaler`

先不要离开源码。

在：

```text
trainer/train_pretrain.py
→ # ========== 3. 设置混合精度 ==========
```

当前代码：

```python
device_type = (
    "cuda"
    if "cuda" in args.device
    else "cpu"
)

dtype = (
    torch.bfloat16
    if args.dtype == "bfloat16"
    else torch.float16
)

autocast_ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.cuda.amp.autocast(dtype=dtype)
)
```

然后在：

```text
# ========== 5. 定义模型、数据、优化器 ==========
```

又有：

```python
scaler = torch.cuda.amp.GradScaler(
    enabled=(args.dtype == 'float16')
)
```

这一句极其关键。

---

# 14. MiniMind 默认到底是 BF16 还是 FP16？

参数默认：

```python
parser.add_argument(
    "--dtype",
    type=str,
    default="bfloat16"
)
```

所以默认：

```text
dtype = bfloat16
```

那么：

```python
enabled=(args.dtype == 'float16')
```

就是：

```python
enabled=False
```

也就是说：

# MiniMind 默认 BF16 训练时 GradScaler 是禁用的

这一点很重要。

所以**不要看到：**

```python
scaler.scale(loss)
```

**就误以为默认 BF16 时真的做了 loss scaling。**

没有。

---

# 15. 那为什么代码还统一写 `scaler.scale()`？

因为 PyTorch 的 `GradScaler(enabled=False)` 可以**作为一个无操作包装层。**

**PyTorch 官方 AMP 文档明确说明，`enabled=False` 时 GradScaler 相关调用成为 no-op，也就是方便代码不用写两套 if/else。([PyTorch Docs][1])**

所以 MiniMind **默认 BF16 时，可以把：**

```python
scaler.scale(loss).backward()
```

**近似理解成：**

```python
loss.backward()
```

**把：**

```python
scaler.step(optimizer)
```

**近似理解成：**

```python
optimizer.step()
```

**而：**

```python
scaler.unscale_(...)
scaler.update()
```

**也基本没有实际 scaling 工作。**

---

# 16. 那 FP16 时为什么需要 GradScaler？

假设你运行：

```bash
python train_pretrain.py --dtype float16
```

这时：

```python
GradScaler(enabled=True)
```

真正启用。

原因是 **FP16 能表示的数值范围比较有限。**

**反向传播时一些梯度非常小：**

```text
0.00000001
```

**在 FP32 能表达。**

**到了 FP16 可能直接变：**

```text
0
```

**这叫：**

# **underflow**

**如果很多小梯度直接归零：**

> 模型明明应该学习，但那部分 gradient 消失了。

---

# 17. GradScaler 的想法其实很简单

假设真正：

```text
loss = 0.001
```

s**caler 当前 scale：**

```text
65536
```

**先变：**

$$
0.001\times65536
$$

**再：**

```python
scaled_loss.backward()
```

**那么梯度也整体被放大。**

例如**原梯度：**

```text
0.00000001
```

**变成：**

```text
0.00065536
```

**FP16 更容易保存下来。**

**最后真正更新模型以前：**

> **再把梯度除回原来的 scale。**

**这就是：**

```python
scaler.unscale_(optimizer)
```

的意义。

PyTorch 官方说明也是：`scale(loss).backward()` 产生**被 scale 的梯度，若需要在 step 前检查或修改梯度，应先 `unscale_`。([PyTorch Docs][1])**

---

# 18. 为什么 BF16 通常不需要这套东西？

这是 BF16 和 FP16 一个很重要的区别。

简单记：

```text
FP16
指数位少
→ 数值动态范围较小
→ 小梯度更容易 underflow

BF16
指数位与 FP32 更接近
→ 动态范围大很多
→ 通常不需要 loss scaling
```

所以 MiniMind 当前设计：

```text
BF16
→ autocast 开
→ GradScaler 关

FP16
→ autocast 开
→ GradScaler 开
```

直接体现在**源码**：

```python
dtype = torch.bfloat16 if ... else torch.float16

scaler = torch.cuda.amp.GradScaler(
    enabled=(args.dtype == 'float16')
)
```

---

# 19. `autocast` 和 `GradScaler` 别混为一谈

这是非常容易混的两个东西。

MiniMind：

```python
with autocast_ctx:
    res = model(...)
    loss = ...
```

是：

# autocast

负责：

> forward 里哪些操作用 BF16/FP16，哪些保留更高精度，由框架合理选择。

而：

```python
scaler.scale(loss)
```

是：

# Gradient Scaling

解决：

> FP16 backward **小梯度下溢。**

所以：

```text
autocast
→ 主要管 forward / dtype

GradScaler
→ 主要保护 FP16 backward gradient
```

不是一个东西。

---

# 20. 为什么 `backward()` 不写在 autocast 里面？

看 MiniMind 当前缩进：

```python
with autocast_ctx:
    res = model(...)
    loss = ...
    loss = loss / ...

scaler.scale(loss).backward()
```

注意：

```python
backward()
```

已经在：

```text
with autocast_ctx:
```

之外。

这也是 PyTorch **官方 AMP 推荐结构：**

```text
autocast:
    forward
    loss

退出 autocast

backward
```

因为 backward 会**按照 forward 对应操作选择的 dtype 执行，不推荐再单独把 backward 包进 autocast。**([PyTorch Docs][2])

---

# 21. 接下来为什么要 `unscale_`

进入 step=8 后：

```python
scaler.unscale_(optimizer)
```

紧接着：

```python
torch.nn.utils.clip_grad_norm_(...)
```

为什么顺序必须：

```text
unscale
↓
clip
```

而不是：

```text
clip
↓
unscale
```

？

---

# 22. 假设 FP16 scale=1000

真实梯度：

```text
grad = 0.5
```

经过 GradScaler：

```text
scaled_grad = 500
```

MiniMind 设置：

```text
grad_clip = 1.0
```

如果直接：

```python
clip_grad_norm_(..., 1.0)
```

看到的却是：

```text
500
```

它会以为梯度巨大，疯狂裁剪。

但真实梯度其实：

```text
0.5
```

根本不用裁。

所以必须：

```text
scaled grad = 500

↓ scaler.unscale_

real grad = 0.5

↓ clip

判断真实梯度是不是超过 1.0
```

PyTorch 官方 AMP 文档也明确要求：若**在 `scaler.step()` 前执行 gradient clipping，应先 `unscale_()`。([PyTorch Docs][2])**

所以 MiniMind 这里的顺序是正确且有意义的：

```python
scaler.unscale_(optimizer)

torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    args.grad_clip
)
```

---

# 23. 现在正式看 Gradient Clipping

当前参数：

```python
--grad_clip 1.0
```

源码：

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    args.grad_clip
)
```

这里不是：

> 每一个 gradient 大于 1 就改成 1。

那叫 value clipping。

这里是：

# norm clipping

---

# 24. 它看的是什么？

假设整个模型只有 3 个参数，梯度：

```text
g1 = 3
g2 = 4
g3 = 0
```

**整体 L2 norm：**

$$
\sqrt{3^2+4^2}
=5
$$

**但是：**

```text
max_norm = 1
```

于是整体按比例缩：

```text
3 → 0.6
4 → 0.8
```

新的 norm：

$$
\sqrt{0.6^2+0.8^2}=1
$$

也就是说：

> **方向基本不改变，只把整个 gradient vector 的尺度压小。**

---

# 25. 为什么训练 LLM 时要做这个？

偶尔某个 batch 可能导致：

```text
gradient norm
突然巨大
```

例如：

```text
正常：
0.7
1.1
0.9

突然：
82
```

如果**直接 AdamW 更新，可能造成参数巨大波动。**

于是：

```text
grad_clip=1.0
```

像一个**保险丝：**

```text
梯度正常
→ 不动

梯度过大
→ 整体缩回合理范围
```

**这不是让模型每一步梯度都等于 1。**

**只有超过阈值才缩。**

---

# 26. 接下来真正改变模型的代码终于来了

源码：

```python
scaler.step(optimizer)
```

这才是今天最重要的一句。

**默认 BF16、scaler disabled 时，可以近似看成：**

```python
optimizer.step()
```

**而 optimizer 在哪里创建？**

回到：

```text
trainer/train_pretrain.py
→ # ========== 5. 定义模型、数据、优化器 ==========
```

源码：

```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=args.learning_rate
)
```

所以：

# MiniMind 用 AdamW 更新全部 `model.parameters()`

---

# 27. `model.parameters()` 到底包括什么？

结合我们第三轮读过的 `model_minimind.py`，里面包括类似：

```text
Embedding weight

每一层：
    RMSNorm weight

    q_proj.weight
    k_proj.weight
    v_proj.weight
    o_proj.weight

    gate_proj.weight
    up_proj.weight
    down_proj.weight

Final RMSNorm

LM Head
```

只要：

```text
requires_grad=True
```

并且**传进 AdamW，就属于 optimizer 管理的 Parameter。**

所以：

```python
optimizer.step()
```

**不是只更新最后的 LM Head。**

而是：

> 根据各自的 `.grad` 更新整个 MiniMind 的可训练参数。

---

# 28. AdamW 到底比最简单 SGD 多了什么？

你现在不用背完整数学证明。

先从 SGD：

$$
\theta
\leftarrow
\theta-\eta g
$$

其中：

```text
θ = 参数
g = gradient
η = learning rate
```

假设：

```text
gradient = 10
lr = 0.001
```

就沿梯度走：

```text
0.01
```

---

# 29. AdamW 不直接只看当前 gradient

它会**维护每个参数的历史状态。**

最重要两个：

```text
m
≈ gradient 的滑动平均

v
≈ gradient² 的滑动平均
```

可以粗略理解：

```text
m
→ 最近大致应该往哪个方向走

v
→ 最近这个方向梯度波动/尺度有多大
```

因此不同参数：

```text
gradient 大小不同
训练历史不同
```

AdamW 会**自动调节实际更新尺度。**

这就是为什么 Transformer/LLM 训练里 AdamW 非常常见。

---

# 30. 当前 MiniMind 没手动传 `betas` 等参数

源码只写：

```python
optim.AdamW(
    model.parameters(),
    lr=args.learning_rate
)
```

所以：

> 仓库自己**明确指定的只有 `params` 和 `lr`。**

**其余使用安装的 PyTorch 对应 `AdamW` 默认值。**

**当前 PyTorch 主线文档列出的默认值包括：**

```text
betas       = (0.9, 0.999)
eps         = 1e-8
weight_decay = 0.01
```

([PyTorch Docs][3])

注意这里我特意区分：

```text
MiniMind 仓库明确写的
≠
PyTorch API 默认提供的
```

以后读科研代码也要有这个意识。

---

# 31. AdamW 的 W 是什么？

就是：

# Weight Decay

大致思想：

> **不希望模型参数无限膨胀。**

**AdamW 把 weight decay 与 Adam 的梯度矩估计解耦处理**。PyTorch 官方文档明确指出，**AdamW 的 weight decay 不会累积进 momentum 和 variance**。([PyTorch Docs][3])

现在不用展开论文推导。

你只需要知道：

```text
CrossEntropy gradient
→ 让模型拟合训练数据

Weight Decay
→ 同时轻微把参数往较小方向压
```

作为一种正则化。

---

# 32. 但 `scaler.step()` 还有一个 FP16 细节

启用 GradScaler 时：

```python
scaler.step(optimizer)
```

并不是无条件：

```python
optimizer.step()
```

PyTorch 会**先检查梯度是否有：**

```text
inf
NaN
```

**如果存在异常，可能：**

```text
跳过本次 optimizer.step()
```

**避免把坏梯度写进模型。**

官方 AMP 文档就是这样描述的。([PyTorch Docs][1])

所以 FP16 情况：

```text
unscale gradients
↓
检查 inf / NaN
↓
正常
→ AdamW.step()

异常
→ skip update
```

---

# 33. `scaler.update()` 又是干嘛？

紧接着：

```python
scaler.update()
```

如果 FP16：

> 根据**近期有没有 overflow / inf / NaN，动态调整 loss scaling factor。**

大体上**可以理解：**

```text
一直很稳定
→ scale 可以慢慢变大

出现 overflow
→ scale 变小
```

**目的就是寻找：**

```text
既尽量避免 underflow
又不要大到 overflow
```

的**合适 scale。**

**默认 BF16：**

```text
GradScaler disabled
```

**所以这一步基本 no-op。**

---

# 34. 最后终于看到 `zero_grad`

源码：

```python
optimizer.zero_grad(
    set_to_none=True
)
```

这发生在：

```text
optimizer update 完成之后
```

为什么必须清？

因为：

```text
.grad 默认累加
```

如果不清：

**第 9 个 batch 就会继续叠加：**

```text
第1~8批旧 gradient
+
第9批 gradient
```

那下一组就全乱了。

---

# 35. MiniMind 为什么不是每个 batch 都 `zero_grad()`？

因为它**故意**要做：

```text
gradient accumulation = 8
```

如果你写：

```python
for each batch:
    optimizer.zero_grad()
    loss.backward()
```

那么每次：

```text
上一批 grad
直接没了
```

就不可能累计 8 个 batch。

所以当前顺序必须：

```text
backward
backward
backward
...
backward
↓
step
↓
zero_grad
```

而不是：

```text
zero
backward

zero
backward

zero
backward
```

---

# 36. `set_to_none=True` 又是什么意思？

普通理解：

```python
optimizer.zero_grad()
```

像把：

```text
grad = [0,0,0,...]
```

但 MiniMind：

```python
zero_grad(set_to_none=True)
```

是直接：

```text
param.grad = None
```

而不是分配一整块全零 gradient tensor。

好处主要包括：

```text
减少一些内存写入
+
通常性能更好
```

PyTorch AMP 官方示例也使用/推荐这种方式作为可选性能优化。([PyTorch Docs][1])

所以下一轮 backward 时：

```text
grad=None

↓ 第一次写 gradient

grad=g
```

随后才继续累加。

---

# 37. 为什么训练开始之前没有看到一次 `zero_grad()`？

你可能已经注意到了。

`train_epoch()` 一进来，并没有：

```python
optimizer.zero_grad()
```

而是直接：

```python
backward()
```

这能行吗？

可以。

新创建的 Parameter：

```text
.grad 初始就是 None
```

而每一次正常 optimizer update 之后：

```python
optimizer.zero_grad(set_to_none=True)
```

又重新变成 None。

所以正常连续训练时成立。

---

# 38. 整段源码现在应该这样读

以后你再看到：

```python
with autocast_ctx:
    res = model(input_ids, labels=labels)
    loss = res.loss + res.aux_loss
    loss = loss / args.accumulation_steps
```

脑中翻译：

```text
低精度 forward
↓
算当前 micro-batch loss
↓
除以 8
为梯度累积做平均
```

---

看到：

```python
scaler.scale(loss).backward()
```

脑中：

```text
FP16：
    先 scale loss
    再 backward

BF16 默认：
    基本等价普通 backward

结果：
    梯度写入/累加到 param.grad
```

---

看到：

```python
if step % args.accumulation_steps == 0:
```

脑中：

```text
前 7 个 micro-batch：
只积梯度

第 8 个：
真正更新
```

---

看到：

```python
scaler.unscale_(optimizer)
```

：

```text
FP16：
恢复真实梯度尺度

BF16：
no-op
```

---

看到：

```python
clip_grad_norm_(...)
```

：

```text
梯度 norm > 1
→ 整体按比例缩小
```

---

看到：

```python
scaler.step(optimizer)
```

：

```text
调用 AdamW
↓
★★★★★ 参数真正变化 ★★★★★
```

---

看到：

```python
scaler.update()
```

：

```text
FP16：
调整下一轮 scaling factor

BF16：
基本无操作
```

---

看到：

```python
optimizer.zero_grad(set_to_none=True)
```

：

```text
清除上一组累计梯度
准备下一组 8 micro-batch
```

---

# 39. 用 MiniMind 默认配置完整模拟一次

这是你最应该记住的一张图。

```text
batch_size = 32
accumulation_steps = 8

────────────────────────

step 1
32 samples
forward
loss / 8
backward
grad += g1/8

────────────────────────

step 2
32 samples
forward
loss / 8
backward
grad += g2/8

────────────────────────

step 3
grad += g3/8

...

step 7
grad += g7/8

────────────────────────

step 8
32 samples
forward
loss / 8
backward
grad += g8/8

现在：

grad =
(g1+...+g8)/8

↓

unscale
（BF16 默认基本 no-op）

↓

clip_grad_norm(max_norm=1.0)

↓

AdamW.step()

★★★★★ MODEL WEIGHTS CHANGED ★★★★★

↓

GradScaler.update
（BF16 默认基本 no-op）

↓

zero_grad(set_to_none=True)

↓

开始 step 9
```

所以：

```text
8 × 32 = 256 samples
```

才对应一个 optimizer update。

---

# 40. 当前 MiniMind 源码还有一个很值得注意的尾 batch 处理

继续看 `train_epoch()` 最后：

```python
if (
    last_step > start_step
    and
    last_step % args.accumulation_steps != 0
):
    scaler.unscale_(optimizer)

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        args.grad_clip
    )

    scaler.step(optimizer)
    scaler.update()

    optimizer.zero_grad(set_to_none=True)
```

为什么存在？

假设一个 epoch 只有：

```text
83 个 micro-batch
```

正常：

```text
1~80
```

已经形成：

```text
10 个完整 accumulation group
```

**但是：**

```text
81
82
83
```

**还积着梯度。**

**如果程序直接结束 epoch：**

> **这三个 batch 的 gradient 就白算了。**

**所以作者在 epoch 末尾额外做一次：**

```text
step
```

**把剩余梯度也用掉。**

---

# 41. 但这里有一个源码层面的细节值得你看出来

注意前面每个 batch：

```python
loss = loss / args.accumulation_steps
```

始**终：**

```text
除以 8
```

**假设最后其实只剩：**

```text
3 个 batch
```

**那么尾部梯度：**

$$
\frac{g_{81}+g_{82}+g_{83}}8
$$

**而不是：**

$$
\frac{g_{81}+g_{82}+g_{83}}3
$$

**所以当前实现意味着：**

> **最后不足 8 个 micro-batch 的尾组，梯度尺度会比一个按实际尾组大小取平均的实现更小。**

这不是我拿通用教程替换源码，而是直接从当前源码推出来的。

对于大数据集来说，只发生 epoch 最后一小组，实际影响通常有限；但这是你以后做科研时应该能自己审代码发现的细节。

这就是我们想培养的能力：

> 不只是“代码能运行”，而是能判断训练语义到底是什么。

---

# 42. 还有一个你以后一定会混淆的术语问题

按照当前 MiniMind：

```python
for step, ...
```

中的 `step` 是：

# micro-batch step

不是：

# optimizer step

所以**如果：**

```text
step = 1000
accumulation_steps = 8
```

**大致只有：**

```text
125 次 optimizer update
```

**这一点后面看：**

```text
learning-rate schedule
checkpoint step
log interval
```

特别重要。

因为 MiniMind 当前很多地方使用的：

```text
step
```

都是这个 micro-batch counter。

---

# 43. 今天这段代码你现在应该能自己改写成最简版

把 AMP 等外围去掉，MiniMind 这段本质就是：

```python
for step, (x, y) in enumerate(loader, 1):

    loss = model(x, labels=y).loss

    loss = loss / 8

    loss.backward()

    if step % 8 == 0:

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        optimizer.zero_grad(
            set_to_none=True
        )
```

然后当前仓库为支持 FP16/BF16，把它升级成：

```python
with autocast_ctx:
    loss = ...

scaler.scale(loss).backward()

if step % accumulation_steps == 0:

    scaler.unscale_(optimizer)

    clip_grad_norm_(...)

    scaler.step(optimizer)

    scaler.update()

    optimizer.zero_grad(...)
```

**先理解简版，再看完整源码，你就不会被 API 名字吓住。**

---

# 44. 这一小节验收

现在你应该能直接对着 MiniMind 源码回答这些问题：

1. `loss / accumulation_steps` 是为了什么？
2. 为什么连续 8 次 `backward()` 的梯度不会互相覆盖？
3. step=1～7 时 MiniMind 参数有没有改变？
4. 真正修改 Parameter 的是哪一句？
5. 为什么 FP16 下要 `unscale_()` 后才能 clip？
6. 为什么 MiniMind 默认 BF16 时虽然写了 `scaler.scale()`，实际上没有真正做 Grad Scaling？
7. `clip_grad_norm_=1.0` 是不是把每一个梯度都截成 `[-1,1]`？
8. 为什么 `zero_grad()` 在 `optimizer.step()` 后面，而不是每个 batch 前都调用？
9. `step=800` 是否等于模型参数更新了 800 次？
10. 最后不足 8 个 batch 时，当前 MiniMind 是怎么处理的？

如果这 10 个你已经能解释，那么**MiniMind 的 backward → optimizer update 主链你就真正看懂了**。

---

# 下一部分我们继续顺着同一个 `train_pretrain.py`

下一小节我建议专门看这几行：

```python
lr = get_lr(
    epoch * iters + step,
    args.epochs * iters,
    args.learning_rate
)

for param_group in optimizer.param_groups:
    param_group['lr'] = lr
```

然后跳到：

```text
trainer/trainer_utils.py
→ get_lr()
```

当前源码：

```python
def get_lr(current_step, total_steps, lr):
    return lr * (
        0.1
        + 0.45 * (
            1
            + math.cos(
                math.pi * current_step / total_steps
            )
        )
    )
```

也就是说下一节我们可以直接从仓库算：

**MiniMind 的 Learning Rate 到底怎么从 `5e-4` 变化、为什么是 cosine decay、当前源码实际上有没有 warmup、梯度累积后这个 `step` 应该怎样理解，以及 LR 最终怎样真正影响 AdamW 参数更新。**

这部分尤其值得认真学，因为以后你做 MiniMind baseline 和自己的优化实验，**learning rate 会是你第一个真正需要会调的训练超参数之一**。



可以，而且从这一部分开始我直接把**主线切到 `jingyaogong/minimind-v` 当前 `master`**。前面 MiniMind 学到的 `loss → backward → optimizer → AMP → gradient accumulation` 全部继续有效；后面凡是和多模态训练有关的地方，我们都直接看 MiniMind-V 源码，不再让你学完 LLM 后重新对接一遍。

先把这个转换关系钉死。当前 MiniMind-V 的预训练入口已经变成：

```text
trainer/train_pretrain_vlm.py
```

训练 batch 也从：

```python
(input_ids, labels)
```

变成：

```python
(input_ids, labels, pixel_values)
```

然后 forward 变成：

```python
res = model(
    input_ids,
    labels=labels,
    pixel_values=pixel_values
)
```

所以之前的语言模型训练主干没丢，**只是多了一条视觉输入支路。**

---

# 第四轮 · 第 3 部分：MiniMind-V 的 Learning Rate 到底怎么变化

这一节我们直接看 MiniMind-V。

先打开：

```text
jingyaogong/minimind-v
└── trainer/
    ├── train_pretrain_vlm.py
    └── trainer_utils.py
```

今天只看两处源码。

第一处：

```text
trainer/train_pretrain_vlm.py
→ train_epoch()
```

第二处：

```text
trainer/trainer_utils.py
→ get_lr()
```

---

# 1. 先看 MiniMind-V 的真实代码

`trainer/train_pretrain_vlm.py → train_epoch()`：

```python
lr = get_lr(
    epoch * iters + step,
    args.epochs * iters,
    args.learning_rate
)

for param_group in optimizer.param_groups:
    param_group['lr'] = lr
```

这发生在每一个 batch forward **之前**。

然后跳到：

```text
trainer/trainer_utils.py
```

找到：

```python
def get_lr(current_step, total_steps, lr):
    return lr * (
        0.1
        + 0.45 * (
            1
            + math.cos(
                math.pi * current_step / total_steps
            )
        )
    )
```

这就是 MiniMind-V 当前**实际使用的全部 LR Scheduler。**

注意一个非常重要的事实：

> **当前 MiniMind-V 这段代码没有 warmup。**

我们之前讲**“大模型训练通常会有 warmup”，那是通用知识；但这个仓库当前源码并没有实现 warmup。**

以后一定区分：

```text
大模型训练通常怎么做
≠
MiniMind-V 当前代码怎么做
```

而你现在学项目，后者优先。

---

# 2. `learning_rate` 到底是什么？

MiniMind-V 预训练参数：

```python
parser.add_argument(
    "--learning_rate",
    type=float,
    default=4e-4,
    help="初始学习率"
)
```

所以：

```text
base learning rate = 4e-4
                   = 0.0004
```

最简单 SGD 的参数更新可以写成：

$$
\theta_{new}
=
\theta_{old}
-
\eta g
$$

其中：

```text
θ = 参数
g = 梯度
η = learning rate
```

所以 LR 可以先理解成：

> **梯度告诉你往哪个方向走，而 Learning Rate 控制这一轮大致走多远。**

虽然这里实际 optimizer 是 AdamW，不是简单 SGD，但 LR 仍然控制整体更新尺度。

---

# 3. 为什么 LR 太大会出问题？

假设某个参数：

```text
θ = 1.0
```

梯度：

```text
g = 5
```

如果：

```text
lr = 0.001
```

简单想象：

```text
更新量 ≈ 0.005
```

但如果：

```text
lr = 0.5
```

更新量可能变成：

```text
2.5
```

一步直接从：

```text
1.0
```

跨到：

```text
-1.5
```

就可能不停：

```text
左边越过去
→ 右边越过去
→ loss 震荡
```

甚至：

```text
gradient explosion
NaN
```

---

# 4. LR 太小同样有问题

如果：

```text
lr = 1e-10
```

每次参数：

```text
只动一点点
```

那么：

```text
loss：
5.1
5.099999
5.099998
...
```

训练几乎不动。

所以 LR 一直是你以后科研实验里最敏感的训练超参数之一。

---

# 5. MiniMind-V 不是固定 `4e-4`

这一点很重要。

源码不是：

```python
optimizer = AdamW(..., lr=4e-4)
```

然后从头到尾一直 4e-4。

虽然初始化时确实：

```python
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=args.learning_rate
)
```

但**每个训练 step 前都会：**

```python
lr = get_lr(...)
```

**然后：**

```python
for param_group in optimizer.param_groups:
    param_group['lr'] = lr
```

**把 AdamW 当前 LR 改掉。**

所以：

```text
4e-4
```

实际上是：

# 初始 / 最大 LR

不是整个训练过程中永远固定的 LR。

---

# 6. 把 MiniMind-V 的公式真的算一遍

源码：

$$
lr_t
=
lr_0
\left[
0.1+
0.45
\left(
1+\cos\left(
\pi \frac{t}{T}
\right)
\right)
\right]
$$

其中：

```text
lr0 = args.learning_rate
t   = current_step
T   = total_steps
```

为了理解方便，我们先忽略“第一步从 1 而不是 0 开始”的小细节。

---

# 7. 训练刚开始

假设：

$$
t=0
$$

那么：

$$
\cos(0)=1
$$

于是：

$$
0.1+0.45(1+1)
$$

$$
=0.1+0.9
$$

$$
=1
$$

所以：

$$
lr=lr_0
$$

**MiniMind-V Pretrain：**
$$
lr=4\times10^{-4}
$$

即：

```text
0.0004
```

---

# 8. 训练进行到一半

此时：

$$
t/T=0.5
$$

所以：

$$
\cos(\pi/2)=0
$$

得到：

$$
0.1+0.45(1+0)
=0.55
$$

所以 LR 变成初始值：

```text
55%
```

MiniMind-V：

$$
4e-4\times0.55
=
2.2e-4
$$

也就是：

```text
0.000220
```

---

# 9. 训练结束

此时：

$$
t=T
$$

所以：

$$
\cos(\pi)=-1
$$

于是：

$$
0.1+0.45(1-1)
=
0.1
$$

所以最终：

```text
learning rate = 初始值的 10%
```

MiniMind-V：

$$
4e-4\times0.1
=
4e-5
$$

所以完整趋势大约是：

```text
4.0e-4
   │
   │\
   │ \
   │  \
   │   ╲
   │    ╲
   │      ╲
   │        ╲____
4.0e-5
   └────────────────→ training progress
   0%       50%      100%
```

不是线性下降，而是：

# Cosine Decay

---

# 10. 为什么 cosine 比突然减 LR 合理？

假设训练过程：

```text
早期
模型一团乱
```

需要：

```text
比较大的更新
```

到了训练后期：

```text
模型已经基本成型
```

希望：

```text
小步精修
```

因此：

```text
训练早期：
大 LR
→ 快速学习

训练后期：
小 LR
→ 精细调整
```

Cosine 的**特点是：**

> **LR 平滑变化，不突然跳变。**

---

# 11. 但 MiniMind-V 的 schedule 有一个特点

它最终不是：

```text
→ 0
```

而是：

```text
→ 0.1 × base_lr
```

即预训练最终：

```text
4e-5
```

因为公式里有固定：

```python
0.1
```

所以你可以把当前实现理解成：

```text
cosine decay
from 100%
to 10%
```

而不是：

```text
100%
to 0%
```

---

# 12. 源码中的 `current_step` 到底是什么？

现在回去看调用：

```python
lr = get_lr(
    epoch * iters + step,
    args.epochs * iters,
    args.learning_rate
)
```

所以：

```text
current_step
=
epoch * iters + step
```

假设：

```text
epochs = 2
每个 epoch = 1000 batch
```

那么：

```text
total_steps
=
2 × 1000
=
2000
```

---

第一轮：

```text
epoch = 0

step=1
→ current_step=1

step=500
→ current_step=500

step=1000
→ current_step=1000
```

第二轮：

```text
epoch=1

step=1
→ current_step=1001

...

step=1000
→ current_step=2000
```

所以 **LR：**

> **不会每个 epoch 重新从 4e-4 开始。**

**而是贯穿整个训练过程连续衰减。**

---

# 13. 这里的 `step` 仍然是 DataLoader step

继续承接上一节。

源码：

```python
for step, (
    input_ids,
    labels,
    pixel_values
) in enumerate(loader, ...):
```

所以这里的 `step`：

```text
一个 batch
=
一个 micro-batch step
```

不是必然等于：

```text
optimizer update 次数
```

---

# 14. 但 MiniMind-V 默认刚好没有这个问题

这里就出现和 MiniMind 一个非常重要的区别。

你之前学 MiniMind：

```text
accumulation_steps = 8
```

但是当前 MiniMind-V Pretrain：

```python
parser.add_argument(
    "--accumulation_steps",
    default=1,
    ...
)
```

所以默认：

```text
每一个 DataLoader step
=
一次 backward
=
一次 optimizer update
```

因为：

```python
if step % 1 == 0:
```

永远成立。

所以当前默认 MiniMind-V：

```text
step 数
≈ optimizer update 数
```

这**比 MiniMind 那个默认 8 倍梯度累积简单。**

---

# 15. 如果以后你把 `accumulation_steps` 改成 8 呢？

这就非常值得注意。

LR 还是：

```python
get_lr(epoch * iters + step, ...)
```

所以：

```text
每个 micro-batch
LR 都变化一次
```

但：

```text
每 8 个 micro-batch
参数才真正 update 一次
```

例如：

```text
step 1 → 算 LR1 → 不更新
step 2 → 算 LR2 → 不更新
...
step 8 → 算 LR8 → optimizer.step()
```

真正执行 AdamW update 时使用的是：

```text
LR8
```

因为 optimizer 当前最后被设置成了它。

所以：

> 当前 MiniMind-V scheduler 是按 **micro-batch step** 计数，而不是按 optimizer update 计数。

默认 `accumulation_steps=1` 没问题。

如果以后我们为了显存把梯度累积调大，这一点就必须记住。

---

# 16. 现在出现 MiniMind → MiniMind-V 的第一个真正区别

我们不能只看：

```text
lr 从多少下降到多少
```

还得问：

> **这个 LR 到底在更新哪些参数？**

当前 MiniMind-V Pretrain 的 optimizer：

```python
optimizer = optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=args.learning_rate
)
```

注意它没有直接：

```python
AdamW(model.parameters())
```

而是：

```python
filter(
    lambda p: p.requires_grad,
    ...
)
```

也就是说：

> **只把没有被冻结的参数交给 AdamW。**

这对 VLM 极其重要。

---

# 17. 那 MiniMind-V 到底冻结了谁？

直接跳源码：

```text
trainer/trainer_utils.py
→ init_vlm_model()
```

当前首先执行：

```python
for name, param in model.named_parameters():
    if 'vision_proj' not in name:
        param.requires_grad = False
```

也就是**一上来：**

```text
除了 vision_proj
其他全冻结
```

**然后根据：**

```python
freeze_llm
```

**决定额外开放哪些 LLM 参数。**

---

# 18. MiniMind-V Pretrain 默认 `freeze_llm=2`

看：

```text
trainer/train_pretrain_vlm.py
```

：

```python
parser.add_argument(
    '--freeze_llm',
    default=2,
    type=int,
    choices=[0, 1, 2],
    help="冻结策略（0=完全可训练，1=冻结+解冻首尾层，2=完全冻结仅训练proj）"
)
```

所以**当前默认 Pretrain：**

# **只训练 `vision_proj`**

**而：**

```text
LLM
冻结

Vision Encoder
冻结

Vision Projector
训练
```

**这是你之后正式 MiniMind-V 项目非常关键的一张图：**

```text
Image
  │
  ▼
Vision Encoder
(SigLIP)
❄ frozen
  │
  ▼
visual features
  │
  ▼
Vision Projector
🔥 trainable
  │
  ▼
LLM embedding space
  │
  ▼
MiniMind LLM
❄ frozen
```

这一步本**质上不是：**

> **从零重新训练整个 VLM。**

而是先学：

> **怎样把视觉特征“翻译”到 LLM 能理解的 hidden space。**

这一点就是我们**从 MiniMind 转 MiniMind-V 后需要重点补的新知识。**

---

# 19. 这也解释了为什么 VLM Pretrain 可以用 `4e-4`

当前 VLM Pretrain：

```text
learning_rate = 4e-4
freeze_llm = 2
```

也就是说主要训练：

```text
vision_proj
```

而不是拿 `4e-4` 去猛烈更新已经训练好的整个 LLM。

所以你以后不能只说：

> “MiniMind-V 学习率是 4e-4。”

更准确应该说：

> **MiniMind-V 当前预训练阶段默认冻结视觉编码器与 LLM，仅训练 vision projector，此阶段 base LR 为 4e-4，并进行 cosine decay。**

这是科研代码阅读应该达到的准确程度。

---

# 20. MiniMind-V 的 SFT 又完全不一样

再打开：

```text
trainer/train_sft_vlm.py
```

当前默认：

```python
parser.add_argument(
    "--learning_rate",
    default=5e-6,
    ...
)

parser.add_argument(
    '--freeze_llm',
    default=1,
    ...
)
```

也就是 SFT：

```text
LR = 5e-6
freeze_llm = 1
```

比 pretrain：

```text
4e-4
```

小了非常多。

比例：

$$
\frac{4e-4}{5e-6}=80
$$

也就是说：

> SFT 默认 base LR 只有 VLM Pretrain 的 **1/80**。

---

# 21. 为什么 SFT 学习率小这么多？

因为到了这个阶段，你已经有：

```text
训练好的 LLM
+
训练好的视觉投影层
```

**不希望再大幅改坏已有能力。**

**所以从直觉上：**

```text
VLM Pretrain
→ 让视觉和语言先对齐
→ 可以走大一些

VLM SFT
→ 在已有能力上精细调整
→ LR 很小
```

**而且 `freeze_llm=1` 会额外解冻 LLM 的首尾层。**

当前源码：

```python
elif freeze_llm == 1:
    last_idx = vlm_config.num_hidden_layers - 1

    for name, param in model.model.named_parameters():
        if (
            'layers.0.' in name
            or
            f'layers.{last_idx}.' in name
        ):
            param.requires_grad = True
```

默认 8 层的话，就是：

```text
layer 0
+
layer 7
```

被解冻。

于是 SFT 大致变成：

```text
Vision Encoder
❄

Vision Projector
🔥

LLM 首层
🔥

LLM 中间层
❄

LLM 尾层
🔥
```

所以这时候小 LR 很合理。

---

# 22. 同一个 `get_lr()` 被 Pretrain 和 SFT 共用

这也很方便。

当前：

```text
train_pretrain_vlm.py
```

和：

```text
train_sft_vlm.py
```

都调用：

```python
get_lr(...)
```

所以 schedule 形状一样：

```text
100%
↓ cosine
10%
```

只是 base LR 不同。

---

# 23. 所以 MiniMind-V Pretrain 的 LR 曲线

Base：

```text
4e-4
```

对应：

| 训练进度 |  LR 系数 |   实际 LR |
| -------: | -------: | --------: |
|       0% |     1.00 | `4.00e-4` |
|      25% | 约 0.868 | `3.47e-4` |
|      50% |     0.55 | `2.20e-4` |
|      75% | 约 0.232 | `9.27e-5` |
|     100% |     0.10 | `4.00e-5` |

---

# 24. MiniMind-V SFT 的同一条曲线

Base：

```text
5e-6
```

：

| 训练进度 | LR 系数 |   实际 LR |
| -------: | ------: | --------: |
|       0% |    1.00 | `5.00e-6` |
|      50% |    0.55 | `2.75e-6` |
|     100% |    0.10 | `5.00e-7` |

结构相同，尺度完全不同。

---

# 25. 现在把 MiniMind 和 MiniMind-V 的训练差异正式对齐

以后我们以右边两列为主。

| 项目             | MiniMind Pretrain | MiniMind-V Pretrain |             MiniMind-V SFT |
| ---------------- | ----------------: | ------------------: | -------------------------: |
| **batch 内容**   |          **text** |    **text + image** |   **conversation + image** |
| `batch_size`     |                32 |                  16 |                          4 |
| `max_seq_len`    |               340 |                 450 |                    **768** |
| base LR          |            `5e-4` |              `4e-4` |                     `5e-6` |
| accumulation     |                 8 |                   1 |                          1 |
| 初始化权重       |            `none` |               `llm` |             `pretrain_vlm` |
| **默认冻结策略** |            **无** |  **只训 projector** | **projector + LLM 首尾层** |
| 图像输入         |                无 |      `pixel_values` |             `pixel_values` |
| Dataset          | `PretrainDataset` |        `VLMDataset` |               `VLMDataset` |

MiniMind-V **参数都来自当前两个训练脚本。**

看到这里你应该已经能感觉到：

> 前面 MiniMind 不是白学，而是在给你建立 LLM backbone 的训练语言；现在我们只是在这个 backbone 上多接视觉支路。

---

# 26. 你之后真正跑 MiniMind-V 的训练链应该这样理解

不是：

```text
随机初始化整个 VLM
↓
图片+文本全模型一起训
```

当前仓库默认路线更接近：

```text
已经训练好的 MiniMind LLM
        │
        │
        ├────────────────────┐
        │                    │
        │                ❄ LLM
        │                    │
Image                       Text
 ↓                           ↓
SigLIP                  Tokenizer
❄ frozen                    │
 ↓                           │
visual features              │
 ↓                           │
vision_proj 🔥               │
 ↓                           │
visual tokens ───────────────┘
        │
        ▼
      LLM
        │
        ▼
      Loss
```

然后先：

```text
VLM Pretrain
```

再：

```text
VLM SFT
```

这才是后面我们真正要跑的主路线。

---

# 27. 这一节源码你应该形成的阅读反应

以后看到：

```python
lr = get_lr(
    epoch * iters + step,
    args.epochs * iters,
    args.learning_rate
)
```

自动理解：

```text
根据整个训练进度
重新算当前 cosine LR
```

看到：

```python
param_group['lr'] = lr
```

自动理解：

```text
把 AdamW 下一次参数更新使用的 LR 改成当前值
```

看到：

```python
filter(
    lambda p: p.requires_grad,
    model.parameters()
)
```

自动想到：

```text
不是所有 VLM 参数都会被更新
↓
要去查 freeze strategy
```

看到：

```python
freeze_llm=2
```

自动想到：

```text
VLM Pretrain 默认：
只训 vision_proj
```

看到：

```python
freeze_llm=1
```

自动想到：

```text
VLM SFT 默认：
vision_proj
+
LLM 首尾层
```

这已经是从普通 PyTorch 学习进入：

> **真正读 VLM 训练代码**

了。

---

# 28. 从现在开始我们的路线也正式改一下

前面原本第四轮后面还准备继续围着 MiniMind 的 checkpoint / DDP。

现在不用在纯文本 MiniMind 上继续耗太久。

接下来我会直**接以 MiniMind-V 为主体，顺序变成：**

```text
当前：
✓ loss / backward
✓ gradient accumulation
✓ AMP / GradScaler
✓ clipping / AdamW
✓ MiniMind-V Learning Rate / freeze strategy

下一部分：
MiniMind-V 到底比 MiniMind 多了什么
↓
model/model_vlm.py

重点看：
Vision Encoder
↓
pixel_values
↓
image features
↓
Vision Projector
↓
image tokens
↓
怎样插进 LLM token sequence
↓
最终怎样和文本一起进入 MiniMind
```

这一步我认为现在比继续讲 checkpoint 更重要。

因为你已经会：

```text
LLM forward
+
训练 loop
```

现在真正阻碍你开始 MiniMind-V 的东西已经变成：

> **图片到底怎样进入 Decoder-only LLM。**

下一部分我们就直接打开当前仓库：

```text
model/model_vlm.py
```

按照你要求的方式，逐段源码带读：

```text
VLMConfig
→ Vision Encoder
→ Vision Projector
→ encode_image()
→ image token 替换/融合
→ MiniMind LLM forward
→ logits/loss
```

同时再跳：

```text
dataset/lm_dataset.py
→ VLMDataset
→ vlm_collate_fn
```

把：

```text
一张 jpg
→ pixel_values
→ vision feature
→ visual token
→ LLM hidden state
```

完整串起来。

**从这一部分开始，我们就已经算正式切入你最终要训练的 MiniMind-V 了，后面不需要再“学完 MiniMind 再转 V”。**