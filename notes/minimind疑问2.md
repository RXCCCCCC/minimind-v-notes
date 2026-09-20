[TOC]

# k,v后面两个维度相乘不是对应不上784吗,那不是少了一半信息吗

对，你这里看到的“少一半”确实存在，但**不是在矩阵乘法时维度对不上，也不是最后 hidden state 少了一半**。它发生在 **GQA 故意让 K/V 的表示容量比 Q 小一半**。

你说的应该是 MiniMind 里类似：

```text
hidden_size C = 768
Q heads = 8
KV heads = 4
head_dim D = 96
```

所以：

```text
Q: 8 × 96 = 768 维
K: 4 × 96 = 384 维
V: 4 × 96 = 384 维
```

你于是发现：

> K/V 明明**只有 384，Q 是 768，怎么乘？而且是不是丢了一半信息？**

这两个问题分开看。

---

# 1. Q 和 K 不是直接拿 768 和 384 去乘

不会做：

```text
[768] × [384]
```

这种东西。

Attention 是**按 head 分开算的**。

**每个 Q head 都是：**

```text
[96]
```

**每个 KV head 也是：**

```text
[96]
```

所以**真正计算的是：**

```text
Q head 0 [96]
    ×
K head 0 [96]
```

**完全对应。**

**GQA 只是这样分组：**

```text
Q0 [96] ─┐
          ├── 使用 KV0 [96]
Q1 [96] ─┘

Q2 [96] ─┐
          ├── 使用 KV1 [96]
Q3 [96] ─┘

Q4 [96] ─┐
          ├── 使用 KV2 [96]
Q5 [96] ─┘

Q6 [96] ─┐
          ├── 使用 KV3 [96]
Q7 [96] ─┘
```

所以每次做：

$$
q_h k_g^T
$$

的时候，都是：

$$
[96]\cdot[96]
$$

**没有维度对不上的问题。**

---

# 2. MiniMind 代码甚至会把 K/V 在 head 维“重复”

MiniMind 先得到：

```text
Q : [B,T,8,96]

K : [B,T,4,96]
V : [B,T,4,96]
```

然后**调用：**

```python
repeat_kv(xk, self.n_rep)
repeat_kv(xv, self.n_rep)
```

**这里：**

```text
n_rep = 8 / 4 = 2
```

**所以逻辑上会变成：**

```text
原 K heads：

K0 K1 K2 K3

↓

重复：

K0 K0 K1 K1 K2 K2 K3 K3
```

V 同理：

```text
V0 V0 V1 V1 V2 V2 V3 V3
```

于是 Attention 真正算的时候，**可以看成：**

```text
Q : [B,8,T,96]
K : [B,8,T,96]
V : [B,8,T,96]
```

**shape 完全对应。MiniMind 当前源码就是先保留 4 个 KV heads，**再通过 `repeat_kv` 配合 8 个 Q heads 做注意力。

---

# 3. 但你第二个判断其实很敏锐：那 K/V 不就是少一半信息容量吗？

### 是的。

更准确地说：

> **相对于普通 MHA，GQA 确实减少了 K/V 的表示容量。**

普通 MHA：

```text
Q : 8 × 96 = 768
K : 8 × 96 = 768
V : 8 × 96 = 768
```

GQA：

```text
Q : 8 × 96 = 768
K : 4 × 96 = 384
V : 4 × 96 = 384
```

也就是说 K/V 从：

```text
每个 token 8 套表示
```

变成：

```text
每个 token 4 套表示
```

这是**故意的 trade-off**。

---

# 4. 但不是“先有 768 维 K/V，然后砍掉一半”

这个区别很重要。

不是：

```text
hidden [768]
↓
先产生 K [768]
↓
扔掉一半
↓
K [384]
```

而是**一开始：**

```python
k_proj = Linear(768, 384)
v_proj = Linear(768, 384)
```

**模型训练时自己学习：**

> **如何把原来 768 维 hidden state 中，对 K/V 有用的信息编码进这 384 维里面。**

**所以更像：**

```text
768维原始信息
↓
学习一个压缩映射
↓
384维 K 表示
```

而**不是粗暴截断：**

```text
前384维留下
后384维扔掉
```

---

# 5. 为什么敢把 K/V 压到一半？

因为 Q、K、V 的职责不一样。

Q：

> 当前这个 head **想怎么查询信息？**

K：

> 我作为历史 token，**应该怎么被匹配？**

V：

> 如果别人关注我，**我提供什么信息？**

实验上**发现：**

> **很多不同 Query heads 没有必要各自配一套完全独立的 K/V。**

**所以可以让：**

```text
Q0、Q1
```

**使用同一个：**

```text
K0、V0
```

**但：**

```text
Q0 ≠ Q1
```

**因此最后结果还是不同。**

---

# 6. 例如两个 Q head 虽然共用同一个 K

假设历史 token 是：

```text
我 喜欢 机器 学习
```

共享的：

```text
K0
```

是一整组：

```text
K(我)
K(喜欢)
K(机器)
K(学习)
```

Q head 0 去查：

```text
q0 · K
```

得到：

```text
我       0.05
喜欢     0.10
机器     0.75
学习     0.10
```

Q head 1：

```text
q1 · K
```

可能得到：

```text
我       0.30
喜欢     0.40
机器     0.20
学习     0.10
```

虽然 K 一模一样，

但是：

```text
q0 ≠ q1
```

所以 attention weights 完全可以不同。

---

# 7. 更重要的是：虽然 V 只有 4 个 head，最后输出仍然有 8 个 head

这个地方特别容易产生“少一半”的错觉。

假设：

```text
Q0、Q1
```

共享：

```text
V0
```

但是：

```text
Q0
↓
得到 attention weights A0

Q1
↓
得到 attention weights A1
```

于是：

$$
O_0=A_0V_0
$$

和：

$$
O_1=A_1V_0
$$

虽然使用同一个 V，

因为：

$$
A_0\ne A_1
$$

所以：

$$
O_0\ne O_1
$$

通常**仍然是两份不同的：**

```text
[96]
```

**输出。**

---

# 8. 因此最后还是恢复成 768

**八个 Query heads** 最后分别得到：

```text
O0 [96]
O1 [96]
O2 [96]
O3 [96]
O4 [96]
O5 [96]
O6 [96]
O7 [96]
```

拼起来：

```text
Concat
↓
8 × 96
↓
768
```

所以：

```text
Attention 输出：
[B,T,768]
```

仍然完整回到 hidden size。

之后：

```text
o_proj
768 → 768
```

再回到 Transformer 主干。

所以不是：

```text
输入 768
↓
Attention
↓
只剩 384
```

而是：

```text
输入 hidden
[B,T,768]

↓ QKV

Q: [B,T,8,96]
K: [B,T,4,96]
V: [B,T,4,96]

↓ K/V共享给Q heads

8套 Attention output
每套 [96]

↓ concat

[B,T,768]
```

---

# 9. 那到底“损失”的是什么？

不是最终输出维度。

**损失的是：**

> **K/V 表示的独立性和容量。**

**MHA 可以：**

```text
Q0 ↔ K0,V0
Q1 ↔ K1,V1
Q2 ↔ K2,V2
...
Q7 ↔ K7,V7
```

八套**完全独立。**

GQA：

```text
Q0,Q1 ↔ K0,V0
Q2,Q3 ↔ K1,V1
Q4,Q5 ↔ K2,V2
Q6,Q7 ↔ K3,V3
```

因此**自由度确实少了。**

所以 GQA **不是“免费优化”，而是一个：**

$$
\boxed{\text{模型表达能力} \leftrightarrow \text{推理效率}}
$$

的折中。

---

# 10. 为什么值得这么换？

因为**在自回归推理里，真正需要长期保存的是 K/V。**

MHA：

```text
K cache: [B,T,8,96]
V cache: [B,T,8,96]
```

GQA：

```text
K cache: [B,T,4,96]
V cache: [B,T,4,96]
```

直接：

> **KV Cache 减半。**

**而长上下文下：**

```text
T = 8k
32k
128k
...
```

**KV Cache 会非常大。**

所以**用一定程度的 K/V 表示能力换：**

```text
更少显存
+
更高推理吞吐
```

**通常是很划算的。**

---

# 11. 你可以顺便理解 MHA → GQA → MQA 的光谱

### MHA

```text
8 Q
8 K
8 V

表达能力更充分
KV Cache 最大
```

### GQA

```text
8 Q
4 K
4 V

中间折中
```

### MQA

```text
8 Q
1 K
1 V

所有 Q 都共享一套 KV
KV Cache 最小
但共享最激进
```

所以 GQA 本质上就是：

> **我觉得 K/V 没必要像 Q 那样有那么多完全独立的 heads，但也不想像 MQA 那样所有 Q 都共用一套，因此取个中间值。**

---

## 最后把你这个问题一句话说透

你看到：

```text
Q = 8×96 = 768
KV = 4×96 = 384
```

确实意味着：

> **GQA 把每个 token 的 K/V 表示容量从 768 压缩到了 384，这是有意减少的表示自由度。**

但是 Attention 计算时：

> **每个 Query head 仍然只和一个同为 96 维的 KV head 运算，所以不存在维度不匹配；共享的 KV 会被多个不同 Q 以不同 attention weights 使用，最终仍产生 8 个 96 维 head outputs，concat 后重新得到 768 维 hidden state。**

你这个疑问其实非常到位，因为这正是理解“**为什么 GQA 能省 KV Cache，但又不是把 hidden size 直接砍半**”的关键。

对，你这里看到的“少一半”确实存在，但**不是在矩阵乘法时维度对不上，也不是最后 hidden state 少了一半**。它发生在 **GQA 故意让 K/V 的表示容量比 Q 小一半**。

你说的应该是 MiniMind 里类似：

```text
hidden_size C = 768
Q heads = 8
KV heads = 4
head_dim D = 96
```

所以：

```text
Q: 8 × 96 = 768 维
K: 4 × 96 = 384 维
V: 4 × 96 = 384 维
```

你于是发现：

> K/V 明明只有 384，Q 是 768，怎么乘？而且是不是丢了一半信息？

这两个问题分开看。

------

# 1. Q 和 K 不是直接拿 768 和 384 去乘

不会做：

```text
[768] × [384]
```

这种东西。

Attention 是**按 head 分开算的**。

每个 Q head 都是：

```text
[96]
```

每个 KV head 也是：

```text
[96]
```

所以真正计算的是：

```text
Q head 0 [96]
    ×
K head 0 [96]
```

完全对应。

GQA 只是这样分组：

```text
Q0 [96] ─┐
          ├── 使用 KV0 [96]
Q1 [96] ─┘

Q2 [96] ─┐
          ├── 使用 KV1 [96]
Q3 [96] ─┘

Q4 [96] ─┐
          ├── 使用 KV2 [96]
Q5 [96] ─┘

Q6 [96] ─┐
          ├── 使用 KV3 [96]
Q7 [96] ─┘
```

所以每次做：

qhkgTq_h k_g^T

的时候，都是：

[96]⋅[96][96]\cdot[96]

没有维度对不上的问题。

------

# 2. MiniMind 代码甚至会把 K/V 在 head 维“重复”

MiniMind 先得到：

```text
Q : [B,T,8,96]

K : [B,T,4,96]
V : [B,T,4,96]
```

然后调用：

```python
repeat_kv(xk, self.n_rep)
repeat_kv(xv, self.n_rep)
```

这里：

```text
n_rep = 8 / 4 = 2
```

所以逻辑上会变成：

```text
原 K heads：

K0 K1 K2 K3

↓

重复：

K0 K0 K1 K1 K2 K2 K3 K3
```

V 同理：

```text
V0 V0 V1 V1 V2 V2 V3 V3
```

于是 Attention 真正算的时候，可以看成：

```text
Q : [B,8,T,96]
K : [B,8,T,96]
V : [B,8,T,96]
```

shape 完全对应。MiniMind 当前源码就是先保留 4 个 KV heads，再通过 `repeat_kv` 配合 8 个 Q heads 做注意力。

------

# 3. 但你第二个判断其实很敏锐：那 K/V 不就是少一半信息容量吗？

### 是的。

更准确地说：

> **相对于普通 MHA，GQA 确实减少了 K/V 的表示容量。**

普通 MHA：

```text
Q : 8 × 96 = 768
K : 8 × 96 = 768
V : 8 × 96 = 768
```

GQA：

```text
Q : 8 × 96 = 768
K : 4 × 96 = 384
V : 4 × 96 = 384
```

也就是说 K/V 从：

```text
每个 token 8 套表示
```

变成：

```text
每个 token 4 套表示
```

这是**故意的 trade-off**。

------

# 4. 但不是“先有 768 维 K/V，然后砍掉一半”

这个区别很重要。

不是：

```text
hidden [768]
↓
先产生 K [768]
↓
扔掉一半
↓
K [384]
```

而是一开始：

```python
k_proj = Linear(768, 384)
v_proj = Linear(768, 384)
```

模型训练时自己学习：

> 如何把原来 768 维 hidden state 中，对 K/V 有用的信息编码进这 384 维里面。

所以更像：

```text
768维原始信息
↓
学习一个压缩映射
↓
384维 K 表示
```

而不是粗暴截断：

```text
前384维留下
后384维扔掉
```

------

# 5. 为什么敢把 K/V 压到一半？

因为 Q、K、V 的职责不一样。

Q：

> 当前这个 head **想怎么查询信息？**

K：

> 我作为历史 token，**应该怎么被匹配？**

V：

> 如果别人关注我，**我提供什么信息？**

实验上发现：

> **很多不同 Query heads 没有必要各自配一套完全独立的 K/V。**

所以可以让：

```text
Q0、Q1
```

使用同一个：

```text
K0、V0
```

但：

```text
Q0 ≠ Q1
```

因此最后结果还是不同。

------

# 6. 例如两个 Q head 虽然共用同一个 K

假设历史 token 是：

```text
我 喜欢 机器 学习
```

共享的：

```text
K0
```

是一整组：

```text
K(我)
K(喜欢)
K(机器)
K(学习)
```

Q head 0 去查：

```text
q0 · K
```

得到：

```text
我       0.05
喜欢     0.10
机器     0.75
学习     0.10
```

Q head 1：

```text
q1 · K
```

可能得到：

```text
我       0.30
喜欢     0.40
机器     0.20
学习     0.10
```

虽然 K 一模一样，

但是：

```text
q0 ≠ q1
```

所以 attention weights 完全可以不同。

------

# 7. 更重要的是：虽然 V 只有 4 个 head，最后输出仍然有 8 个 head

这个地方特别容易产生“少一半”的错觉。

假设：

```text
Q0、Q1
```

共享：

```text
V0
```

但是：

```text
Q0
↓
得到 attention weights A0

Q1
↓
得到 attention weights A1
```

于是：

O0=A0V0O_0=A_0V_0

和：

O1=A1V0O_1=A_1V_0

虽然使用同一个 V，

因为：

A0≠A1A_0\ne A_1

所以：

O0≠O1O_0\ne O_1

通常仍然是两份不同的：

```text
[96]
```

输出。

------

# 8. 因此最后还是恢复成 768

八个 Query heads 最后分别得到：

```text
O0 [96]
O1 [96]
O2 [96]
O3 [96]
O4 [96]
O5 [96]
O6 [96]
O7 [96]
```

拼起来：

```text
Concat
↓
8 × 96
↓
768
```

所以：

```text
Attention 输出：
[B,T,768]
```

仍然完整回到 hidden size。

之后：

```text
o_proj
768 → 768
```

再回到 Transformer 主干。

所以不是：

```text
输入 768
↓
Attention
↓
只剩 384
```

而是：

```text
输入 hidden
[B,T,768]

↓ QKV

Q: [B,T,8,96]
K: [B,T,4,96]
V: [B,T,4,96]

↓ K/V共享给Q heads

8套 Attention output
每套 [96]

↓ concat

[B,T,768]
```

------

# 9. 那到底“损失”的是什么？

不是最终输出维度。

损失的是：

> **K/V 表示的独立性和容量。**

MHA 可以：

```text
Q0 ↔ K0,V0
Q1 ↔ K1,V1
Q2 ↔ K2,V2
...
Q7 ↔ K7,V7
```

八套完全独立。

GQA：

```text
Q0,Q1 ↔ K0,V0
Q2,Q3 ↔ K1,V1
Q4,Q5 ↔ K2,V2
Q6,Q7 ↔ K3,V3
```

因此自由度确实少了。

所以 GQA 不是“免费优化”，而是一个：

模型表达能力↔推理效率\boxed{\text{模型表达能力} \leftrightarrow \text{推理效率}}

的折中。

------

# 10. 为什么值得这么换？

因为在自回归推理里，真正需要长期保存的是 K/V。

MHA：

```text
K cache: [B,T,8,96]
V cache: [B,T,8,96]
```

GQA：

```text
K cache: [B,T,4,96]
V cache: [B,T,4,96]
```

直接：

> **KV Cache 减半。**

而长上下文下：

```text
T = 8k
32k
128k
...
```

KV Cache 会非常大。

所以用一定程度的 K/V 表示能力换：

```text
更少显存
+
更高推理吞吐
```

通常是很划算的。

------

# 11. 你可以顺便理解 MHA → GQA → MQA 的光谱

### MHA

```text
8 Q
8 K
8 V

表达能力更充分
KV Cache 最大
```

### GQA

```text
8 Q
4 K
4 V

中间折中
```

### MQA

```text
8 Q
1 K
1 V

所有 Q 都共享一套 KV
KV Cache 最小
但共享最激进
```

所以 GQA 本质上就是：

> **我觉得 K/V 没必要像 Q 那样有那么多完全独立的 heads，但也不想像 MQA 那样所有 Q 都共用一套，因此取个中间值。**

------

## 最后把你这个问题一句话说透

你看到：

```text
Q = 8×96 = 768
KV = 4×96 = 384
```

确实意味着：

> **GQA 把每个 token 的 K/V 表示容量从 768 压缩到了 384，这是有意减少的表示自由度。**

但是 Attention 计算时：

> **每个 Query head 仍然只和一个同为 96 维的 KV head 运算，所以不存在维度不匹配；共享的 KV 会被多个不同 Q 以不同 attention weights 使用，最终仍产生 8 个 96 维 head outputs，concat 后重新得到 768 维 hidden state。**

你这个疑问其实非常到位，因为这正是理解“**为什么 GQA 能省 KV Cache，但又不是把 hidden size 直接砍半**”的关键。

# **LM Head**  又是什么东西

这里的 **LM Head** 和我们前面一直说的 **Attention Head 完全不是一回事**。你现在最好把这两个 “Head” 彻底分开。

> **Attention Head：存在于每一层 Attention 内部，负责从上下文里取信息。**
> **LM Head：整个 Transformer 最后面的输出层，负责把最终 hidden state 变成“下一个 token 的词表分数”。**

# 1. 它出现在整个生成流程的最后

我们刚刚讲的生成流程其实是：

```text
新 token
↓
Embedding
↓
Transformer Layer 0
  └─ 多个 Attention Heads
↓
Transformer Layer 1
  └─ 多个 Attention Heads
↓
...
↓
Transformer Layer 7
↓
Final RMSNorm
↓
最终 hidden state
↓
LM Head       ← 就是这里
↓
整个词表的 logits
↓
选出 next token
```

所以 Attention Head 和 LM Head 的位置完全不同。

---

# 2. LM Head 到底干了什么？

假设 MiniMind 的：

```text
hidden_size C = 768
vocab_size V = 6400
```

一个 token 经过所有 Transformer 层以后，得到最终：

```text
hidden state

[C]
=
[768]
```

这个 768 维向量**已经融合了整个前缀的信息。**

比如：

```text
中国 的 首都 是
```

最后一个位置 `"是"` 的 final hidden state：

```text
h_final : [768]
```

可以理解成：

> 模型对 `"中国 的 首都 是"` 这个前缀经过所有层加工后的表示。

但现在它仍然只是一个 **768 维内部表示**。

你还不能直接说：

```text
下一词 = 北京
```

于是需要 LM Head。

---

# 3. LM Head 本质上就是一个 Linear

MiniMind 里非常直接：

```python
self.lm_head = nn.Linear(
    self.config.hidden_size,
    self.config.vocab_size,
    bias=False
)
```

也就是：

```text
Linear(C, V)
```

当前源码默认**就是从 hidden size 映射到 vocabulary size。**

所以：

```text
[768]
↓
LM Head
↓
[6400]
```

这 6400 个数，就是：

```text
token 0 的分数
token 1 的分数
token 2 的分数
...
token 6399 的分数
```

也就是：

# logits

---

## 4. 举个极小词表示例

假设词表只有 5 个 token：

```text
0 = 我
1 = 北京
2 = 上海
3 = 中国
4 = 苹果
```

最终 hidden state：

```text
h = [C维]
```

送进：

```text
LM Head
```

输出：

```text
logits =

我      0.3
北京    8.6
上海    3.2
中国    1.4
苹果   -2.1
```

那么**很明显：**

```text
“北京”
```

**分数最高。**

**如果直接 greedy decoding：**

```text
argmax(logits)
```

就选择：

```text
北京
```

作为下一个 token。

---

# 5. 所以 LM Head 可以理解成“词表分类器”

这一点特别好理解。

普通分类任务：

```text
hidden state
↓
Linear
↓
10 classes
```

比如：

```text
猫
狗
鸟
...
```

LM 其实也非常像一个分类任务。

只不过类别不是：

```text
猫 / 狗 / 鸟
```

而是：

> **词表里的所有 token。**

所以如果词表大小：

```text
V = 6400
```

那**每一步本质就是：**

> **在 6400 个 token 中预测下一个是哪一个。**

因此 LM Head：

```text
C → V
```

本质上就是一个：

> **V 类分类头。**

---

# 6. 为什么叫 “LM Head”？

`LM`：

```text
Language Model
```

`Head`：

这里的意思类似：

> 接在主干网络最后，用来完成某个任务的输出模块。

例如一个 BERT 主干后面可以接：

```text
Classification Head
```

做分类。

某些 RL 模型后面可以接：

```text
Value Head
```

预测 value。

而语言模型后面接：

```text
LM Head
```

完成：

> **next-token prediction**

所以：

```text
Transformer
= 主干 / backbone

LM Head
= 语言模型任务的输出头
```

---

# 7. 它和 Attention Head 为什么都叫 Head？

只是英文里的两个不同用法，不要把它们混在一起。

### Attention Head

```text
Multi-Head Attention

head 0
head 1
...
head 7
```

是：

> Attention **内部并行的不同注意力通道。**

### LM Head

```text
Transformer 最后
↓
LM Head
```

是：

> 整个**模型的任务输出头。**

所以：

```text
Attention Head
≠
LM Head
```

它们没有“一对一对应”这种关系。

---

# 8. 联系我们刚才讲的自回归生成

假设现在已经有：

```text
我 喜欢 机器
```

模型生成：

```text
学习
```

这一步完整过程：

```text
“机器”位置 / 当前输入
↓
经过每层 Attention
  ├─ Q/K/V
  ├─ 多 Head Attention
  ├─ KV Cache
  └─ SwiGLU
↓
最终 hidden state
[768]
↓
LM Head
↓
logits
[6400]
↓
选择“学习”
```

然后把：

```text
学习
```

追加进去：

```text
我 喜欢 机器 学习
```

---

下一轮：

```text
“学习”
↓
Embedding
↓
所有 Transformer 层
↓
final hidden state
↓
LM Head
↓
6400 logits
↓
可能选出“。”
```

于是：

```text
我 喜欢 机器 学习 。
```

所以 **LM Head 就是连接“模型内部 hidden state”与“真正生成 token”之间的最后一道桥梁。**

---

# 9. 为什么每个位置都能经过 LM Head？

训练时不是只有最后一个位置。

假设：

```text
hidden_states.shape
=
[B,T,C]
```

直接通过：

```text
LM Head: C → V
```

Linear **会对每一个 `[b,t,:]` 独立应用。**

于是：

```text
[B,T,C]

↓

LM Head

↓

[B,T,V]
```

例如：

```text
[B,4,768]
↓
[B,4,6400]
```

因此：

```text
位置0 → 6400 logits
位置1 → 6400 logits
位置2 → 6400 logits
位置3 → 6400 logits
```

训练时**每个位置都用来预测下一个 token。**

---

# 10. 但推理为什么通常只拿最后一个位置？

假设 prompt：

```text
中国 的 首都 是
```

forward 以后：

```text
hidden states:

h0
h1
h2
h3
```

其中：

```text
h0 → 根据“中国”预测下一个
h1 → 根据“中国 的”预测下一个
h2 → 根据“中国 的 首都”预测下一个
h3 → 根据“中国 的 首都 是”预测下一个
```

我们现在真正关心的是：

> `"中国 的 首都 是"` 后面是什么？

所以**只需要：**

```text
h3
```

**经过 LM Head。**

**MiniMind 生成时也是取最后位置：**

```python
logits = outputs.logits[:, -1, :]
```

**再进行 temperature、top-k、top-p 等处理，最后选下一个 token。**

---

# 11. LM Head 本身通常没有 Softmax

这也是一个容易误解的地方。

LM Head **输出的是：**

```text
logits
```

**例如：**

```text
[1.3, 8.2, -0.4, 3.1, ...]
```

**而不是概率。**

**之后才根据用途处理：**

### 训练

```text
logits
↓
CrossEntropyLoss
```

**CrossEntropy 内部会处理 softmax/log-softmax 相关计算。**

### 推理

```text
logits
↓
temperature
↓
softmax
↓
top-k / top-p
↓
sampling
```

所以：

```text
LM Head
≠ softmax
```

更准确：

```text
LM Head
=
hidden state → logits
```

---

# 12. MiniMind 还有个很有意思的设计：LM Head 和 Embedding 共享参数

你前面知道：

```text
Embedding：

token id
↓
C维向量
```

而 **LM Head：**

```text
C维向量
↓
token logits
```

**它俩方向正好相反：**

```text
Embedding:

Token Space
V
↓
Hidden Space
C


LM Head:

Hidden Space
C
↓
Token Space
V
```

MiniMind **中默认：**

```python
tie_word_embeddings = True
```

**并且：**

```python
self.model.embed_tokens.weight = self.lm_head.weight
```

**也就是 weight tying。**

**你可以先粗略理解成：**

> 输入端“token **应该映射成什么向量**”和输出端“**这个 hidden vector 和哪个 token 最匹配”**，**共享同一套 token 表示参数。**

这样还**能减少一些参数。**

---

# 13. 你现在把整个模型分成三层理解最好

### 第一层：Embedding

```text
token
↓
hidden vector
```

回答：

> “把**离散 token 转成模型能算的连续向量**。”

---

### 第二层：Transformer

```text
hidden vectors
↓
Attention + SwiGLU + ...
↓
contextualized hidden vectors
```

回答：

> “结合上下文，把**这些表示加工得越来越有信息**。”

---

### 第三层：LM Head

```text
final hidden vector
↓
vocabulary logits
```

回答：

> “根据现在理解到的上下文，下一个 token 应该是谁？”

---

## 最后一张图把你目前学的东西全部串起来

```text
token ids
[B,T]
   │
   ↓
Embedding
   │
   ↓
[B,T,C]
   │
   ↓
┌──────────────────────┐
│ Transformer Layers   │
│                      │
│ RMSNorm              │
│ Q/K/V                │
│ Attention Heads      │
│ RoPE / GQA           │
│ KV Cache（推理）      │
│ SwiGLU               │
└──────────────────────┘
   │
   ↓
final hidden states
[B,T,C]
   │
   ↓
LM Head
Linear(C,V)
   │
   ↓
logits
[B,T,V]
   │
   ↓
选择最后一个位置
[B,V]
   │
   ↓
sampling / argmax
   │
   ↓
next token
   │
   └────────→ 下一轮生成
```

如果你之前看到的是 **7600 维 logits**，那只是你看的版本/配置里 `vocab_size=7600`；当前 MiniMind-3 主线源码**默认 `vocab_size=6400`。原理完全一样：**

> **LM Head 的输出维数永远等于词表大小 `V`，因为每一维就是一个候选 token 的 logit。**

# 融合的上下文信息到底是在h向量里面还是在k,v里面也有

都有，但要分清“**上下文融合发生在哪里**”和“**K/V 里面有没有上下文信息**”。

最核心的一句话是：

> **真正完成“把其他 token 信息融合进当前 token”的结果，落在新的 hidden state \(h\) 里；而 K、V 是由当前层输入的 hidden state 投影出来的，所以如果这个 hidden state 本身已经融合过前文，那么 K、V 当然也携带上下文信息。**

**关键在于“第几层”。**

---

# 1. 第一层时，K/V 基本还没有融合其他 token

假设：

```text
我  喜欢  机器  学习
```

E**mbedding 后：**

```text
h0_我
h0_喜欢
h0_机器
h0_学习
```

**这时候：**

```text
h0_机器
```

**主要还是 `"机器"` 自己的表示，还没经过 Attention 去读取 `"我 喜欢"`。**

第一层 Attention 先做：

$$
Q=W_Qh
$$

$$
K=W_Kh
$$

$$
V=W_Vh
$$

于是**第一层里：**

```text
K_机器
V_机器
```

**主要也是由 `"机器"` 自己当前的 hidden state 产生。**

然后**真正发生信息融合：**

```text
Q_机器
↓
和 我/喜欢/机器 的 K 比较
↓
得到 attention weights
↓
加权读取 我/喜欢/机器 的 V
↓
Attention Output
↓
加 residual
↓
新的 h_机器
```

所以：

> **这一层 Attention 的输出，才第一次真正把其他 token 的信息融合进 `"机器"` 的 hidden state。**

---

# 2. 到第二层，情况就变了

第一层结束后：

```text
h1_机器
```

已经**不是单纯的：**

```text
“机器”
```

**而是类似：**

```text
“机器 + 我喜欢机器 这个前缀的上下文信息”
```

**然后进入第二层。**

**第二层重新计算：**
$$
K^{(2)}_{\text{机器}}
=
W_K^{(2)}h^{(1)}_{\text{机器}}
$$

$$
V^{(2)}_{\text{机器}}
=
W_V^{(2)}h^{(1)}_{\text{机器}}
$$

因为：

```text
h1_机器
```

本身已经包含上下文了，

所以：

```text
K2_机器
V2_机器
```

当然也携带上下文信息。

---

# 3. 所以不是“K/V 永远只表示单个 token”

这是一个非常重要的修正。

第一层：

```text
Embedding
↓
h^0_t

主要是当前 token 自身信息
↓
K^0_t / V^0_t

主要基于当前 token 自身
```

但是第一层 Attention 后：

```text
h^1_t
=
已经融合前缀
```

第二层：

```text
h^1_t
↓
K^1_t / V^1_t

已经带有前缀信息
```

然后第二层 Attention 又继续融合：

```text
h^2_t
```

上下文表示进一步加工。

所以越来越往上：

```text
Embedding
↓
token 自身

Layer 0 Attention
↓
初步上下文化 h¹

Layer 1 K/V
↓
已经是上下文化的 K/V

Layer 1 Attention
↓
进一步上下文化 h²

Layer 2 K/V
↓
包含更复杂上下文信息

...
```

---

# 4. 那 h、K、V 的区别究竟是什么？

这个地方**最好不要理解成：**

```text
h = 上下文
K = token自己
V = token自己
```

**不是。**

**更准确是：**

### hidden state \(h\)

> **当前 token 在当前网络层中的完整内部表示。**

例如：

```text
h_t ∈ R^C
```

它是 Transformer 主干真正往下一层传递的东西。

---

### K

$$
K_t=W_Kh_t
$$

是**从 hidden state 中投影出的：**

> **“我应该怎样被别的 Query 匹配？”的表示。**

**所以 K 更偏向：**

```text
用于算 attention score
```

---

### V

$$
V_t=W_Vh_t
$$

是**从 hidden state 投影出的：**

> **“别人关注我之后，我应该提供什么信息？”的表示。**

**所以 V 更偏向：**

```text
实际被加权传递的信息
```

---

# 5. 一个很好理解的比喻

假设：

```text
h_t
```

是一份关于当前 token 的**完整档案**。

随着 Transformer 越来越深，**这份档案里已经不只是这个 token 自己，还写入了上下文。**

然后：

```text
h_t
├─ Wq → Q：我想查询什么
├─ Wk → K：别人怎么检索到我
└─ Wv → V：别人找到我以后我提供什么内容
```

所以：

> **Q/K/V 都来自 h，它们是 h 的不同功能性投影。**

**不是三份独立于 hidden state 的信息。**

---

# 6. 那 Attention 融合的信息最终为什么说“存进 h”？

因为 V **被加权以后：**

$$
o_t
=
\sum_j\alpha_{tj}V_j
$$

**得到：**

```text
attention output
```

然后经过 output projection：

```text
o_proj
```

再加 residual：

$$
h'_t
=
h_t+\operatorname{Attention}(h)_t
$$

所以其他 token 的信息最终写进：

```text
h'_t
```

然后这个：

```text
h'_t
```

继续经过 MLP，再进入下一层。

MiniMind 的 Block 也是：

```python
residual = hidden_states

hidden_states, _ = self.self_attn(
    self.input_layernorm(hidden_states),
    ...
)

hidden_states += residual
```

也就是 Attention 得到的信息最终重新并回 `hidden_states`。

---

# 7. KV Cache 存的 K/V 为什么也能代表历史上下文？

现在你应该就能理解了。

假设有 8 层 Transformer。

对于 `"机器"` 这个历史 token：

### Layer 0 cache

```text
K_machine^0
V_machine^0
```

来自：

```text
Embedding 后的 h
```

上下文程度比较低。

---

### Layer 1 cache

**来自第一层已经加工完的 hidden state：**

```text
h_machine^1
```

所以：

```text
K_machine^1
V_machine^1
```

**已经携带第一层融合后的上下文。**

---

### Layer 7 cache

来自经过前 7 层加工后的：

```text
h_machine^7
```

因此：

```text
K_machine^7
V_machine^7
```

已经是**很深的 contextual representation。**

所以 **KV Cache 并不是：**

> **“把每个 token 原始信息永久存起来”。**

而是：

> **每一层都缓存这个 token 在“进入该层 Attention 时”的 K/V 表示。**

---

# 8. 特别注意一个细节

某一层的 K/V：

```text
K_t^l
V_t^l
```

是在**这一层 Attention 发生之前**，由**这一层的输入 hidden state 生成的。**

所以**它包含：**

> **前面层已经融合好的信息**

**但不包含：**

> **当前这一层 Attention 刚刚新融合出来的信息。**

当前层**融合结果会进入：**

```text
h^{l+1}
```

**然后下一层再根据这个新 `h` 产生下一层的 K/V。**

可以写成：

$$
h^{(l)}
$$

↓

$$
Q^{(l)},K^{(l)},V^{(l)}
$$

↓

**Attention**

↓

$$
h^{(l+1)}
$$

↓

**下一层：**

$$
Q^{(l+1)},K^{(l+1)},V^{(l+1)}
$$

**这条链非常重要。**

---

# 9. 所以最终回答你的问题

### “融合后的上下文信息在 h 里面吗？”

**是，最主要、完整地保存在 hidden state 中。**

因为 hidden state 是层与层之间**真正传递的表示。**

### “K/V 里面有上下文信息吗？”

**也有。**

因为 **K/V 是从 hidden state 投影出来的。**

但具体**有多少上下文信息取决于层数：**

```text
第1层 K/V
→ 基本基于 embedding / 当前 token

越深层 K/V
→ 来自已经高度上下文化的 hidden state
→ 因而也携带丰富上下文
```

### 三者最准确的关系

```text
h
= 当前层完整的 token 表示

K = Wk h
= 从 h 中提取出的“供别人匹配我的表示”

V = Wv h
= 从 h 中提取出的“供别人读取我的信息”

Attention
= 用 Q/K 决定从哪些 V 拿多少

Attention结果
= 再写回新的 h
```

所以整个 Transformer 本质上就是不断循环：

$$
\boxed{
h
\rightarrow Q,K,V
\rightarrow Attention
\rightarrow 新h
\rightarrow 新Q,K,V
\rightarrow Attention
\rightarrow \cdots
}
$$

你现在如果把这一条理解透了，QKV 和 hidden state 的关系就基本彻底通了。

# 详细地列一下一整个序列经过处理的详细流程图,里面的参数都需要详细标出,什么h,QKV,W_Q,head,KV cache等等我们遇到的学的参数都必须在流程图中标出来,我要清楚的理解他们都是在什么流程中起什么作用的

![ChatGPT_Image_Sep_15,_2026,_03_37_53_PM](./minimind%E7%96%91%E9%97%AE2/ChatGPT_Image_Sep_15,_2026,_03_37_53_PM.png)

# 为什么训练的时候又不使用KV cache了,然后每个层hidden state都要记录存下来去看去计算呢

这两个现象其实分别对应两件完全不同的事：

* **KV Cache**：是为了“推理时别重复算历史 token”。
* **训练时保存 hidden state / 中间激活**：是为了“反向传播算梯度”。

它们看起来都是“存东西”，但目的完全不一样。

---

# 1. 为什么推理要 KV Cache，训练却通常不用？

先看推理。

假设已经有：

```text
我 喜欢 机器
```

下一步只新来了一个 token：

```text
学习
```

如果**没有 KV Cache，你每生成一个新 token，都得重新算：**

```text
我
我 喜欢
我 喜欢 机器
我 喜欢 机器 学习
...
```

**历史部分大量重复。**

**所以推理时缓存：**

```text
过去每层每个 token 的 K
过去每层每个 token 的 V
```

**下一步只算新 token 的：**

```text
Q_new
K_new
V_new
```

然后：

```text
Q_new × K_cache
→ attention weights
→ × V_cache
```

这样就省掉了历史 token 的重复计算。

---

# 2. 但是训练不是“一次进来一个 token”

**训练时通常直接给模型整段：**

```text
我 喜欢 机器 学习 。
```

**假设：**

```text
T = 5
```

**一次 forward 就输入：**

```text
[B, T]
```

**然后 Attention 一次性计算：**

```text
Q : [B,H,T,D]
K : [B,H,T,D]
V : [B,H,T,D]
```

直接算：

$$
QK^T
$$

得到：

```text
[B,H,T,T]
```

也就是说：

> **所有位置的 attention 一次并行算完。**

**只需要 causal mask 保证：**

```text
位置0 看 0
位置1 看 0..1
位置2 看 0..2
...
```

所以**训练时根本没有：**

```text
先算 token0
再算 token1
再算 token2
```

**这种重复计算问题。**

因此 **KV Cache 没什么可省的。**

---

# 3. 这是训练和推理最大的区别之一

## 训练：并行

例如：

```text
我 喜欢 机器 学习
```

一次性：

```text
h0 h1 h2 h3
↓
同时产生所有 Q/K/V
↓
同时算所有位置 Attention
↓
同时预测：

我       → 喜欢
我喜欢   → 机器
我喜欢机器 → 学习
...
```

**这叫：**

> **teacher forcing / parallel next-token training**

---

## 推理：串行

**推理必须：**

```text
先生成 token4
↓
才能知道 token4 是什么
↓
才能生成 token5
↓
才能知道 token5
```

**所以只能：**

```text
一个 token 一个 token 地 decode
```

**于是才需要 KV Cache。**

---

# 4. 如果训练也强行用 KV Cache 会怎样？

假设训练序列：

```text
我 喜欢 机器 学习
```

你非要像推理一样：

```text
先 “我”
存 KV

再 “喜欢”
存 KV

再 “机器”
存 KV

再 “学习”
```

当然理论上能算。

但是你**把本来 GPU 特别擅长的：**

```text
整段矩阵并行计算
```

**变成了：**

```text
token-by-token 串行计算
```

**会慢很多。**

**所以训练时通常选择：**

```text
整段一次算完
```

**而不是 cache。**

---

# 5. 而且训练还有一个更麻烦的问题：要反向传播

**推理时：**

```text
K_cache
V_cache
```

**只需要用于 forward。**

**没有：**

```text
loss.backward()
```

**所以：**

```text
过去算完
→ 存结果
→ 以后拿来用
```

**很简单。**

**但训练中：**

```text
K
V
```

**也是计算图的一部分。**

因为：

$$
K = hW_K
$$

$$
V = hW_V
$$

最终：

```text
loss
```

**需要对：**

```text
W_K
W_V
h
```

**求梯度。**

**如果你把不同 token 分开 cache 再串起来，计算图会跨很多步连接起来，反而很麻烦，而且没有速度优势。**

---

# 6. 那训练为什么又要保存 hidden state？

这个就是另外一回事了。

你可能想到：

> 前向传播已经算完 Layer 0 的 hidden state 了，为什么还留着？直接给 Layer 1 不就行了吗？

如果**只有 forward，确实可以：**

```text
Layer0 output
↓
Layer1
↓
Layer2
```

**用完就扔。**

**但训练还需要：**

# **backward**

**也就是：**

```python
loss.backward()
```

---

# **7. 反向传播为什么需要以前的 hidden state？**

**看最简单的：**

$$
y=xW
$$

**假设最后 loss 对 `y` 的梯度：**

$$
\frac{\partial L}{\partial y}
$$

**已经知道。**

现在要算：

$$
\frac{\partial L}{\partial W}
$$

根据**链式法则：**

$$
\frac{\partial L}{\partial W}
=
x^T
\frac{\partial L}{\partial y}
$$

注意这里需要什么？

需要：

```text
x
```

**也就是当时 forward 的输入。**

**所以如果 forward 时：**

```text
x
```

**已经彻底扔掉了，**

**backward 到这里：**

```text
我连 W 的梯度怎么算都不知道。
```

---

# 8. Transformer 里也是同一个道理

例如某层：

```text
h
↓
Q = h W_Q
```

反向传播时需要：

$$
\frac{\partial L}{\partial W_Q}
$$

而：

$$
\frac{\partial L}{\partial W_Q}
=
h^T
\frac{\partial L}{\partial Q}
$$

所以需要保存当时的：

```text
h
```

同样：

```text
K = h W_K
V = h W_V
```

要算：

```text
W_K
W_V
```

的**梯度，也需要这个 `h`。**

---

# 9. Attention 本身也需要保存一些中间结果

Forward：

$$
S=\frac{QK^T}{\sqrt D}
$$

$$
A=softmax(S)
$$

$$
O=AV
$$

Backward 时，**要分别求：**

```text
dV
dA
dQ
dK
```

**就需要 forward 时的一些中间结果**，例如：

```text
Q
K
V
softmax后的 attention weights
或者足够重建它们的信息
```

所以**训练显存才会这么大。**

---

# 10. MLP 也是一样

MiniMind 的 **SwiGLU：**

$$
y=W_{down}
(
SiLU(W_{gate}h)
\odot
W_{up}h
)
$$

**forward 会产生：**

```text
h
↓
gate_proj
↓
中间 activation

h
↓
up_proj
↓
中间 activation

两者相乘
↓
down_proj
```

**反向传播需要对：**

```text
W_gate
W_up
W_down
```

**分别求梯度。**

**所以很多中间 activation 都需要保留。**

---

# 11. 所以训练真正“存”的东西通常叫 Activation

你之前已经碰到过这个词。

训练内存大致包括：

```text
模型参数 Parameters
+
梯度 Gradients
+
优化器状态 Optimizer States
+
中间激活 Activations
```

这里：

```text
hidden states
Q
K
V
MLP中间结果
某些norm结果
attention中间结果
```

都属于广义的：

> **activation / forward intermediate**

---

# 12. 但不是“每层所有东西全部永久保存”

这个也要纠正一下。

**PyTorch Autograd 会保存：**

> **反向传播需要的那些张量。**

**并不是简单粗暴：**

```text
Layer0所有东西全存
Layer1所有东西全存
Layer2所有东西全存
```

**具体某个算子 backward 需要什么，PyTorch 就保存对应信息。**

但对 Transformer 来说，**确实会留下大量中间 activation，所以显存很大。**

---

# 13. 这就是为什么有 Gradient Checkpointing

这东西你以后一定会遇到。

正常训练：

```text
Forward：

Layer0
→ 保存中间结果

Layer1
→ 保存中间结果

Layer2
→ 保存中间结果

...

Backward：
直接拿保存结果算
```

**优点：**

```text
backward 快
```

**缺点：**

```text
显存大
```

---

**Gradient Checkpointing：**

```text
Forward：

某些中间结果不保存
只保存少数 checkpoint
```

到 backward 需要时：

```text
重新 forward 算一次
```

所以变成：

> **用额外计算换显存。**

也就是：

```text
少存 activation
↓
显存下降

但 backward 时重算
↓
训练变慢
```

---

# 14. 这里特别容易把“KV Cache”和“训练 Activation”混淆

你可以用这个表彻底分开：

|                           | KV Cache                 | 训练 Activation                         |
| ------------------------- | ------------------------ | --------------------------------------- |
| 使用场景                  | 推理生成                 | 训练                                    |
| 目的                      | **避免重新算历史 token** | **为 backward 求梯度**                  |
| 保存什么                  | 每层历史 token 的 K/V    | hidden、QKV、MLP 中间结果等需要反传的量 |
| 是否为了反向传播          | ❌                        | ✅                                       |
| 是否跨生成 token 长期保留 | ✅                        | ❌ 一次训练 step 后通常释放              |
| 典型生命周期              | **整次生成**             | **一次 forward + backward**             |

---

# 15. 再看一遍训练完整流程

假设：

```text
input_ids
[B,T]
```

## Forward

```text
Embedding
↓
h⁰ [B,T,C]
↓
Layer0
    RMSNorm
    ↓
    Q⁰,K⁰,V⁰
    ↓
    Attention
    ↓
    h¹
    ↓
    SwiGLU
↓
Layer1
    Q¹,K¹,V¹
    ...
↓
...
↓
hᴸ
↓
LM Head
↓
logits [B,T,V]
↓
CrossEntropy
↓
loss
```

同时 Autograd 保存 backward 需要的中间信息。

---

## Backward

从：

```text
loss
```

反方向走：

```text
loss
↑
LM Head
↑
Layer L
↑
Layer L-1
↑
...
↑
Embedding
```

逐层计算：

```text
dW_lm_head
dW_Q
dW_K
dW_V
dW_O
dW_gate
dW_up
dW_down
...
```

所以需要 forward 的 activation。

---

# 16. MiniMind 训练也是整段 sequence 一次进去

它的**预训练代码大致就是：**

```python
res = model(input_ids, labels=labels)
loss = res.loss + res.aux_loss
loss.backward()
```

**而不是：**

```python
for token in sequence:
    model(token, past_kv)
```

数据集**也是直接准备完整：**

```text
input_ids
labels
```

**然后一次送进去。**

**所以训练阶段本质是：**

> **整段并行计算 + causal mask + backward。**

---

# 17. 你问“每层 hidden state 都要记录下来去看去计算”可以分成两个意思

### 前向传播时

`h^l` 是下一层的输入：

$$
h^{l+1}=Block_l(h^l)
$$

**所以当然必须至少“传下去”。**

---

### 反向传播时

还**需要它来算：**

$$
\frac{\partial L}{\partial W}
$$

**所以通常还要临时保存。**

**这就是为什么训练过程中 hidden states 不只是“路过一下”。**

---

# 18. 最后把最核心的区别压成一句话

## 推理为什么存 KV？

因为：

> **过去 token 不会再变，所以把它们算好的 K/V 存起来，防止下一 token 又重复计算。**

## 训练为什么不存 KV Cache？

因为：

> **整个序列的所有 Q/K/V 本来就一次并行算完，没有历史重复计算的问题，而且还需要完整计算图做反向传播。**

## 训练为什么保存 hidden / activation？

因为：

> **反向传播算每层参数梯度时，需要 forward 当时的输入和中间结果。**

你可以记成两个完全不同的“存”：

$$
\boxed{\text{KV Cache = 为了以后别重算}}
$$

$$
\boxed{\text{Activation = 为了待会能反向传播}}
$$

这两个区分一旦建立起来，后面你再看 **显存占用、gradient checkpointing、FlashAttention、KV cache、prefill/decode** 就不会混在一起了。
