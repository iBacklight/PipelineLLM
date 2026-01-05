# LLM Background : Transformer 模型深度解析与代码实现

**基于 PyTorch 的从零实现**

## 1. 简介

Transformer 是现代自然语言处理（NLP）和大型语言模型（LLMs）的基石。不同于 RNN 或 LSTM，它完全依赖于 **Attention Mechanism（注意力机制）** 来捕捉序列中的长距离依赖关系。

------

## 2. 核心组件：多头注意力机制 (Multi-Head Attention)

这是 Transformer 中最重要的部分。它允许模型同时关注输入序列的不同部分。

### 2.1 数学原理

单头注意力的计算公式为：
$$
Attention(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$
其中：

- **Q (Query):** 查询向量
- **K (Key):** 键向量
- **V (Value):** 值向量
- **$d_k$:** 缩放因子，用于防止点积结果过大导致 Softmax 梯度消失。

### 2.2 代码实现分析 (`class MultiHeadAttention`)

在我们的代码中，`MultiHeadAttention` 类实现了这一逻辑。

**关键步骤拆解：**

1. 线性投影 (Linear Projections):

   代码首先将输入 $X$ 投影为 Q, K, V。

   Python

   ```python
   # d_model 是输入维度，d_model 也通常是输出维度
   self.q_proj = nn.Linear(d_model, d_model, bias=True)
   self.k_proj = nn.Linear(d_model, d_model, bias=True)
   self.v_proj = nn.Linear(d_model, d_model, bias=True)
   ```

2. 多头切分 (Splitting Heads):

   为了实现“多头”，代码将特征维度切分为 n_head 份。

   Python

   ```python
   # 形状变换: [Batch, Time, Features] -> [Batch, Time, Heads, Head_Dim]
   q = q.view(B, T, self.n_head, d_q // self.n_head)
   # 为了后续矩阵乘法，需要交换维度，变成 [Batch, Heads, Time, Head_Dim]
   q = q.transpose(1, 2)
   ```

3. 缩放点积注意力 (Scaled Dot-Product Attention):

   这是计算的核心。

   Python

   ```python
   # 计算 Q * K^T / sqrt(d_k)
   # k.transpose(2, 3) 将最后两个维度交换，以便进行矩阵乘法
   attn = q @ k.transpose(2, 3) / math.sqrt(self.d_k)
   ```

4. 掩码机制 (Masking):

   代码中处理了两种 Mask：

   - **Causal Mask (因果掩码/上三角掩码):** 用于 Decoder，防止模型看到未来的信息。

     Python

     ```python
     # 创建上三角矩阵 (diagonal=1)，将上方元素设为负无穷
     causal_mask = torch.triu(torch.ones(T, T, ...), diagonal=1)
     attn = attn.masked_fill(causal_mask, float('-inf'))
     ```

     ```python
       * **Padding Mask:** 忽略输入中的 Padding Token。
     ```

5. **Softmax 与输出:**

   Python

   ```python
   attn_weights = self.softmax(attn) # 归一化得到权重
   attn = attn_weights @ v           # 权重加权求和
   ```

------

## 3. 编码器 (The Encoder)

Encoder 负责理解输入序列（如翻译任务中的源语言）。

### 3.1 编码器层 (`class TransformerEncoderLayer`)

每个层包含两个子层，每个子层后接残差连接（Residual Connection）和层归一化（LayerNorm）。

**结构流程：**

1. **Self-Attention:** `x + dropout(attention(x))`
2. **LayerNorm:** `norm1(x)`
3. **Feed Forward (FFN):** `x + dropout(ffn(x))`
4. **LayerNorm:** `norm2(x)`

代码细节（Post-LN）：代码使用的是 Post-LN 结构（先相加，后归一化），这是原始 Transformer 论文的做法。

```python
# Self-Attention + Residual + Norm
attn = self.attn(x, x, x, None, key_padding_mask)
x = x + self.dropout(attn)
x = self.norm1(x) 

# FFN + Residual + Norm
x = x + self.dropout(self.ffn(x))
x = self.norm2(x)
```

> **注意：** 现代模型（如 LLaMA, GPT-3）常使用 **Pre-LN**（先归一化，后计算），训练更稳定。在这里的代码中注释掉的部分展示了 Pre-LN 的写法。

------

## 4. 解码器 (The Decoder) 与代码中的特殊情况

Decoder 负责利用 Encoder 的输出生成目标序列。

### 4.1 结构升级 (`class TransformerDecoderLayer`)

我们使用了标准的三个子层结构，修复了之前缺失 Cross-Attention 的问题：

1. **Masked Self-Attention:**

   - **作用：** 让 Decoder 关注已生成的 Token。
   - **参数：** `tgt_mask=True` (启用因果掩码)。

   ```python
   self_attn_out = self.self_attn(x, x, x, tgt_mask, tgt_key_padding_mask)
   ```

2. **Cross-Attention (交叉注意力):**

   - **作用：** 让 Decoder 将当前的 Query 与 Encoder 的输出（Memory）进行交互，提取源句子的信息。
   - **关键代码：** `query` 来自 Decoder 的 `x`，而 `key` 和 `value` 来自 `memory`。

   ```Python
   # x 是 Decoder 当前状态，memory 是 Encoder 输出
   cross_attn_out = self.cross_attn(x, memory, memory, None, memory_key_padding_mask)
   ```

3. **Feed Forward Network (FFN):**

   - 最后经过全连接层和激活函数（GELU）。

### 4.2 级联调用

在 `TransformerDecoder` 中，数据流传递了 `memory`：

```Python
for layer in self.layers:
    output = layer(output, memory, ...) # memory 被传入每一层
```

------

## 5. 整体模型组装 (`class TransformerModel`)

模型将 Embedding、Encoder 和 Decoder 串联起来。

### 5.1 位置编码 (Positional Encoding)

由于 Transformer 并行处理所有 Token，无法像 RNN 那样天然捕捉顺序，因此必须注入位置信息。 代码使用了正弦/余弦位置编码：

Python

```
pe[:, 0::2] = torch.sin(position * div_term) # 偶数维度 sin
pe[:, 1::2] = torch.cos(position * div_term) # 奇数维度 cos
```

### 5.2 前向传播流程 (`forward`)

1. **Embedding:** `Source` 和 `Target` ID 转换为向量并加上位置编码。
2. **Encoding:** 源序列通过 Encoder，输出 `memory`。
3. **Decoding:** 目标序列 + `memory` 通过 Decoder。
   - 注意：这里正确传递了 `src_key_padding_mask` 给 Decoder 的 Cross-Attention，确保 Decoder 不会关注到源句子的 Pad Token。
4. **Projection:** `Decoder Output` -> `Vocab Size`。

------

## 5. 位置编码 (Positional Encoding)

由于 Attention 机制本身没有顺序概念（并行计算），我们需要显式地注入位置信息。

### 5.1 正弦/余弦编码

代码中使用了经典的 Sinusoidal Positional Encoding：
$$
PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}}) \\ PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})
$$
**代码对应：**

```Python
div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
```

这为每个 Token 的 Embedding 加上了一个独特的位置指纹。

------

## 6. 完整模型流 (`class TransformerModel`)

最后，模型将所有部分组装起来。

### 数据流向

1. **Input (Source & Target):** 输入 Token ID。
2. **Embedding:** `token_embedding(src) * sqrt(d_model)`.
3. **Add Positional Encoding:** 加上位置向量。
4. **Encoder:** 处理 Source 序列，输出上下文矩阵（在本代码中被计算了但未被 Decoder 使用）。
5. **Decoder:** 处理 Target 序列（使用 Causal Mask 防止作弊）。
6. **Output Projection:** `nn.Linear(d_model, vocab_size)` 将向量映射回词表大小。

```Python
# Forward Pass
memory = self.encoder(src_emb, src_key_padding_mask=...)
decoder_output = self.decoder(memory, tgt_emb, tgt_mask=True, ...)
output = self.output_projection(decoder_output)
```