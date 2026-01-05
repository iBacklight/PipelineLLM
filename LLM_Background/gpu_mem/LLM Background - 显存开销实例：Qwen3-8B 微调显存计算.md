# 显存开销实例：QWEN3 8B 微调计算

[TOC]

<br>

## 0. TL;DR

> **训练/微调 Qwen3-8B，显存主要分两块使用：**
>
> 1. **底座显存（和 batch、序列无关）**：模型参数 + 梯度 + 优化器状态。
> 2. **激活显存（和 batch、序列强相关）**：前向过程产生的中间结果，用于反向传播。

下面我们结构它的内容，使得结构更直观一点：先讲“底座”，再讲“激活”，最后给一个“速算模板”（基于QWEN3-8B）。

> 估算时我们假设：
>
> - Qwen3-8B ≈ **8.2B 参数**；
> - 隐层维度 $d \approx 4096$，层数 $L \approx 36$，FFN 维度 $d_{ff} \approx 4d$（其实 Qwen3 用的是 3.x 倍，这里用 4 倍便于心算，结果会稍微偏保守一点）。

计算的公式总结统一写成：

$$
\boxed{
\text{Total}(B,S) \approx \text{Base} + \text{Act}(B,S) + \text{Overhead}
}
$$

结果大致如下：

| Case       | 配置说明                                                     | 参数相关显存 Base（GiB） | 是否开 ckpt | 激活显存 Act（GiB） | 预留 Overhead（GiB） | 总显存估算 Total ≈（GiB） | 80GB 单卡可行性                               |
| ---------- | ------------------------------------------------------------ | ------------------------ | ----------- | ------------------- | -------------------- | ------------------------- | --------------------------------------------- |
| **B-no**   | 经典 FP16 混合精度：FP16 权重 + **FP32 主权重** + FP32 m/v（16 B/param） | ≈ **122.5**              | ❌           | ≈ 12.4              | ≈ 2                  | **≈ 136.9**               | ❌ 远超 80GB，只能多卡/ZeRO/offload            |
| **B-ckpt** | 同上，但开梯度检查点                                         | ≈ 122.5                  | ✅           | ≈ 1.1               | ≈ 2                  | **≈ 125.6**               | ❌ 仍远超 80GB                                 |
| **C-no**   | 现代 **BF16 训练**：bf16 权重+梯度，FP32 m/v，**无 FP32 主权重**（12 B/param） | ≈ **91.8**               | ❌           | ≈ 12.4              | ≈ 2                  | **≈ 106.2**               | ❌ 仍大于 80GB                                 |
| **C-ckpt** | 同上，但开梯度检查点                                         | ≈ 91.8                   | ✅           | ≈ 1.1               | ≈ 2                  | **≈ 94.9**                | ⚠️ 依然略高于 80GB，需再配 ZeRO/offload 才现实 |
| **D-no**   | **BF16 + 8bit Adam**：bf16 权重+梯度 + 8bit 优化器状态（约 7 B/param） | ≈ **53.5**               | ❌           | ≈ 12.4              | ≈ 2                  | **≈ 67.9**                | ✅ 勉强能在 80GB 上跑（B=1,S=2048）            |
| **D-ckpt** | BF16 + 8bit Adam + 梯度检查点                                | ≈ 53.5                   | ✅           | ≈ 1.1               | ≈ 2                  | **≈ 56.6**                | ✅ 80GB 上比较从容，还有二十几 GB 缓冲         |

<br>

## 1. 底座显存：只和“参数个数 N”有关

把参数量记为 $N$（Qwen3-8B 这里 $N \approx 8.2\times10^9$）。这里基于不同的精度选择，我们分为Case A / B / C / D ，但是case A为FP32全量训练几乎已经没有公司在使用，所以我们只关注BCD。同时在这一节中，我们只看「参数相关」显存（不算激活），用“每个参数多少字节（Bytes/param）”来比较：

| Case | 名称                         | 权重 dtype  | 是否有 FP32 主权重 | 优化器状态（m,v） | 每参数显存（约） |
| ---- | ---------------------------- | ----------- | ------------------ | ----------------- | ---------------- |
| B    | 经典 FP16 混合精度 + masterW | 前向用 FP16 | **有** FP32 副本   | FP32 (m,v)        | ≈ 16 B/param     |
| C    | 现代 BF16 训练               | 权重 BF16   | **无** 单独 master | FP32 (m,v)        | ≈ 12 B/param     |
| D    | BF16 + 8-bit 优化器          | 权重 BF16   | 无                 | int8（+scale 等） | ≈ 7 B/param      |

- **B**：算力快、激活省，但**参数本身几乎没省显存（16B/param 和纯 FP32 差不多）**。
- **C**：现代主流 baseline，参数显存比 FP32 少约 25%（16→12B/param），数值稳定。
- **D**：在 C 的基础上再压缩优化器状态，**显存最省**，代价是优化器更复杂，更新稍慢一点。

注意，FP32 副本（master weights）本来就是为了解决 FP16训练数值不稳定设计的历史产物。BF16 因为动态范围和 FP32 一样大，很多现代实践就不再维护这份 FP32 副本，只保留 FP32 的优化器状态（m、v）就够稳定了。



## 2. Case B：经典 FP16 混合精度（有 FP32 主权重，传统方法，不常用）

### 2.1 它实际存了什么？

典型老式 FP16 mixed-precision 做法（NVIDIA 早期教程是这种），每个类型的参数量均为*N*，但是由于精度不同，实际显存开销也不相同：

- 前向 / 反向：
  - `W_fp16`：用于计算的权重（2B/param）
  - `grad_fp16`：用于 backward 的梯度（2B/param）
- 优化器内部：
  - `master_W_fp32`：**FP32 主权重**（4B/param）
  - `m_fp32`：Adam 一阶动量（4B/param）
  - `v_fp32`：Adam 二阶动量（4B/param）

总共：

$$
2 + 2 + 4 + 4 + 4 = 16\ \text{B/param}
$$

所以参数这一块显存 **并没有比纯 FP32（16B/param）少**，只是换了一种分配方式。

### 2.2 更新流程直观感受一下

一轮训练大概是：

1. 用 `W_fp16` 做前向，得到 loss；
2. backward 得到 `grad_fp16`；
3. 把 `grad_fp16` cast 到 FP32，和 `master_W_fp32`、m、v 一起算 Adam 更新；
4. 更新后的 FP32 主权重再 cast 回 FP16，写回 `W_fp16`，供下次 forward 用。

**要点**：

- FP16 权重 + FP32 主权重是一一对应的两份拷贝。
- 稳定性来自全部更新在 FP32 世界完成。

### 2.3 什么时候会/曾经会用这个 Case？

- **硬件不支持 BF16，只支持 FP16 Tensor Core**（如 V100、T4 早期）
- 想用 FP16 提升吞吐和节省激活显存，但又怕 FP16 直接更新不稳定
- 现在新项目里不太推荐作为首选，更多是历史兼容。

**代码层面（早期典型写法的示意）**：

```python
model.half()  # 把权重转成 fp16
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scaler = torch.cuda.amp.GradScaler()

for x, y in dataloader:
    optimizer.zero_grad()
    with torch.cuda.amp.autocast(dtype=torch.float16):
        loss = model(x).loss
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

在一些手写混合精度例子里，还会显式维护 `master_params` 列表，用 FP32 存一份副本。现代 PyTorch 的实现会在 `optimizer.state` 里存 FP32 的状态，这里主要是概念区分。


## 3. Case C：现代 BF16 训练（无单独 FP32 主权重）

### 3.1 它实际存了什么？

在有 BF16 支持的 GPU（A100/H100/RTX40 等）上：

- BF16 的指数位和 FP32 一样 → 动态范围足够；
- 通常做法：**权重和梯度都用 BF16，优化器状态仍用 FP32**，但不再额外存一份 FP32 主权重。

也就是：

- `W_bf16`：2 B
- `grad_bf16`：2 B
- `m_fp32`：4 B
- `v_fp32`：4 B
- **不再有单独的 `master_W_fp32`**

合计：

$$
2 + 2 + 4 + 4 = 12\ \text{B/param}
$$

相比 Case B/纯 FP32 的 16 B/param，确实省了约 25%。

### 3.2 更新流程

一轮训练：

1. 前向：用 `W_bf16` 在 BF16 精度下算 loss；
2. backward：算出 `grad_bf16`；
3. 优化器中通常会把梯度临时转为 FP32，与 `m_fp32`、`v_fp32` 计算更新；
4. 更新后的 FP32 结果再 cast 回 BF16 写入 `W_bf16`（有的实现直接在 BF16 上更新，但核心思路类似）。

**关键差异**：

- 不再单独维护一份「完全的 FP32 主权重副本」，只保留 BF16 权重 + FP32 动量。

### 3.3 什么时候选这个 Case？（推荐的默认选项）

- **硬件支持 BF16（Ampere / Hopper / RTX40 等）**
- 模型规模中等偏大，但你有 40–80GB 显存
- 希望：
  - 训练稳定可靠；
  - 显存比纯 FP32 / FP16+master 明显省一截；
  - 代码简单，不想引入 8bit 优化器的新超参。

**代码层面（PyTorch 原生）**：

```python
model.to(torch.bfloat16)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for x, y in dataloader:
    optimizer.zero_grad()
    # 你可以选择直接用 bf16（前向/反向都 bf16）
    out = model(x.to(torch.bfloat16))
    loss = out.loss
    loss.backward()
    optimizer.step()
```

更常见的是使用 `autocast`，在某些 op 使用 bf16：

```python
scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=False)  # bf16 通常不需要 scaler

for x, y in dataloader:
    optimizer.zero_grad()
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        loss = model(x).loss
    loss.backward()
    optimizer.step()
```

**HuggingFace Transformers 中**：

```python
training_args = TrainingArguments(
    ...,
    bf16=True,   # 关键开关
    fp16=False,  # 一般不用 fp16 了
)
```

DeepSpeed 中则在 config 里写 `bf16.enabled: true`，不再配置 fp16。



## 4. Case D：BF16 + 8-bit 优化器（显存优先）

### 4.1 它实际存了什么？

在 Case C 的基础上，把 FP32 的 `m`, `v` 换成 **8bit 表示**（例如 bitsandbytes 的 Adam8bit / PagedAdamW-8bit）：

- `W_bf16`：2 B
- `grad_bf16`：2 B
- 量化后的优化器状态（m, v）：约 3 B/param（实际取决于实现，有 scale/元数据）

合计 ≈ **7 B/param**：

$$
2 + 2 + 3 \approx 7\ \text{B/param}
$$

相比 Case C 的 12 B/param 又省了 ~40%，**对于 8B/14B/32B 这类模型非常显著**。

### 4.2 训练/更新过程上的差异

逻辑上仍然是「在高精度域计算更新」，只是：

- m、v 不再蒙在 FP32 里，而是存为 int8 + 每片的 scale；
- 每次更新时：
  1. 从 int8+scale 恢复一个近似的 FP32 m、v；
  2. 用它们算出更新；
  3. 再把新的 m、v 重新量化回 int8 存储。

因此：

- **显存更省**，但：
  - 更新时有一些**额外的量化/反量化**开销；
  - 有少量近似误差，理论上可能**稍微影响收敛/最终精度**（通常可接受）。

### 4.3 什么时候选这个 Case？

- **单卡显存吃紧**，但你又想：
  - 微调更大的模型（比如 14B/32B 全参）
  - 用更长的序列 / 更大的 batch
- 已经在用 BF16（Case C），但发现：
  - 显存仍然顶到 80GB 上限；
  - 不想上 ZeRO-3/offload 太复杂的方案。

简单说：

> **在当前 Qwen3-8B/14B/H100 的场景下，Case D 是「单卡继续抠显存」时非常现实的选择**。

### 4.4 代码层面怎么开 8bit 优化器？

**PyTorch + bitsandbytes（全参微调）**：

```python
from bitsandbytes.optim import Adam8bit

model.to(torch.bfloat16)
optimizer = Adam8bit(model.parameters(), lr=1e-4)
```

**HuggingFace Transformers（全参 finetune）**：

有两种用法，容易混淆：

1. **只用 8bit 优化器，不量化权重**：

   ```python
   training_args = TrainingArguments(
       ...,
       bf16=True,
       optim="adamw_bnb_8bit",  # 或 paged_adamw_8bit 等
   )
   ```

   配置 `optim` 就是告诉 Trainer 使用 bitsandbytes 的 8bit AdamW，**权重本身仍然是 bf16**。

2. **QLoRA 那种“4bit 权重 + LoRA + 8bit 优化器”的组合**就不展开了，那是另外一种优化策略，想请可以见SFT微调实战章节。

**DeepSpeed** 侧如果要用 8bit 优化器，一般是：

- 在 DeepSpeed config 里关闭 offload/zero 的 optimizer 管理；
- 自己在 Python 里把 `optimizer` 换成 bnb 的 `Adam8bit`；
- 然后让 DeepSpeed 只管理参数/梯度 sharding（需要检查兼容性，稍微复杂一点）。



## 5. 激活显存：和 B、S、L、d 强相关

### 5.0 激活显存介绍

“激活”就是前向传播中间算出来的隐藏状态（hidden states），要在反向时用，所以会占显存。

这块和：

- 层数 $L$ （ Transformer 的block 数 / 层数）
- 隐层维度 $d$
- FFN 维度 $d_{ff}$
- 微批大小（per-GPU batch）$B$
- 序列长度 $S$

成正比。这里需要注意的是，上面的**Case B/C/D 这些只是改「参数 + 优化器状态」怎么存的，**只要不更改激活的精度和梯度检查点策略，**激活显存几乎不受影响**。唯一的例外是如果有意修改了激活精度策略，例如：

- Case B 下用 `autocast(fp16)`，
- Case C 下用 `autocast(bf16)`，
- Case D 下某些框架为了安全把一部分激活保持在 fp32，

那会有一些差异，但：

- fp16 / bf16 在显存上都是 **2 Bytes/元素**，
- 只有变成 fp32 激活才会翻倍到 4 Bytes/元素，通常是个别 op，而不是全局。

为了估算，不需要精确到每一行代码，只要记住一个经验公式就够了。下面的小节是不同情况下的经验公式。

### 5.1 不开梯度检查点（不重算）

每一层、每个 token 需要保留大概这么多激活元素：

$$
\text{元素数} \approx 6d + 4d_{ff}
$$

- **6d** ：来自 self-attention 分支（Q、K、V、LN 输入输出、输出向量$W_o$等）。
- **4d_{ff}**：来自门控 MLP（SwiGLU / GeGLU）的几个中间量（上投影两个分支 + 激活后输出等）。

因为我们用 bf16，一般假设：

> **每个激活元素 ≈ 2 Bytes**

所以**每层每 token 激活字节数**：

$$
\text{Bytes}_{\text{per-layer-per-token,no-ckpt}} \approx (6d + 4d_{ff}) \times 2
$$


### 5.2 开“按层梯度检查点”（重算整层）

如果你对某一层开启 **block-wise gradient checkpointing**：

- 不再长期保存层内所有中间量（Q、K、V、MLP 中间态等），
- 只保留**层的输入/少量边界向量**，反向时候再利用这些边界**重算**层内计算。

这时每层每个 token 需要保留的长期激活大约变成：

$$
\text{元素数} \approx 2d
$$

也就是**只保留两个 d 维向量**（比如层输入和某个残差点），其它全靠重算。对应字节数：

$$
\text{Bytes}_{\text{per-layer-per-token,ckpt}} \approx (2d)\times 2
$$


### 5.3 代入 Qwen3-8B 的典型结构

用一组“够接近”的参数：

- 隐层维度 $d \approx 4096$
- FFN 维度 $d_{ff} \approx 4d \approx 16{,}384$（略高估一点无妨）；
- 层数 $L = 36$（Qwen3-8B HF 卡上是 36 层）；
- 激活精度 bf16 → 2 B/元素。

#### （1）不使用梯度检查点

每层每 token 的元素数：

$$
6d + 4d_{ff} = 6\times 4096 + 4\times 16{,}384
= 24{,}576 + 65{,}536
= 90{,}112\ \text{元素}
$$

对应字节：

$$
90{,}112 \times 2\ \text{B} \approx 180{,}224\ \text{Bytes} \approx 0.172\ \text{MB}
$$

**激活总显存**：

$$
\text{Act}_{\text{no-ckpt}} \approx  (6d+4d_{ff}) \times 2\ \text{B} \times L \times B \times S
$$

举几个对你有意义的组合（以 2048 序列为例）：

- **B = 1, S = 2048**

$$
\text{Act}_{\text{no-ckpt}} \approx 0.172\ \text{MB} \times 36 \times 2048 \approx \mathbf{12.4\ GB}
$$

- **B = 2, S = 2048**

翻倍即可：
$$
\approx \mathbf{24.8\ GB}
$$
也就是说：

> 在 Qwen3-8B 上，如果不开梯度检查点，**就算 batch=1，S=2048，光激活就可能占 12GB 左右**；batch=2 时就 25GB 左右了。

#### （2）使用按层梯度检查点

每层每 token 激活元素数：

$$
2d = 8192\ \text{元素}
$$

字节数：

$$
8192 \times 2\ \text{B} = 16{,}384\ \text{Bytes} \approx 0.0156\ \text{MB}
$$

总激活显存：

$$
\text{Act}_{\text{ckpt}} \approx (2d)\times 2\ \text{B} \times L \times B \times S
$$
代入：

- **B = 1, S = 2048**

$$
\text{Act}_{\text{ckpt}} \approx 0.0156\ \text{MB} \times 36 \times 2048 \approx \mathbf{1.1\ GB}
$$

- **B = 2, S = 2048**

$$
\approx \mathbf{2.2\ GB}
$$

可以看出：

> 对 Qwen3-8B 这种 8B 级模型，**梯度检查点可以把激活显存从 12GB → ~1GB 这个量级**，收益非常大，尤其是在长序列/多层模型里。



## 6. 把“底座 + 激活”加在一起：几个直观场景

这里统一考虑：

- 模型：**Qwen3-8B**，参数量 $N \approx 8.2\times 10^9$
- 训练精度：**bf16**
- 序列长度：**$S = 2048$**
- 每层梯度检查点是否开启，会影响激活显存
- 额外预留：**PyTorch 缓冲区 / 临时 tensor / 碎片等 约 2 GiB**，记作 `Overhead ≈ 2GB`

> 下文单位都用 “GiB” 粗略近似，不纠结 2¹⁰ 的细节，只看量级。



### 6.1 场景一：bf16 + 标准 AdamW（Case C，不省优化器）

这里假设：

- 参数 / 梯度：bf16（2 B/param）
- 优化器状态 m、v：FP32（各 4 B/param）
- **没有单独的 FP32 主权重副本**（现代 bf16 训练常用做法）

则每个参数显存约：

$$
2(\text{权重}) + 2(\text{梯度}) + 4(m) + 4(v) = 12\ \text{B/param}
$$

对应 **底座显存**：

$$
\text{Base}_{\text{C}} \approx 12 \times N \approx 12 \times 8.2\text{B} \approx \mathbf{92\ GB}
$$

激活显存（不开梯度检查点、$B=1,S=2048$ 时的估算）≈ **12 GB**（前面推过）。

综合：

- **Base（bf16+AdamW）**：~92 GB
- **Act（B=1, S=2048, 无 ckpt）**：~12 GB
- **Overhead**：~2 GB

总显存约：

$$
\text{Total} \approx 92 + 12 + 2 \approx \mathbf{106\ GB}
$$

> 🔎 结论：**在 80GB 单卡上，即便用现代 bf16+AdamW，也很难全参微调 Qwen3-8B（不开 ckpt）**，除非再配 ZeRO/offload 等更激进手段。



### 6.2 场景二：bf16 + 8bit AdamW（Case D，不开 ckpt）

在 Case C 的基础上，把优化器状态换成 8bit 版本：

- 权重：bf16 → 2 B
- 梯度：bf16 → 2 B
- 8bit 优化器状态（m、v 量化）：经验上约 3 B/param

每参数显存：

$$
2 + 2 + 3 \approx 7\ \text{B/param}
$$

底座显存：

$$
\text{Base}_{\text{D}} \approx 7 \times N \approx 7 \times 8.2\text{B} \approx \mathbf{53.5\ GB}
$$

激活显存还是用不开 ckpt、$B=1,S=2048$ 的估算：

- **Act（无 ckpt）**：~12 GB

再加 Overhead ≈ 2 GB：

- **Base（8bit Adam）**：~53.5 GB
- **Act（B=1, S=2048, 无 ckpt）**：~12 GB
- **Overhead**：~2 GB

总显存约：

$$
\text{Total} \approx 53.5 + 12 + 2 \approx \mathbf{67.5\ GB} \ (\approx 68\ GB)
$$

> 🔎 结论：
>  **在 80GB 卡上，bf16+8bit AdamW + 无 ckpt 可以“硬跑” Qwen3-8B 全参微调，但很紧张：**
>
> - B 再稍微上去、S 再长一点，就会逼近 80GB 上限。
> - 如果你把 B=2、S=2048，激活变 ~24GB，总显存就接近 80GB 边缘了。



### 6.3 场景三：bf16 + 8bit AdamW + 梯度检查点（Case D + ckpt）

现在在 Case D 上再打开 **按层梯度检查点**，激活显存大幅下降：

- $B=1,S=2048$ 时，激活显存从 ~12 GB 降到 **~1.1 GB**（前面已推导）。

于是：

- **Base（8bit Adam）**：~53.5 GB
- **Act（B=1, S=2048, 有 ckpt）**：~1.1 GB
- **Overhead**：~2 GB

总显存约：

$$
\text{Total} \approx 53.5 + 1.1 + 2 \approx \mathbf{56.6\ GB} \ (约\ 57\ GB)
$$

如果把 batch 稍微变大一点：

- $B=2,S=2048$：
  - 激活显存 ≈ 2.2 GB
  - 总显存 ≈ 53.5 + 2.2 + 2 ≈ **57.7 GB**（还是很从容）

> ✅ 这就是一个非常现实、而且在工程上广泛使用的组合：**「bf16 权重 + 8bit Adam 优化器 + 梯度检查点」在单卡 80GB 上全参微调 8B 级模型**，既省显存又相对容易实现，还给你留足了 20GB+ 的缓冲空间。



## 7. Qwen3-8B 的显存“速算模板”

这里给一个更通用的速算模板，区分两种常用场景：

- Case C：**bf16 + 标准 AdamW（12 B/param）**
- Case D：**bf16 + 8bit AdamW（7 B/param）**

### 7.1 底座显存（Base）

记：

- 参数量：$N \approx 8.2\times 10^9$

**Case C：bf16 + AdamW**

- 每参数显存：约 **12 B/param**
- 底座显存：

$$
\text{Base}_{\text{C}} \approx 12 \times N 
\approx 12\times 8.2\text{B} 
\approx \mathbf{92\ GB}
$$

**Case D：bf16 + 8bit AdamW**

- 每参数显存：约 **7 B/param**
- 底座显存：

$$
\text{Base}_{\text{D}} \approx 7 \times N 
\approx 7\times 8.2\text{B} 
\approx \mathbf{53.5\ GB}
$$

> 如果你想要最保守、永远不低估的上界，也可以用 **16 B/param**（相当于“有 FP32 主权重 + FP32 m/v”的老方案），那对应 8B 模型就是我们一开始算的 ~122GB。



### 7.2 激活显存（Act，给定 $B, S$）

设模型结构为：

- 隐层维度：$d \approx 4096$
- FFN 维度：$d_{ff} \approx 4d$
- 层数：$L=36$
- 激活精度：bf16（2 B/元素）

则有两个近似公式：

1. **不开梯度检查点（保存整层中间量）**：

$$
\text{Act}_{\text{no-ckpt}} \approx (6d + 4d_{ff})\times 2\ \text{B} \times L \times B \times S
$$

对 Qwen3-8B：
$$
\text{Act}_{\text{no-ckpt}} \approx 90{,}112 \times 2\ \text{B} \times 36BS
$$

1. **开按层梯度检查点（只保留层输入/少量边界）**：

$$
\text{Act}_{\text{ckpt}} \approx (2d)\times 2\ \text{B} \times L \times B \times S
$$

对 Qwen3-8B：
$$
\text{Act}_{\text{ckpt}} \approx 8192 \times 2\ \text{B} \times 36BS
$$
把这些结果除以 $1024^3$ 就是 GiB 量级。
 例如 $B=1,S=2048$ 得到的就是我们前面用的 ~12GB（无 ckpt）和 ~1.1GB（有 ckpt）。



### 7.3 总显存：一行公式记住

统一写成：
$$
\boxed{
\text{Total}(B,S) \approx \text{Base} + \text{Act}(B,S) + \text{Overhead}
}
$$

- **Base**：
  - 用 Case C 就代入 $\text{Base}_\text{C} \approx 92\text{GB}$；
  - 用 Case D 就代入 $\text{Base}_\text{D} \approx 53.5\text{GB}$；
  - 如果想要更保守上界，代入 122GB 也行（老 16 B/param 方案）。
- **Act(B,S)**：
  - 看你是否开梯度检查点，选择 $\text{Act}_{\text{no-ckpt}}$ 或 $\text{Act}_{\text{ckpt}}$ 公式；
  - 代入实际的 batch $B$ 和 seq $S$ 计算。
- **Overhead**：
  - PyTorch 内部缓冲区、临时张量、损失 / logit 等，一般预留 **1–3GB** 就够了。

> 有了这个速算模板，你换模型（比如从 8B → 14B）、换精度（Case C → D）、换 batch/seq 的时候，只要替换 Base 或 $B,S$，就能快速估一个「能不能塞进这张卡」的数量级。



## Appendix 

### 1. 为什么是6d+4d_ff?

#### 1.1 这里的 L 到底指什么？

在我们前面所有显存公式里：

$$
\text{Act} \propto L \times B \times S \times d \times (\text{常数})
$$

**L 就是 Transformer 的「block 数 / 层数」**，也就是：

- 对 LLaMA/Qwen 这类结构，一层通常包含：
  - 一个注意力子层（Self-Attention）
  - 一个前馈子层（FFN / Gated MLP）
  - 再加上前后的 LayerNorm / RMSNorm、残差连接
- 例如 Qwen3-8B 的 HF 配置里 `num_hidden_layers=36`，那这里的 **L=36**。

不把「embedding 层 + 最后一层 LM head」单独算进 L，是因为：

- 它们只出现一次，不会像 block 那样乘以 L；
- 激活量相对「L 层 × B × S × d」这部分来说是小头，可以先忽略。

所以可以直接记：

> **L = 有多少个 Transformer block（自注意力+FFN 的那种层），就填多少。**

#### 1.2 为什么每一层是 `6d + 4d_ff`？

先说一句：

> 这个 **“6d + 4d_ff” 不是某个定律**，而是一个「**比较保守的估算常数**」，用来抓住**不做 checkpoint 时，每层需要长期保留的大致激活量**。

我们看一层标准的「前归一化（pre-norm）」 Transformer block，大致结构（Qwen/LLaMA 系都是这个路子）：

```
x(l)                # 输入，形状 [B, S, d]
 ├─ norm1 → attn → +residual
 └─ norm2 → FFN  → +residual
得到 x(l+1)
```

在**不开梯度检查点**的情况下，为了能做反向传播，通常会把每个子模块内部的一些中间结果保留在显存中。我们粗略地数一下：

**1.2.1 注意力部分大概给出「≈ 6d」**

对注意力子层，一般会有这些激活（按 token 粗略算）：

- `x_attn_in`：进注意力前的向量（归一化后），维度 d
- `q, k, v`：通过三个线性层投影出来，形状还是 d（头拆开只是 reshape）
- `attn_out`：注意力输出，再投影回 d
- 可能还有：
  - 残差前后的中间值
  - norm 的输入/输出

如果你把这些需要在 backward 时用到的向量都算上，一个 token 在一层里的注意力部分，**大概会有若干个 “d 维向量” 需要保留**：

- 粗略估一下：
  - `x_in`、`q`、`k`、`v`、`attn_out`、`x_out`……
  - 算下来就是一个常数 × d，比如 5d、6d，具体取决于实现和框架。

所以这里我直接用一个「略偏保守」的估计：

> **注意力子层激活 ≈ 6d 维向量**

也就是说，**注意力部分长期存活的激活量 ~ 6×d**（每 token 每层）。

**1.2.2 FFN / Gated MLP 部分大概给出「≈ 4d_ff」**

如果不讨论门控结构，在这里，对于u,v 最经典的结构（原始 Transformer）是：
$$
\text{FFN}(x) = W_2 \,\sigma(W_1 x + b_1) + b_2
$$

- 输入：维度是 **d**（隐藏维度）
- 中间：先用 $W_1$ 投到一个更宽的维度 **d_ff**
- 激活：ReLU / GELU 等
- 输出：再用 $W_2$ 投回 d

所以这里有一个“中间层宽度”的超参数：**d_ff**。大家通常会 **把它设成 $k \cdot d$**，k 是一个常数，比如 2、3、4、8 等：
$$
d_{ff} = k \cdot d
$$
但是，在现代 LLM 里FFN一般是**门控的 MLP**，比如 SwiGLU / GeGLU 这种形式：

```python
u = W1 * x        # 维度 d_ff
v = W2 * x        # 维度 d_ff
a = activation(u) # d_ff
g = a ⊙ v         # 维度 d_ff（逐元素乘）
y = W3 * g        # 回到维度 d
x_out = x + y
```

在不开 checkpoint 的前提下，为了反向传播，往往需要保存：

- MLP 输入：`x_ffn_in`（d）
- 中间：`u`、`v`、`a`、`g`（这些都是 d_ff）
- 有些可以在 backward 时临时 recompute，但如果不做特别优化，框架倾向于把大部分中间结果都留着。

所以，从数量级上看，FFN 子层里需要保存的 **“大维度” 激活主要是 d_ff 那几块**：

- 粗略估：
  - `u`、`v`、`a`、`g`，这几乎就是 **4 × d_ff** 级别的向量数目。

为了让估算简单又略偏保守，我直接用：

> **FFN 子层激活 ≈ 4d_ff 维向量**

1.**2.3 合在一起：一层 ≈ `6d + 4d_ff`**

总结注意力 + FFN：

- 注意力：~ 6d
- FFN：~ 4d_ff

于是我们得到：

$$
\text{每层每 token 需要长期保存的激活元素数} 
\approx 6d + 4d_{ff}
$$
再乘以 2 Bytes（bf16 / fp16）：
$$
\text{Bytes}_{\text{per-layer-per-token,no-ckpt}}
\approx (6d + 4d_{ff}) \times 2
$$
这就是我们在公式里用的那一项。

你可以把它理解成：

> 一个实用的「大常数」：
>  **不开 checkpoint 时，一层的长期激活量 ≈“若干个 d 维向量 + 几个 d_ff 级别向量”，用 6 和 4 这两个系数来盖住它。**

具体实现（比如某些框架会主动丢掉部分中间量，在 backward 时重算一点）会略有不同，可能变成 5d+3d_ff 或 4d+4d_ff，但数量级相同，我们这里用 6d+4d_ff 是为了：

- 不低估显存；
- 又不陷在逐 API 的细节里。

<br>

### 2. 从反向传播原理理解GC的工作机制

#### A. MLP 的反向传播公式（先看最简单的，再到门控）

**1) 标准两层 MLP**

设

$$
y = W_2\,\sigma(W_1 x),\qquad L=\ell(y)
$$

记 $g_y=\frac{\partial L}{\partial y}$、$a=W_1x$、$s=\sigma(a)$。则

$$
\begin{aligned}
\frac{\partial L}{\partial W_2} &= g_y\, s^\top,\\
g_s &= W_2^\top g_y,\\
g_a &= g_s \odot \sigma'(a),\\
\frac{\partial L}{\partial W_1} &= g_a\, x^\top,\\
\frac{\partial L}{\partial x} &= W_1^\top g_a.
\end{aligned}
$$

> 依赖项：反传需要 **$x$** 与 **$a=W_1x$**（或能重算出 $a$），以及激活的导数 $\sigma'(a)$。

**2) 门控 MLP（SwiGLU/GeGLU 形式）**

设

$$
u=W_{\text{up}} z,\quad v=W_{\text{gate}} z,\quad m=\phi(u)\odot v,\quad b=W_{\text{down}} m,\quad L=\ell(b)
$$

则

$$
\begin{aligned}
g_b &= \frac{\partial L}{\partial b},\\
\frac{\partial L}{\partial W_{\text{down}}} &= g_b\, m^\top,\qquad g_m = W_{\text{down}}^\top g_b,\\
g_v &= g_m \odot \phi(u),\qquad g_u = g_m \odot \phi'(u)\odot v,\\
\frac{\partial L}{\partial W_{\text{up}}} &= g_u\, z^\top,\qquad \frac{\partial L}{\partial W_{\text{gate}}} = g_v\, z^\top,\\
g_z &= W_{\text{up}}^\top g_u + W_{\text{gate}}^\top g_v.
\end{aligned}
$$

> 依赖项：反传需要 **$z$**（MLP 的输入）以及可得到 **$u,v,\phi(u)$**。若不开检查点，需要把这些**中间量**存下来；开了检查点，可由保存的 **$z$** 正向**重算** $u,v,\phi(u)$。

#### B. LayerNorm 的反向公式（关键点）

设

$$
z=\text{LN}(x)=\gamma\odot\hat{x}+\beta,\quad \hat{x}=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}},\quad \mu=\textstyle\frac{1}{d}\sum_i x_i,\ \ \sigma^2=\frac{1}{d}\sum_i (x_i-\mu)^2
$$

给定 $g_z=\frac{\partial L}{\partial z}$，则

$$
\begin{aligned}
g_{\hat{x}} &= g_z \odot \gamma,\\
g_\sigma &= -\frac{1}{2}(g_{\hat{x}}\odot (x-\mu))(\sigma^2+\epsilon)^{-3/2},\\
g_\mu &= -\sum_i \frac{g_{\hat{x},i}}{\sqrt{\sigma^2+\epsilon}} - \frac{2}{d} g_\sigma \sum_i (x_i-\mu),\\
\frac{\partial L}{\partial x} &= \frac{g_{\hat{x}}}{\sqrt{\sigma^2+\epsilon}} + \frac{2}{d} g_\sigma (x-\mu) + \frac{1}{d} g_\mu.
\end{aligned}
$$

> 依赖项：反传需要 **LN 的输入 $x$** 来计算 $\mu,\sigma^2$。这正是为什么**在层边界要保留 LN 的输入张量**（或能从边界重算得到）。

#### C. Transformer（Pre-LN 解码层）的反向链路

层结构（与你之前的记号一致）：

$$
\begin{aligned}
z_1 &= \mathrm{LN}_1(x_l),\\
a &= \mathrm{MHA}(z_1),\\
h_1 &= x_l + a,\\
z_2 &= \mathrm{LN}_2(h_1),\\
b &= \mathrm{MLP}(z_2),\\
x_{l+1} &= h_1 + b,\qquad L=\ell(x_{l+1}).
\end{aligned}
$$
设 $g_{x_{l+1}}=\frac{\partial L}{\partial x_{l+1}}$。按链式法则与残差“分流”：

1. **第二个残差口** $x_{l+1}=h_1+b$

$$
g_{h_1} \mathrel{+}= g_{x_{l+1}},\qquad g_b = g_{x_{l+1}}.
$$

1. **MLP 反传**（用上节 MLP 公式）

$$
g_{z_2} = \frac{\partial b}{\partial z_2}^\top g_b,\quad \text{并得到 } \frac{\partial L}{\partial W_{\text{down/up/gate}}}.
$$

1. **LN$_2$ 反传**（用 LN 公式，需要 $h_1$）

$$
g_{h_1} \mathrel{+}= \frac{\partial z_2}{\partial h_1}^\top g_{z_2}.
$$

> 到此，$g_{h_1}$ 汇聚了两条路的梯度：一条来自残差直连（第 1 步），一条来自 MLP→LN$_2$（第 2–3 步）。

1. **第一个残差口** $h_1=x_l+a$

$$
g_{x_l} \mathrel{+}= g_{h_1},\qquad g_a = g_{h_1}.
$$

1. **MHA 反传**（需要 $z_1=\mathrm{LN}_1(x_l)$ 重建 Q,K,V, 注意力概率等）

$$
g_{z_1} = \frac{\partial a}{\partial z_1}^\top g_a,\quad \text{并得到 } \frac{\partial L}{\partial W_Q,W_K,W_V,W_O}.
$$

1. **LN$_1$ 反传**（用 LN 公式，需要 $x_l$）

$$
g_{x_l} \mathrel{+}= \frac{\partial z_1}{\partial x_l}^\top g_{z_1}.
$$

#### D. 为什么只需保留 $x_l$ 与 $h_1$（按层梯度检查点的视角）

- **残差分流需要边界激活：**
  - 在 $x_{l+1}=h_1+b$ 处，必须把 $g_{x_{l+1}}$ **一份直接给 $h_1$**，另一份走 $b$（MLP）链路；这要求你在该边界拥有 **$h_1$**（或 $x_{l+1}$ 之一）来“接”住梯度并继续传播。
  - 在 $h_1=x_l+a$ 处，同理需要把 $g_{h_1}$ 分流到 **$x_l$** 与 $a$（注意力）两路，所以要有 **$x_l$** 或与之等价的边界张量。
- **LayerNorm 需要其输入：**
   LN 的反传显式依赖其**输入**来计算 $\mu,\sigma^2$（见上式）。因此：
  - 对 LN$_2$ 你需要 **$h_1$**；
  - 对 LN$_1$ 你需要 **$x_l$**。
- **中间量可由边界“重算”：**
   开启**按层**梯度检查点时，不长期保存 $Q,K,V$、注意力概率、MLP 的 $u,v,\phi(u)$ 等；反向阶段用保存的 **$x_l$** 重算 $z_1\!\to$ MHA 全流程，用保存的 **$h_1$** 重算 $z_2\!\to$ MLP 全流程。
   因而长期驻留显存的只需 **两份 $d$-维张量**（数量级即 $\approx 2d$ 每 token 每层）。

> 小结：
>  公式揭示了**梯度分流（残差）\**与 \*\*LN 反传的输入依赖\*\*这两件事，决定了按层检查点\**必须**在两处边界保留激活（$x_l$ 与 $h_1$ 或等价选择）。其它中间激活都可以在反传时“现算现用”，从而把激活显存从 $(6d+4d_{ff})$ 压到 $\approx 2d$。
