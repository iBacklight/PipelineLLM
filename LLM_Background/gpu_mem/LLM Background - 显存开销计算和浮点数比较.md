# LLM 笔记 - 显存开销计算和浮点数比较

[TOC]



## 推理

#### **模型参数**

在全精度（FP32）下，模型每个参数占用4个字节（32/8=4），首先recall 1G显存 = 1024MB = 1024^3 Bytes ~= 10^9 Bytes

推理占用显存公式
$$
Mem = \frac{\text{模型大小 in Bytes} \times 4(\text{单个参数占用的字节数})}{1024^3} GB
$$
e.g. : 1B model, Mem = $\frac{10^9\times4}{1024^3}$=3.725GB

如果是FP16，则减半，INT8则为1/4，以此类推；同理7B=1B*7， 32B=1B * 32，以此类推

| Model Size (Parameters) | FP32 (GB) | FP16/BF16 (GB) | INT8 (GB) | INT4 (GB) |
| :---------------------- | :-------- | :------------- | :-------- | :-------- |
| 7 Billion (7B)          | 28        | 14             | 7         | 3.5       |
| 13 Billion (13B)        | 52        | 26             | 13        | 6.5       |
| 30 Billion (30B)        | 120       | 60             | 30        | 15        |
| 65 Billion (65B)        | 260       | 130            | 65        | 32.5      |
| 70 Billion (70B)        | 280       | 140            | 70        | 35        |
| 168 Billion (168B)      | 672       | 336            | 168       | 84        |

除此以外，可能还要保存：

#### **激活值 (activations)**

- 前向传播过程中每一层的中间输出，需要在下一层使用。
- 通常 batch size 较小时，这部分占用可控；较大占用比较大。
- 层数越多，该项越多。

#### **缓存 (KV cache)**

- 对于 Transformer 推理，注意力层需要保存 Key/Value（自回归推理时要复用）。
- 这部分显存开销随 **上下文长度 (sequence length)** 和 **层数** 增加，可能非常大。



## 训练

在训练中（增加了反向传播），**除了模型参数还要保存梯度值和优化器状态**

#### **模型参数 $\theta$**

神经网络的权重矩阵、偏置向量等。

- 记作 $\theta$，每一次迭代都会更新。

------

#### **梯度 $g_t$**

在一次前向 + 反向传播后，得到损失函数 $L(\theta)$ 对参数的梯度：
$$
g_t = \nabla_\theta L(\theta_t)
$$
它告诉我们：在当前参数位置，应该往哪个方向移动才能降低损失。大小与模型推理中模型参数占用的显存相同。

------

#### **优化器状态 (Optimizer states)**

不同优化器会为每个参数额外维护一些“历史信息”，辅助更新。这些状态就是显存里的“大头”，典型情况：

(1) SGD（随机梯度下降，无动量）

只用到梯度：
$$
\theta_{t+1} = \theta_t - \eta g_t
$$

- $\eta$为学习率
- 状态：无
- 显存占用：仅参数和梯度（即总共模型推理显存的2倍）

------

(2) SGD with Momentum

会多存一个“速度向量” $v_t$：
$$
v_t = \beta v_{t-1} + g_t
$$

- 状态：$v_t$（和参数同维度）
- 显存：1 倍参数量的额外存储（即总共模型推理显存的3倍）

------

(3) Adam / AdamW（最常用的）

维护两个状态：一阶矩 (momentum) 和二阶矩 (RMS)：
$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$
再做偏差修正：
$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad 
\hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$
参数更新：
$$
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

- 状态：$m_t, v_t$，各与参数同维度
- 显存：参数大小的 **2 倍额外存储**（即总共模型推理显存的4倍）

------

#### **显存占用关系总结**

假设模型参数大小为 $N$：

- **参数 (weights)**：$N$
- **梯度 (gradients)**：$N$
- **优化器状态**：
  - SGD：0
  - Momentum：+1 × $N$
  - Adam/AdamW：+2 × $N$

所以：

- SGD 总共需要 $2N$（参数+梯度）
- Momentum 需要 $3N$
- Adam/AdamW 需要 $4N$（参数+梯度+2个状态）

这就是为什么训练大模型时，显存开销里优化器状态往往最大（尤其是 Adam）。

------

#### **图示直觉**

- **参数**：就是“我们要爬山的当前位置”。
- **梯度**：就是“山坡的当前斜率”。
- **优化器状态**：就是“记忆我们过去走的方向/斜率”，避免来回震荡，加快收敛。



## 总结

除此之外，可能还有一些额外的现存开销，如下总结：

| 占用项                | 推理 | 训练 | 备注                                                         |
| --------------------- | ---- | ---- | ------------------------------------------------------------ |
| 模型参数 (weights)    | ✅    | ✅    | 必须常驻显存                                                 |
| 激活值 (activations)  | 少   | 多   | 训练必须保留更多激活，**反向传播需要用到前向传播时的中间结果**<br />所以不能像推理一样，激活值在被下一层使用之后，就可以立即释放。 |
| 梯度 (gradients)      | ❌    | ✅    | 与参数同维度                                                 |
| 优化器状态 (Adam m/v) | ❌    | ✅    | 约 2 × 参数大小                                              |
| 临时缓存 (workspace)  | ✅    | ✅    | CUDA 内核执行时，需要一些中间 buffer。<br />cuDNN、FlashAttention、矩阵乘法 (GEMM) 都会申请临时显存 |
| KV cache              | ✅    | ❌    | 只在推理自回归用                                             |
| 通信缓存              | ❌    | ✅    | 分布式训练用                                                 |
| AMP master weights    | ❌    | ✅    | 混合精度训练特有，AMP (FP16/BF16 训练) 下，<br />常常需要维护一份 **FP32 master weights** 来保证稳定更新。 |





## Appendix：数值精度对比

### Overall

| 格式                            | 位宽 (Sign / Exponent / Mantissa) | 动态范围 (约)                       | 精度 (小数部分)              | 优点                                                         | 缺点                                           | 常见应用场景                                              |
| ------------------------------- | --------------------------------- | ----------------------------------- | ---------------------------- | ------------------------------------------------------------ | ---------------------------------------------- | --------------------------------------------------------- |
| **FP32 (单精度浮点)**           | 32 位 (1 / 8 / 23)                | ~1e-38 ~ 1e38                       | ~7 位十进制                  | 精度高，范围大，数值稳定                                     | 显存/计算开销大                                | 标准训练、科学计算、精度敏感推理                          |
| **FP16 (半精度浮点)**           | 16 位 (1 / 5 / 10)                | ~6e-5 ~ 6e4                         | ~3–4 位十进制                | 内存/显存占用低；尾数比 BF16 多，精度较高                    | 范围小，容易溢出/下溢，需要 loss scaling       | 早期混合精度训练，部分推理                                |
| **BF16 (Brain Floating Point)** | 16 位 (1 / 8 / 7)                 | ~1e-38 ~ 1e38 (≈FP32)               | ~2–3 位十进制                | 范围与 FP32 相同，训练稳定，无需 loss scaling                | 尾数少，精度比 FP16 低                         | 现代大模型训练（GPT, LLaMA, PaLM），RLHF                  |
| **FP8 (新兴浮点格式)**          | 8 位 (两种：E5M2 或 E4M3)         | E5M2: ~1e-6 ~ 6e4/E4M3: ~1e-3 ~ 5e2 | 2–3 位十进制                 | 显存压缩极大；NVIDIA H100/Grace Hopper 原生支持；适合多阶段混合精度 | 精度非常低，训练不稳定，常需校准或和高精度结合 | 超大模型训练/推理加速（多在 H100 GPU 上用）               |
| **INT8 (8 位整型量化)**         | 8 位 (无符号或有符号)             | -128 ~ 127 (或 0 ~ 255)             | 无浮点精度，通过 scale 近似  | 存储/推理效率高，广泛支持；精度和性能平衡点好                | 需量化/反量化，精度损失                        | LLM 推理加速，边缘设备部署                                |
| **INT4 (4 位整型量化)**         | 4 位 (常见范围 -8 ~ 7 或 0 ~ 15)  | 很小                                | 精度极低，通常配合量化 scale | 压缩比极高，存储/推理极快                                    | 精度损失大，训练几乎不可行                     | 超大模型推理压缩（如 GPTQ、AWQ），移动端/IoT 超低资源场景 |

举个例子：当我们储存一个数123456.789

| 格式         | 能否存储 `123456.789` | 存储结果 (十进制近似) | 说明                       |
| ------------ | --------------------- | --------------------- | -------------------------- |
| **FP32**     | ✅                     | 123456.7890625        | 精度高，7 位有效数字       |
| **FP16**     | ❌                     | 溢出                  | 范围太小                   |
| **BF16**     | ✅                     | 123456.8              | 精度低，但能存下           |
| **FP8-E5M2** | ✅                     | ~123000               | 动态范围够，但小数精度极低 |
| **FP8-E4M3** | ❌                     | 溢出                  | 范围不足                   |
| **INT8**     | ⚠️                     | ~127000 (需量化)      | 精度取决于 scale           |
| **INT4**     | ⚠️                     | ~140000 (需量化)      | 精度差，极限压缩           |



### 每个位置的意义

1. **符号位 (Sign)**

- **占 1 bit**。
- 作用：决定数值是正数还是负数。`0` → 正数，`1` → 负数

例如：Sign=0 → +123.45，Sign=1 → -123.45

------

2. **指数位 (Exponent)**

- 决定数值的**动态范围**（大小范围，主导但是依然由尾数进行微调最大值，详见下一节）。
- 存的是一个 **偏移后的指数 (biased exponent)**，而不是直接的指数值。
  - 比如在 IEEE FP32（8 位指数）：
    - 真正的指数 = 存储值 – 127 (偏置量 bias=127)。
- 指数越多位 → 可表示的范围越大。
  - FP16：5 位 → 范围大约 10^-5 ~ 10^5
  - BF16/FP32：8 位 → 范围大约 10^-38 ~ 10^38

------

3. **尾数位 (Mantissa / Fraction / Significand)**

- 决定数值的**精度**（小数部分）。

- 默认前面有个隐藏的 `1.`（除非是非规格化数）。

- Mantissa 越长，能保留的小数越多 → 数值更精确。

  - FP32：23 位尾数 → 约 7 位十进制精度
  - FP16：10 位尾数 → 约 3–4 位十进制精度
  - BF16：7 位尾数 → 约 2–3 位十进制精度

  - 



### 动态范围计算

1. **IEEE 风格浮点数**：

$$
(-1)^{sign} \times 1.mantissa \times 2^{exponent-bias}
$$

指数部分有 $k$ 位 → 最大可表示的二进制指数 = $2^k - 1$。
 但是：

- 全 0 = 特殊值（非规格化数、0）
- 全 1 = 特殊值（无穷大/NaN）

所以真正能用的指数范围是：
$$
1 \sim 2^k - 2
$$
对应的**有效指数范围**就是：
$$
(1 - bias) \;\; \text{到} \;\; (2^k - 2 - bias)
$$
其中：
$$
bias = 2^{k-1} - 1
$$


2. **具体计算**

补充说明：浮点数的有效数字部分（significand / mantissa）总是写成：
$$
1.f_1 f_2 f_3 \ldots f_m
$$
其中：

- 前面的 **1** 是隐藏的 **leading 1**（标准化数的约定，这就是为什么说尾数位知识微调，因为在(1,2)之间）。
- $f_i$ 是尾数位（0 或 1），一共 $m$ 位。

所以：
$$
\text{significand} = 1 + \sum_{i=1}^{m} f_i \cdot 2^{-i}
$$

好，现在回到上面的式子，**当指数取最大**：$2^k - 2 - bias$

- **当 mantissa 全 1**：如果 $f_1, f_2, \ldots, f_m$ 都是 1，那么：
  $$
  \text{significand} = 1 + \left(2^{-1} + 2^{-2} + \ldots + 2^{-m}\right)
  $$
  这个和式是一个几何级数：
  $$
  S = 2^{-1} + 2^{-2} + \ldots + 2^{-m} = 1 - 2^{-m}
  $$
  所以值接近 $2 - 2^{-m}$（即小数部分接近 1.111...）。
  总体来看：

$$
\text{max} \approx (2 - 2^{-m}) \times 2^{2^k - 2 - bias}
$$

​	Again, 这里的$(2 - 2^{-m})$实际上就是上面公式中的*1.mantissa*

**当指数取最小**：$1 - bias$

- **当 mantissa = 0**（即 1.0）。所以：

$$
\text{min} \approx 1.0 \times 2^{1-bias}
$$

这里最大最小值计算后，我们知道浮点数本来是基于 **2 的指数**：
$$
2^{e_{\min}} \;\sim\; 2^{e_{\max}}
$$
但我们更习惯用 **10 的指数** 表达（10 进制科学记数法）。所以需要做换底：
$$
2^n = 10^{\log_{10}(2^n)}
$$
具体为什么这么计算，可以看下一section精度计算。于是动态范围大约就是：
$$
10^{\log_{10}(\text{max})} ~ \text{到} ~ 10^{\log_{10}(\text{min})}
$$



### 精度计算

1. **尾数位与二进制精度**

尾数有 $m$ 位二进制位，就意味着你能区分的最小差别是：
$$
2^{-m}
$$
这其实可以理解成分辨率，即小数的最小单位$\Delta$。例如：

- 1 位尾数 → 最小差别 $2^{-1} = 0.5$
- 10 位尾数 → 最小差别 $2^{-10} \approx 0.000976$

**2. 转换成十进制有效位数**

我们现在有：
$$
\Delta = 2^{-m}
$$
要把它换算成“约等于 $10^{-d}$。

取对数：
$$
- d = \log_{10}(\Delta) = \log_{10}(2^{-m}) = -m \cdot \log_{10}(2)
$$
所以：
$$
d \approx m \cdot \log_{10}(2)
$$
这就是 **二进制尾数位数 → 十进制有效位数** 的转换公式。

所以
$$
m \times \log_{10}(2) \;\; \approx \;\; m \times 0.301
$$
也就是说：

- 每 **3.3（1/0.301） 个二进制位** ≈ **1 个十进制有效位**。

**3. 具体例子**

- **FP32 (23 位 mantissa)**
  $$
  23 \times 0.301 \approx 6.9 \;\Rightarrow\; 约 7 位十进制有效数字
  $$

- **FP16 (10 位 mantissa)**
  $$
  10 \times 0.301 \approx 3.0 \;\Rightarrow\; 约 3–4 位十进制有效数字
  $$

- **BF16 (7 位 mantissa)**
  $$
  7 \times 0.301 \approx 2.1 \;\Rightarrow\; 约 2–3 位十进制有效数字
  $$

- **FP8 (E5M2, 2 位 mantissa)**
  $$
  2 \times 0.301 \approx 0.6 \;\Rightarrow\; 约 1 位十进制有效数字
  $$

- **FP8 (E4M3, 3 位 mantissa)**
  $$
  3 \times 0.301 \approx 0.9 \;\Rightarrow\; 约 1 位十进制有效数字
  $$

------

**4. 总结**

- **公式**：十进制有效位数 ≈ 尾数位数 × 0.301
- **直观理解**：
  - FP32：约 7 位有效数字
  - FP16：约 3 位
  - BF16：约 2 位
  - FP8：1 位左右
- **尾数位数越多 → 小数部分越精确**



### 训练卡直观对比

| 型号     | 架构 / 型号版本                                     | 显存 / 类型                                                  | 显存带宽                                                     | 理论算力 / 特定精度                                          | 功耗 / TDP                                                   | 主要用途 / 适合场景                                       | 参考价格 / 租用价 / 市场报价                                 |
| -------- | --------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | --------------------------------------------------------- | ------------------------------------------------------------ |
| **H100** | Hopper 架构，SXM / PCIe / NVL 版本                  | 80 GB HBM3（部分 NVL 提供 94 GB 版本） ([NVIDIA](https://www.nvidia.com/en-us/data-center/h100/?utm_source=chatgpt.com)) | ~ 3.35 TB/s（SXM 版） ([NVIDIA](https://www.nvidia.com/en-us/data-center/h100/?utm_source=chatgpt.com)) | TF32 (Tensor 核心): ~ 989 TFLOPS ([NVIDIA](https://www.nvidia.com/en-us/data-center/h100/?utm_source=chatgpt.com))；FP16/BF16: ~ 1,979 TFLOPS ([NVIDIA](https://www.nvidia.com/en-us/data-center/h100/?utm_source=chatgpt.com))；FP8 / INT8: ~ 3,958 TFLOPS / TOPS ([NVIDIA](https://www.nvidia.com/en-us/data-center/h100/?utm_source=chatgpt.com)) | 总功耗较高（视版本可能达 500W+ 甚至更高）                    | 面向大型 AI 模型训练 / 推理 /混合任务                     | 市面上一些报价如 H100 PCIe 80GB 报价约 $30,000 美元区间 ；云端租用如 ~$2.65／小时 （SXM5／高端版） ([datacrunch.io](https://datacrunch.io/blog/nvidia-h100-gpu-specs-and-price?utm_source=chatgpt.com)) |
| **A100** | Ampere 架构（上一代主力）                           | 40 GB 或 80 GB HBM2e（取决于版本）                           | ~ 1.555 / ~ 2.039 TB/s（视版本） ([OpenMetal IaaS](https://openmetal.io/resources/on-openmetal/data-center-nvidia-gpu-comparison-table-with-specs/?utm_source=chatgpt.com)) | FP32: ~ 19.5 TFLOPS（纯浮点） ([HorizonIQ](https://www.horizoniq.com/blog/h100-vs-a100-vs-l40s/?utm_source=chatgpt.com))；TF32 (Tensor): ~ 312 TFLOPS（有加速） ([HorizonIQ](https://www.horizoniq.com/blog/h100-vs-a100-vs-l40s/?utm_source=chatgpt.com))；FP16/BF16: ~ 624 TFLOPS ([HorizonIQ](https://www.horizoniq.com/blog/h100-vs-a100-vs-l40s/?utm_source=chatgpt.com)) | 功耗较高（通常 400-500W 范围，依版本而定）                   | 依然在很多训练 / 推理系统中广泛使用，是性价比 “老将”      | 有商家报价 40 GB 版约 29,000 左右 ；在云端租用时，A100 80GB 租价 ~3.67／小时（Azure 的报价） |
| **H200** | 更新一代（基于 Hopper / 进化版本）                  | 141 GB HBM3e ([NVIDIA](https://www.nvidia.com/en-us/data-center/h200/?utm_source=chatgpt.com)) | ~ 4.8 TB/s 带宽 ([NVIDIA](https://www.nvidia.com/en-us/data-center/h200/?utm_source=chatgpt.com)) | 支持高精度和低精度混合算力（NVIDIA 宣称其在某些 LLM 推理任务上比 H100 有 ~1.6× 性能提升） ([NVIDIA](https://www.nvidia.com/en-us/data-center/h200/?utm_source=chatgpt.com)) | 功耗较高                                                     | 面向极大规模训练 + 推理任务                               | 目前公开零售信息较少（更多用于大厂 / 数据中心内部或定制系统） |
| **L40S** | Ada Lovelace 架构                                   | 48 GB GDDR6 ([NVIDIA](https://www.nvidia.com/en-us/data-center/l40s/?utm_source=chatgpt.com)) | ~ 864 GB/s 带宽 ([HorizonIQ](https://www.horizoniq.com/blog/h100-vs-a100-vs-l40s/?utm_source=chatgpt.com)) | FP32: ~ 91.6 TFLOPS ([NVIDIA](https://www.nvidia.com/en-us/data-center/l40s/?utm_source=chatgpt.com))；TF32 (Tensor): ~ 366 TFLOPS ([NVIDIA](https://www.nvidia.com/en-us/data-center/l40s/?utm_source=chatgpt.com))；FP16 / FP8: ~ 733 / 1,466 TFLOPS ([NVIDIA](https://www.nvidia.com/en-us/data-center/l40s/?utm_source=chatgpt.com)) | 最大功耗约 350W ([NVIDIA](https://www.nvidia.com/en-us/data-center/l40s/?utm_source=chatgpt.com)) | 在推理、微调、小/中型模型训练、图形渲染加速混合场景中常用 | 市场报价如约 $7,569 美元左右 ([asacomputers.com](https://www.asacomputers.com/GPU.html?utm_source=chatgpt.com))；云端使用时常被定位为推理 / 辅助训练卡 |
| **A30**  | Ampere 架构                                         | 24 GB（部分版本） ([NVIDIA](https://www.nvidia.com/en-us/data-center/products/a30-gpu/?utm_source=chatgpt.com)) | 933 GB/s 带宽（部分资料） ([NVIDIA](https://www.nvidia.com/en-us/data-center/products/a30-gpu/?utm_source=chatgpt.com)) | 支持多精度变换 / Tensor Core 加速                            | 功耗适中                                                     | 混合推理 + 中型训练任务，适合性价比 / 辅助用途            | 相比 H100 / A100，价格要低得多（公开零售报价不多见）         |
| H20      | “合规版 / 削弱版 Hopper / 面向中国 /受出口控制版本” | ~ 96 GB HBM3（报导）                                         | ~ 4.0 TB/s（报导）                                           | FP16 / BF16: ~ 2,368 TFLOPS（报）   INT8/FP8 推理性能较优（报） | ~ 400 W（报）                                                | 推理 / 合规市场 /中大型模型推理场景                       | 报价约 $12,000–$15,000 美元区间（媒体报道） [eWeek](https://www.eweek.com/news/deepseek-ai-models-nvidia-h20-chips/?utm_source=chatgpt.com)   库存受限、订单受限 |