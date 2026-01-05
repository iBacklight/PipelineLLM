# LLM 训练流程：PreTrain and SFT

[TOC]



## LLM 介绍

LLM，即 Large Language Model，中文名为大语言模型或大型语言模型，是一种相较传统语言模型参数量更多、在更大规模语料上进行预训练的语言模型。大模型时间线梳理：

| 时间           | 开源 / 可下载权重 LLM                                        | 闭源 / 专有 LLM（API/产品）                                  |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **2022.11**    | —                                                            | **OpenAI – ChatGPT**（基于 GPT-3.5）                         |
| **2023.02**    | **Meta – LLaMA**；**复旦 – MOSS**                            | —                                                            |
| **2023.03**    | **Stanford – Alpaca**；**Vicuna**；**智谱 – ChatGLM**        | **OpenAI – GPT-4**；**百度 – 文心一言**；**Anthropic – Claude**；**Google – Bard** |
| **2023.04**    | **阿里 – 通义千问（Qwen beta）**；**Stability – StableLM**   | **商汤 – 日日新**                                            |
| **2023.05**    | **TII – Falcon**；（研究向）**Microsoft – Phi-1**            | **讯飞 – 星火**；**Google – PaLM 2**                         |
| **2023.06**    | **智谱 – ChatGLM2**；**上海AI Lab – InternLM（书生·浦语）**；**百川 – BaiChuan**；**虎博 – TigerBot** | **360 – 智脑**                                               |
| **2023.07**    | **Meta – LLaMA 2**                                           | **Anthropic – Claude 2**；**华为 – 盘古3.0**                 |
| **2023.08**    | **Qwen-7B/等开放权重**（阿里首次放权重 7B/后续尺寸） ([阿里云](https://www.alibabacloud.com/help/en/model-studio/what-is-qwen-llm?utm_source=chatgpt.com)) | —                                                            |
| **2023.09**    | **BaiChuan 2**                                               | **Google – Gemini（品牌亮相，后续 1.x/2.x 系列）**；**腾讯 – 混元** ([维基百科](https://en.wikipedia.org/wiki/Gemini_(language_model)?utm_source=chatgpt.com)) |
| **2023.11**    | **零一万物 – Yi**；**幻方 – DeepSeek（V系列起步）**          | **xAI – Grok**                                               |
| **2024.02**    | **Google – Gemma**（开源家族） ([阿里云](https://www.alibabacloud.com/help/en/model-studio/what-is-qwen-llm?utm_source=chatgpt.com)) | —                                                            |
| **2024.03**    | **Databricks – DBRX（MoE，开放模型许可）**                   | —                                                            |
| **2024.04**    | **Meta – LLaMA 3（首批 8B/70B 等）** ([Meta AI](https://ai.meta.com/blog/meta-llama-3/?utm_source=chatgpt.com)) | —                                                            |
| **2024.05**    | —                                                            | **Google – Gemini 1.5 Pro/Flash（广泛可用）** ([Google Cloud](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions?utm_source=chatgpt.com)) |
| **2024.06**    | **Qwen 2（家族更新）**；**DeepSeek-V2（开源）**；**Mistral Large 2（部分开放）** ([维基百科](https://en.wikipedia.org/wiki/Qwen?utm_source=chatgpt.com)) | —                                                            |
| **2024.07**    | **Meta – LLaMA 3.1（扩到 405B 版本、长上下文）** ([Meta AI](https://ai.meta.com/blog/meta-llama-3-1/?utm_source=chatgpt.com)) | —                                                            |
| **2024.09–12** | **Gemini 2.0 Flash（实验→默认）**（这是 Google 自家闭源产品线，但列于对照参照）；**Mixtral 8×7B/8×22B 持续更新（Apache-2.0）** ([The Verge](https://www.theverge.com/news/603821/google-gemini-flash-2-0-ai-app?utm_source=chatgpt.com)) | **Google – Gemini 2.0/2.5 系列持续更新（闭源）** ([维基百科](https://en.wikipedia.org/wiki/Gemini_(language_model)?utm_source=chatgpt.com)) |
| **2025.01**    | —                                                            | **OpenAI – o3-mini（推理/思考系列，小型）**；（月末）**DeepSeek – R1（推理型，MIT 许可）** ([维基百科](https://en.wikipedia.org/wiki/OpenAI_o3?utm_source=chatgpt.com)) |
| **2025.02**    | —                                                            | **OpenAI – o4-mini（同日配套）**；**Gemini 2.0 Pro（闭源）** ([OpenAI](https://openai.com/index/introducing-o3-and-o4-mini/?utm_source=chatgpt.com)) |
| **2025.03**    | **Google – Gemma 3（开源家族续作）**                         | —                                                            |
| **2025.04**    | —                                                            | **OpenAI – o3（GA 发布）/o3-pro（6-10 上线）** ([OpenAI](https://openai.com/index/introducing-o3-and-o4-mini/?utm_source=chatgpt.com)) |
| **2025.05**    | —                                                            | **Anthropic – Claude 4（发布 Opus 4 / Sonnet 4）** ([Anthropic](https://www.anthropic.com/news/claude-4?utm_source=chatgpt.com)) |
| **2025.08–09** | （社区与厂商多轮更新）                                       | **OpenAI – GPT-5（正式发布页面）**（闭源/服务） ([OpenAI](https://openai.com/index/introducing-gpt-5/?utm_source=chatgpt.com)) |

主流模型对比

| 模型                              | 开/闭源  | 主要架构                   | 参数量（总/激活）                                            | 上下文长度（tokens）                                         | 模态                                 | 要点/备注                                                    |
| --------------------------------- | -------- | -------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------ | ------------------------------------------------------------ |
| **Llama 4（Scout / Maverick）**   | 开放权重 | **MoE**（多专家）          | *Scout*：约 **109B/17B 激活**；*Maverick*：更大（报道称 ~402B 级别，激活仍 ~17B） | *官方博文与生态文章称* **10M**（Scout）；另有模型卡示例 **128K** 用法（不同渠道数值不一） | 文本/图像（原生多模态）              | Meta 博文宣称“前所未有”的长上下文，Cloudflare 亦称 Scout **10,000,000**；HF 模型卡示例写到 **128K**（工程默认值）；属 **开放权重**许可而非完全开源。([Meta AI](https://ai.meta.com/blog/llama-4-multimodal-intelligence/?utm_source=chatgpt.com)) |
| **Qwen 3**（家族）                | 开放权重 | Dense/混合（家族多尺⼨）   | 官方未统一披露全家族参数；如 **Qwen3-32B** 为 **32.8B**      | **32K**（官方博文）；部分变体与后续“Next”公告扩展到 **262K** | 文本/多模态（视型号）                | Qwen3 博文指“最终阶段用长上下文数据将上下文扩到 32K”；HF 上 **Qwen3-32B** 模型卡写 32.8B 与 **32K**；“Qwen3-Next”页面提到 **262,144** tokens。([Qwen](https://qwenlm.github.io/blog/qwen3/?utm_source=chatgpt.com)) |
| **Kimi K2**（Moonshot）           | 开放权重 | **MoE**                    | **1T 总参**，**32B 激活**；384 专家、每步选 8                | **128K**（模型卡/官方资料）；**K2-0905** 更新**256K**        | 文本/多模态（视版本）                | 官方页与 HF 模型卡列明 **1T/32B（MoE）**、默认 **128K**；9/05 版卡片与官号公告称 **256K**。([GitHub](https://github.com/MoonshotAI/Kimi-K2?utm_source=chatgpt.com)) |
| **DeepSeek V3.1**                 | 开放权重 | **MoE**                    | 模型卡标注**Total 671B / Activated 37B**                     | **128K**（模型卡）                                           | 文本（部分推理/“thinking”模式）      | HF 模型卡清楚给出 **671B/37B**、**128K**，并区分 base/instruct、thinking 与非 thinking。([Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3.1?utm_source=chatgpt.com)) |
| **GPT-5**（OpenAI）               | **闭源** | （未披露；外界多推测 MoE） | **未披露**                                                   | **400K**（官方页面；各子型 mini/nano 同为 400K）             | 多模态（文/图/视）                   | 官方页给出 **400K** 上下文与价格分档；参数规模未披露。([OpenAI](https://openai.com/gpt-5/?utm_source=chatgpt.com)) |
| **Gemini 2.5 Pro**（Google）      | **闭源** | （未披露）                 | **未披露**                                                   | **1,000,000**（2M “coming soon”）                            | **原生多模态**（文/音/图/视频/代码） | 官方博客称 2.5 Pro “今天”提供 **100 万**上下文，并预告 **200 万**；Vertex 文档亦描述 2.5 Flash/Pro 的“thinking/推理”能力。([blog.google](https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/?utm_source=chatgpt.com)) |
| **Claude 4（Opus 4 / Sonnet 4）** | **闭源** | （未披露）                 | **未披露**                                                   | **200K**（发布时）；**Sonnet 4** 后续**提升至 1M**（API beta/企业通道） | 多模态（视型号）                     | 官方发布与后续公告：发布时 **200K**；8/12 更新 Sonnet 4 **1M**（beta/企业/Vertex/Bedrock）。([Anthropic](https://www.anthropic.com/news/claude-4?utm_source=chatgpt.com)) |
| **Grok**（xAI；Grok-3/4）         | **闭源** | （未披露；推理向变体）     | **未披露**                                                   | **Grok-3：131,072**（官方文档页）；**Grok-4-Fast：2,000,000**（xAI docs 概览）；亦有媒体整理称 Grok-3 公告“理论 1M” | 多模态（视型号）                     | 建议以 xAI 官方文档为准：Grok-3 文档页 **131,072**；xAI 概览/公告提到 **Grok-4-Fast 2M**。第三方文章宣称值更大时需谨慎。([docs.x.ai](https://docs.x.ai/docs/models/grok-3?utm_source=chatgpt.com)) |

<br>

## LLM 训练流程

一般而言，训练一个完整的 LLM 需要经过三个阶段——Pretrain、SFT 和 RLHF。

### Pretrain

预训练任务沿承了 GPT 模型的经典预训练任务——因果语言模型（Causal Language Model，CLM）。因果语言模型建模，即和最初的语言模型一致，通过给出上文要求模型预测下一个 token 来进行训练。

#### 1. 预训练的目标/任务形式

主流任务：**Causal Language Modeling（CLM，自回归下一词预测）**

- 目标：给定前缀 $x_{1:t-1}$，最大化 $\log p_\theta(x_t \mid x_{1:t-1})$。
- 训练时对序列内**每个位置**做预测并累加损失（掩掉 padding）。
- 优势：与推理时“从左到右生成”一致，**单一目标即可覆盖通用能力**（理解、生成、推理可在大规模下涌现）。
- 代表：GPT 系列、LLaMA、Qwen、DeepSeek 等。

其他常见/可选任务（可与 CLM 混合，少数模型用作补充或替代）：

- **PrefixLM / Seq2Seq LM**：对前缀做因果 mask、后缀做全预测（适合指令式/文档续写/检索拼接场景）。
- **Span Corruption（T5/UL2）**：随机抹去 span，让模型输出缺失段（“infill”）；利于编辑/补全类任务。
- **双向掩码 MLM（BERT 系）**：仅理解更强，但不适合长文本生成；当今主流 LLM 预训练很少单独采用。
- **代码特殊目标**：如填空/续写混合、多文件上下文拼接等。
- **MoE 路由辅助损失**：MoE 需要额外的 load-balancing/熵正则等辅助项（只对 MoE）。

> 结论：**CLM 是当今预训练的主流与基座**；其它目标更多是“增强器”或为特定任务/架构（如 T5/UL2、代码/编辑）服务。



#### 2. 数据收集与配比

**2.1 数据来源（多模态先略过文本主线）**

- **网页大语料**：Common Crawl 衍生、开源维基、新闻、论坛（要重度清洗）。
- **书籍/论文**：Books、ArXiv（版权合规尤需注意）。
- **代码**：Git 代码库、包管理生态；语言覆盖 Python/JS/C++/Java/Go 等。
- **对话/指令**：开源指令数据、合成（self-instruct）、转写或产品日志（需严格脱敏与授权）。
- **多语种**：Wikipedia 多语、CCMatrix/CCAligned、新闻平行语料等。
- **高质精选集**：如 RefinedWeb/RedPajama 风格的 curated 混合物。

**2.2 清洗/去重/筛选**

- **去重**：MinHash/SimHash/LSH 进行文档/段落/句子级去重（避免“重复背诵”、降低过拟合）；n-gram 局部去重。
- **质量打分**：语言检测、长度/熵过滤、文体/垃圾识别、模型判别（“critic model” 给质分）。
- **版权/敏感**：许可证过滤、PII（个人信息）脱敏、合规黑白名单。
- **域配比**：通用：代码：高质书面：对话 ≈（例如）**6:2:1:1**（**项目相关**，需 A/B 校准）；多语比例按目标市场与评测分布设定。

**2.3 长上下文友好**

- **拼包（packing）**：将多短文本拼成定长序列并用 **EOD** 特殊 token 分隔，提升利用率。
- **上下文混合**：混合短/中/长序列，后期增加长序列占比（curriculum）。
- **检索数据**：为未来 RAG 做准备，加入“问题+证据段”的拼接样本。



#### 3. 分词与位置编码

- **Tokenizer**：byte-level BPE（Byte Pair Embedding, 合并字符可以**用最少的token来表示语料库**，这也是 BPE 算法的主要目标，即**数据的压缩**。） / SentencePiece（unigram/BPE）。
  - **词表大小**：开源家族常见 32k–100k；多语/代码场景倾向更大词表。
  - **字节友好**：byte-fallback/纯 byte 编码（避免 OOV）。
- **位置编码**：RoPE 为主流；**长上下文**常用 RoPE 变体（NTK、YaRN、Dynamic RoPE 等）。
  - **ALiBi** 等方法也见于部分模型或与 RoPE 结合。



#### 4. 模型与训练策略（自监督学习）

**自监督（self-supervised）**：**标签由数据本身生成**。在 CLM（Causal LM）里，输入是前缀 $x_{1:t-1}$，**标签就是下一个 token $x_t$**。

**4.1 架构与稳定性**

- **架构**：Decoder-only Transformer；SwiGLU FFN、**RMSNorm**（pre-norm）、多头注意力（可能带 **GQA/MQA**）。
- **初始化与缩放**：残差/输出缩放（如 $1/\sqrt{N}$）、SwiGLU 中加入 $\beta=1/\sqrt2$ 缩放；Logit/头部温度稳定 trick。
- **正则化**：Dropout（低）、权重衰减（AdamW 的 wd），Early Phase 可用小量 label smoothing（极少见于 CLM）。
- **MoE**：路由器温度、load-balance loss、capacity factor、top-k experts（常见 top-2 或 top-8）。

**4.2 优化与调度**

- **优化器**：AdamW（β1≈0.9, β2≈0.95–0.98, ε=1e-8/1e-5）、或 Adafactor（少见于超大训练）。
- **学习率**：**预热 + 余弦衰减**（warmup 0.5–2% 训练步数）；峰值 LR 依 batch tokens 与模型规模标定。
- **批量单位**：以 **tokens/step** 为主（例如 1–8M tokens/step）；梯度累积用于“虚拟大 batch”。
- **精度**：混合精度（bf16/fp16）+ Loss-scale；Hopper/Blackwell 上配合 fp8。
- **梯度策略**：Clip（如 1.0）、FSDP/ZeRO + TP/PP（流水线/张量/数据并行）；checkpointing 节省显存。

**4.3 课程学习（curriculum）**

- **长度 curriculum**：先短后长，提高吞吐与稳定；后期加入更长序列以适配长上下文。
- **难度/域 curriculum**：早期偏通用与中等难度，后期增加代码/数学/高质文献与合成推理数据。
- **MoE curriculum**：先小专家/低路由熵，再逐步放开容量与温度。

**4.4 监控与早停**

- **Perplexity（PPL）**：主度量；分域 PPL（英文/中文/代码/长文）同时跟踪。
- **合成小基准**：Few-shot 的词法/句法 sanity checks、短程与长程记忆小测。
- **损失曲线**：训练-验证 gap；过拟合/数据泄露（contamination）检测。

**4.5 多卡并行分布是训练**

**预训练规模律**（Scaling Law）**：以 $C \approx 6ND$ 粗略刻画训练计算量（$C$：FLOPs，$N$：参数量，$D$：训练 tokens）。经验上常据此确定“**算力最优**”的数据量与模型大小配比。

**训练数据规模**：文本中给出两个经验配比：

- OpenAI（示例）：**训练 tokens $\approx 1.7\times N$**（如 175B 需 ~300B tokens）。
- LLaMA 系列经验：**训练 tokens $\approx 20\times N$**（如 175B 用 ~3.5T tokens 以追求更优）。

**算力与时长**：预训练极耗资源；即使 1B 模型也常需多卡集群；文中举例：百亿级模型 ~**1024×A100 训练 1 个月**，十亿级模型 **~256×A100 2–3 天**。

**分布式训练是刚需**：

- **数据并行（DP）**：同一份模型复制到多卡、喂不同 mini-batch，汇总梯度再同步更新（总 batch = 各卡 batch 之和）。
- **模型并行（MP）**：当单卡放不下模型时，把网络按“层/张量切片”分到多卡。
- 在 DP/MP 基础上，演化出 **张量并行（TP）**、**流水线并行（PP）**、**3D 并行（DP+TP+PP）**、以及 **ZeRO（Zero Redundancy Optimizer）** 等更高效策略。

**主流框架**：**DeepSpeed、Megatron-LM、ColossalAI** 等，其中 **DeepSpeed** 的使用最广。

**4.6 DeepSpeed 简介（做什么、怎么做、何时用）**

把“大模型训练与推理”变得**更省显存、更高吞吐、更稳定**，并且易于组装 **DP/TP/PP/ZeRO** 等并行策略。ZeRO 将模型训练阶段每张卡被占用的显存分为两类：

- 模型状态（Model States），包括模型参数、模型梯度和优化器 Adam 的状态参数。假设模型参数量为 1M，一般来说，在混合精度训练的情况下，该部分需要 16M 的空间进行存储，其中 Adam 状态参数会占据 12M 的存储空间。
- 剩余状态（Residual States），除了模型状态之外的显存占用，包括激活值、各种缓存和显存碎片。

关键技术与模块（训练侧）

- **ZeRO 系列（核心）**：显存优化
  - **Stage-1/2/3**：将 **优化器状态、梯度、参数** 在数据并行进程间切分而不是重复保存，显著**省显存**。
  - **ZeRO-Offload / ZeRO-Infinity**：把部分状态/参数**卸载到 CPU / NVMe**，进一步扩大可训练模型规模。
- **3D 并行**：与 Megatron-LM 的 **TP（张量并行）**、**PP（流水线并行）** 组合，实现 **DP+TP+PP+ZeRO** 的大规模并行。
- **Fused 优化器与通信优化**：如 **FusedAdam、1-bit Adam**，梯度压缩/聚合与**通信-计算重叠**，提升吞吐。
- **Activation Checkpointing**：重算激活，**以算换显存**。
- **MoE 支持（DeepSpeed-MoE）**：门控混合专家的高效路由与并行（含均衡损失、容量控制等）。
- **混合精度**：fp16/bf16，支持梯度裁剪、损失缩放。

推理与微调

- **DeepSpeed-Inference**：大模型推理的张量并行、权重量化与 kernel 融合，加速服务端吞吐与延迟。
- **与 PEFT/QLoRA 生态兼容**：SFT/指令微调时可结合 ZeRO 与低比特量化节省资源。

**何时优先考虑 DeepSpeed？**

- 需要**把模型放大到单卡显存放不下**（> 数十亿参数）。
- 希望在**同样 GPU 资源下跑更大的 batch / 更长的序列**。
- 要把 **DP/TP/PP/ZeRO** 以较少工程成本组合起来，并获得成熟的**吞吐与显存优化**。



#### 5. 损失函数与标注

**5.1 主损失：Token 级交叉熵（Cross-Entropy）**

对每个位置 $t$ 的真值 token $y_t$ 与预测分布 $\hat p(\cdot\mid x_{<t})$：
$$
\mathcal{L}_{\text{CLM}} = -\frac{1}{N}\sum_{t=1}^N \log \hat p(y_t \mid x_{<t})
$$

- **label = 下一个 token**（shifted labels）。
- **mask**：只在非 padding/非特殊段计算 loss。
- **可选细节**：
  - **class weighting**：对特殊/罕见 token 加权（很少用）。
  - **label smoothing**：极少量（如 0.1）可抗过拟合，但生成质量影响不确定；主流不加该做法。
  - **温度/entropy 正则**：在合成数据阶段偶有使用，非标配。
- **MoE 额外损失**：路由均衡/负载损失（例如 auxiliary loss with coeff 0.01–0.1）。

**5.2 预训练 vs 对齐（后训练）**

- **预训练**：纯无监督/弱监督（CLM 为主）。
- **对齐阶段**（不属于预训练）：SFT（监督微调）、RM/PRM、RLHF、DPO/IPO、知识蒸馏、工具/RAG 集成等。



#### 6. 典型超参与经验值（供落地参考）

> 以下为“常见落地区间”，实际需结合算力与数据迭代实测。

- **Token 数总量**：中型 200B–1T tokens；大型 2T–10T+。
- **序列长度**：起步 2K–4K，后期 8K–128K；若目标超长（>256K），需在数据与 RoPE 变体上专门设计。
- **词表大小**：32k–100k，开源中文/多语常用 32k–64k byte-level。
- **LR 峰值**：1e-4 到 3e-4（小中模型）；大模型更小（<1e-4），具体看 batch tokens。
- **权重衰减**：0.01（AdamW 常见）。
- **Dropout**：0–0.1（大多偏低）。
- **梯度裁剪**：1.0 左右。
- **SwiGLU β**：$1/\sqrt2$。
- **并行**：FSDP/ZeRO + TP；激活检查点节省显存 30–50%。

------

### 监督微调 SFT

#### 1. 目标与位置

- **目的**：在一个已预训练（多为 CLM）的基座模型上，通过**指令-响应**（或多轮对话）数据，训练其“按要求办事”的能力（遵循格式、礼貌、安全边界、工具调用等）。
- **和预训练的关系**：预训练学“语言+知识+惯常续写”，**SFT 学“该怎么答”**；SFT 通常是对齐流水线的第一步，之后可接 **RM/PRM + RLHF/DPO/IPO** 等偏好对齐。



#### 2. 数据：来源、质量与混合

**2.1 类型**

- 单轮指令→响应（instruction → response）
- 多轮对话（含 system）
- 任务型：代码、数学、推理、检索问答、工具/函数调用（JSON schema）、多模态描述（若是 MLLM）

**2.2 构造与清洗**

- 来源：人工标注、产品日志脱敏、自指令（self-instruct）、过滤后的网络数据、模型生成 + 人工筛选、拒绝采样（rejection sampling）。
- 清洗：去重（文档/段落/n-gram）、语言检测、长度/熵/脏词过滤、PII/版权合规。
- 质量打分：启发式 + 小模型（critic）或 RM 预筛，剔除无效/幻觉/有害样本。
- **混合配比**：通用:代码:数学:检索/工具:安全 ≈ 6:2:1:0.5:0.5（示例，需项目化调参）；中后期可提高困难/稀有域占比（curriculum）。



#### 3. 格式化：模板与分词

**3.1 Chat 模板**

- 统一使用模板（如 ChatML/Alpaca/Llama2/自定义），确保推理与训练一致：

  ```json
  <s>[SYSTEM] {system}\n[USER] {user}\n[ASSISTANT] {assistant} </s>
  ```

- 多轮：按回合拼接；每轮都输出到 ASSISTANT 的末端为监督目标。

**3.2 分词与特殊符号**

- 使用与基座一致的 tokenizer；显式插入 BOS/EOS/分隔符；对**函数调用/JSON**可加“约束前缀”（如 `<tool_call>`）。
- **长上下文**：采用 RoPE 变体的一致配置（见 §7）。



#### 4. 损失与掩码（核心实现）

- **目标**：标准 token-level **Cross-Entropy**（CE）。
- **只对 ASSISTANT 的输出 token 计损失**；**不**对 system/user/分隔符计算。
- **多轮对话**：对每轮的 assistant 段分别打 1，其余打 0（loss mask）。
- **Label shift**：CLM 方式（预测下一个 token），忽略 pad。

**伪代码（关键掩码）**：

```python
# ids: [B, T]; roles: 同形状标注 token 归属（sys/user/assistant）
labels = ids.clone()
loss_mask = (roles == ASSISTANT_ID).int()  # 只监督助手段
# 移位：预测下一 token
input_ids  = ids[:, :-1]
target_ids = labels[:, 1:]
loss_mask  = loss_mask[:, 1:]

logits = model(input_ids, attention_mask=causal_mask)  # [B,T-1,V]
loss_all = F.cross_entropy(
    logits.reshape(-1, V), target_ids.reshape(-1),
    reduction="none", ignore_index=PAD_ID
).view(B, -1)
loss = (loss_all * loss_mask).sum() / (loss_mask.sum().clamp_min(1))
```



#### 5. 训练策略与超参

**5.1 优化**

- **优化器**：AdamW（β1≈0.9, β2≈0.95–0.98, ε=1e-8/1e-5），wd 0–0.1（常 0.01 或更小）。
- **学习率**：warmup（0.5–2% 步）+ cosine decay；全参微调 LR 常 5e-6 ~ 2e-5；LoRA/QLoRA 可高 1e-4 ~ 3e-4。
- **梯度**：clip 1.0；混合精度（bf16/fp16）；梯度累积增大有效 batch。
- **批量度量**：以 **tokens/step** 看吞吐；packing（同批拼文档）提升利用率。

**5.2 轮数与早停**

- 1–3 epoch 常见；以**验证集 CE/PPL** 与**指令跟随集评分**（如 MT-Bench/自建 Rubric）早停，避免过拟合与“口吻僵化”。

**5.3 Curriculum**

- 前期多通用/短指令，后期加入困难/长输入/工具调用/多轮。
- 对代码/数学可最后 20–30% 步提高配比。



#### 6. 参数高效微调（PEFT）

**6.1 LoRA**

- 目标模块：注意力的 q_proj/k_proj/v_proj/o_proj + FFN 的 up/gate/down（常至少 q,v,o,gate/up/down）。
- 超参：`r`=8–64，`alpha`=16–64，dropout 0–0.1；冻结除 LoRA 以外的权重。
- 优点：显存/存储省；可叠加多个任务的 LoRA 适配器。

**6.2 QLoRA**

- 4-bit (nf4) 量化 + LoRA；节省显存，适合 30B+ 模型单机/少机训练。
- 注意使用 **paged optimizers**、梯度检查点、微调时的稳定性（β2 合理、loss-scale）。

**6.3 其他**

- Prefix-/P-Tuning、Adapter、BitFit（只训偏置）——较少用在高质量指令微调但可做对比。



#### 7. 长上下文 SFT

- **数据**：构造长输入样本（文档+问题、RAG 拼接证据），按真实使用拼接；避免“泄漏答案”。
- **位置编码**：与基座一致（RoPE/NTK/YaRN 等）；若要**外推**，训练末期增大长样本比例、对 RoPE 做合适伸缩。
- **策略**：混合长度 curriculum；采样时控制长样本配额（如 20–40%）。



#### 8. 工具/函数调用与结构化输出

- **标注**：把工具调用视作 **assistant 的一段受监督文本**（JSON/函数签名），或使用特殊 tags：`<tool_call>{"name":..., "args":...}</tool_call>`。
- **解码约束**：可用正则/JSON schema/`guided decoding`，训练时提供**成功案例**与**失败/修复案例**。
- **损失**：依旧是 CE；必要时对 JSON 关键字段加权（可选）。



#### 9. 安全对齐与拒绝策略（SFT 阶段）

- 加入**安全样本**（拒答模版、降级回复、说明政策原因），覆盖暴力、仇恨、隐私、违法等场景。
- 标注**拒绝风格与可替代建议**，避免空拒绝。
- 监控 **toxicity**、**leakage**、**jailbreak 成功率**；对“可被越狱”样本加回训练（对抗式数据）。



#### 10. 评测与监控

- **在线度量**：val CE/PPL、指令集准确率/遵循度（rule-based 打分）、多轮一致性。
- **离线基准**：MT-Bench/AlignBench、Arena-Hard、Code（HumanEval/MBPP）、Math（GSM8K/MATH）、多语言能力。
- **A/B**：对话落地 KPI（采纳率/完成率/满意度）、安全触发率、工具调用成功率。



#### 11. 常见坑

- **未做 loss mask** 导致模型去拟合 user/system；口吻“复述用户话”。
- 训练/推理模板不一致，推理时效果崩。
- 过度拟合单一风格/安全条款；回答僵硬。
- 只训短样本，长上下文退化。
- 工具/JSON 缺少失败样例，推理时脆弱。
- LoRA 目标模块过少/过多导致欠拟合/过拟合；`r/alpha` 不合理造成不稳定。



#### 12. 最小可用实现（HuggingFace TRL 样式）

```py
from trl import SFTTrainer
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("your-base", torch_dtype="bfloat16")
tok   = AutoTokenizer.from_pretrained("your-base")
tok.pad_token = tok.eos_token

# 数据格式化：返回一串已经按模板拼好的字符串；只在 [ASSISTANT] 段监督
def format_example(ex):
    # ex: {"system":..., "messages":[{"role":"user","content":...}, ... , {"role":"assistant","content":...}]}
    # 输出一个拼接后的字符串
    return template_render(ex)

peft_config = LoraConfig(
    r=32, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tok,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    peft_config=peft_config,
    max_seq_length=4096,
    packing=True,                             # 训练侧 packing
    formatting_func=lambda ex: format_example(ex),
    dataset_num_proc=8,
    args=dict(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        learning_rate=2e-4,                   # LoRA 常用量级
        weight_decay=0.01,
        num_train_epochs=2,
        lr_scheduler_type="cosine",
        warmup_ratio=0.02,
        bf16=True,
        logging_steps=20,
        save_steps=1000,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        evaluation_strategy="steps",
        eval_steps=1000
    )
)

trainer.train()
```

> 若自己写训练循环，请务必实现 **assistant-only loss mask**（上面的 §4 伪代码）。

#### 13. 怎么看待 SFT vs RL 的“分工”

- **SFT（Supervised Fine-Tuning）**
  - **作用**：把“怎么回答/按什么格式/如何调用工具”**直接示范**给模型；最快速、最稳定地把模型从“会续写”变成“会按指令办事”。
  - **优点**：稳定、易训练、对数据质量最敏感（好数据→好行为）、成本相对可控。
  - **局限**：难以显式优化“偏好/安全/简洁/思考深度”等指标；遇到冲突目标时容易学到折中风格。
- **RL 阶段（广义，含 RLHF/RLAIF/在线RL、过程奖励PRM、o1/o3/R1风格）**
  - **作用**：**在 SFT 基础上**，用**奖励**把模型往“更合用户偏好/更强推理/更安全”方向推；可优化“最终正确率、工具成功率、对话偏好”等**不可微**目标。
  - **优点**：能“拉高天花板”（数学/代码/多步推理、拒绝策略、简洁度等），还能做**在线探索**与自我提升。
  - **代价/风险**：训练复杂、昂贵；**奖励错配/过拟合**会带来**reward hacking**、知识退化或风格异常；需要高质量 RM/PRM 与在线监控。

> 现实里：SFT 是“地基”，RL 是“高层装修与精调”。两者互补，而非替代。

#### 

<br>

## Appendix

### 为什么用 Cross-Entropy?

**0）似然函数（可选了解，数学概念）**

在概率统计里，如果你有一个**参数化模型** $p_\theta(x)$，$\theta$ 是待学习的参数。给定观测数据集 $\mathcal{D}=\{x^{(1)}, x^{(2)},...,x^{(n)}\}$，**似然函数**定义为：
$$
L(\theta;\mathcal{D}) = \prod_{i=1}^n p_\theta(x^{(i)})
$$
它表示 **在模型参数 $\theta$ 下，观测到这些数据的“可能性”大小**。

- **极大似然估计 (MLE)**：寻找能让真实数据概率最大化的参数
  $$
  \hat{\theta} = \arg\max_\theta L(\theta;\mathcal{D})
  $$

但是再实际优化里更常用 **对数似然 (Log-Likelihood)**：
$$
\ell(\theta) = \log L(\theta;\mathcal{D}) = \sum_{i=1}^n \log p_\theta(x^{(i)})
$$
原因：

1. 乘积变加和，数值更稳定。
2. 单调性不变：最大化似然 $\equiv$ 最大化对数似然。

训练目标里我们希望**最小化损失**，所以把最大化 $\ell(\theta)$ 换成最小化它的负数：
$$
\mathcal{L}_{NLL}(\theta) = -\ell(\theta) = -\sum_{i=1}^n \log p_\theta(x^{(i)})
$$
这就是 **负对数似然 (Negative Log-Likelihood, NLL)**。为什么 CE = NLL？考虑分类或语言建模场景：

- 数据样本：输入 $x$，真实标签 $y$（通常是一个 one-hot 分布）。
- 模型输出：一个预测概率分布 $p_\theta(\cdot|x)$。


对真实分布 $q(y|x)$ 与模型分布 $p_\theta(y|x)$，交叉熵定义：

$$
H(q, p_\theta) = - \sum_{y} q(y|x) \log p_\theta(y|x)
$$
若 $q(y|x)$ 是 one-hot（即真实标签），就退化为：
$$
H(q, p_\theta) = - \log p_\theta(y_{\text{true}}|x)
$$


**1. 设定与记号**

- $z\in\mathbb{R}^C$：logits
- $\hat y=\mathrm{softmax}(z)$，$\hat y_i=\frac{e^{z_i}}{\sum_k e^{z_k}}$
- $y\in\{0,1\}^C$ 为 one-hot 真值
- 词表大小 $V=C$（LLM 常见 $3\! \sim\!20$ 万）


**2. 两种损失的定义**

**MSE（对概率）**
$$
L_{\mathrm{MSE}}=\frac{1}{C}\sum_{i=1}^C(\hat y_i-y_i)^2
$$
**CE（交叉熵 / 负对数似然）**
$$
L_{\mathrm{CE}}=-\sum_{i=1}^C y_i\log\hat y_i
$$
如果是one hot 编码，CE只有当真实值才有贡献
$$
L_{\mathrm{CE-OH}}=-\log\hat y_j, \text{   }y_j = 1, y_{other}=0
$$



**3. 梯度推导与“雅可比在哪儿”**

**3.1** 先对 $\hat y$ 求导（不涉及logits, 故也不涉及softmax雅可比）

- **MSE：**

$$
\frac{\partial L_{\mathrm{MSE}}}{\partial \hat y_j}
=\frac{2}{C}(\hat y_j-y_j)
\quad\Rightarrow\quad
\nabla_{\hat y}L_{\mathrm{MSE}}=\frac{2}{C}(\hat y-y)
$$

- **CE：**

$$
\frac{\partial L_{\mathrm{CE}}}{\partial \hat y_j}
=-\frac{y_j}{\hat y_j}
\quad\Rightarrow\quad
\nabla_{\hat y}L_{\mathrm{CE}}= -\,\frac{y}{\hat y}\;
$$

> 这一步都**没有** softmax 的雅可比，因为还没把 $\hat y$ 写成 $\hat y(z)$。

**3.2** 再对 $z$ (logits) 求导（此时必须乘上 softmax 雅可比）

softmax 的雅可比：
$$
J=\frac{\partial \hat y}{\partial z}
=\mathrm{diag}(\hat y)-\hat y\hat y^\top
$$

- **MSE（对 $z$ 的梯度），要使用链式法则：**

$$
\nabla_{z}L_{\mathrm{MSE}}
=\Big(\frac{\partial L}{\partial \hat y}\Big)^\top\!\!J
=\frac{2}{C}\,J(\hat y-y)
$$

> 这里**明确出现**了雅可比 $J$。

- **CE（对 $z$ 的梯度），要使用链式法则：**

$$
\nabla_{z}L_{\mathrm{CE}}
=\Big(\frac{\partial L}{\partial \hat y}\Big)^\top\!\!J
=\left(-\,\frac{y}{\hat y}\right)^\top\!J
=\hat y-y
$$

> CE 的结果**简化**为 $\hat y-y$。直观理解：$-y/\hat y$ 与 $J$ 的结构在链式法则中“对消/合并”，所以最终**看不到显式的 $J$**。这样看就是线性、简洁、数值稳定，**即便预测很错时也有强信号**推动收敛。
>
> - 如果预测概率 $p$ 太小（比真实 y 低），梯度为负，推动 logit 上升。
> - 如果预测概率 $p$ 太大（比真实 y 高），梯度为正，推动 logit 下降。

**4. 大词表下的梯度量级对比（LLM 关键）**

训练初期常见 $\hat y\approx \mathrm{Uniform}(1/V)$。

- **CE：**
   正类 $t$ 的分量：$\hat y_t-1\approx \tfrac{1}{V}-1\approx -1$（**常数量级，不消失**）；
   负类分量：$\hat y_j\approx 1/V$，整体仍有可观合力。

- **MSE：**
   先有 $(\hat y-y)$（正类 $\approx -1$，负类 $\approx 1/V$），再乘 $J$；而
  $$
  J_{ii}\approx \hat y_i(1-\hat y_i)\approx \tfrac{1}{V},\quad
  J_{ij}\approx -\tfrac{1}{V^2}\ (i\neq j)
  $$
  于是
  $$
  \nabla_{z}L_{\mathrm{MSE}}
  =\frac{2}{C}\,J(\hat y-y)
  \quad\text{量级}\ \sim\ \mathcal{O}(1/V)
  $$
  结论：**在 $V$ 很大时，MSE 的对-logits 梯度被 $J$ 再压小一层，训练非常慢**；而 CE 的对-logits 梯度为常数量级。



**5. 统计目标与评测一致性**

- 语言建模等价于**最大似然**：最大化真实下一个 token 的概率；
   **CE 正是负对数似然**，与任务假设完全匹配。
- 评测指标 **Perplexity（PPL）** 与 CE 一一对应：$\mathrm{PPL}=\exp(\text{CE})$。
   ⇒ **用 CE 训练就直接最小化 PPL**；MSE 与 PPL 不严格对齐。


**6. 优化性质与几何（更快更稳）**

- **CE 的 Hessian 与 Fisher 信息相关**，天然适配 softmax 分类的几何；
- 在正确类概率接近 1 的“边界区”，CE 仍能给出合适的更新结构（对错类按概率线性惩罚）；
- **MSE 额外乘 $J$**，使得本就小的更新进一步缩小，更易停滞。

7. **自回归和多分类两者关系**

- **自回归**：强调 **序列生成方式**（从左到右，递归依赖历史）。
- **分类**：强调 **单步预测目标**（softmax over vocab，CE loss）。

换句话说：

- 从整体看：语言模型是 **自回归模型**（序列生成的 factorization）。
- 从单步训练看：它是 **大规模多分类任务**（在词表上选下一个 token）。

**总结： 为什么不用MSE？**

- **目标对齐**：CE = 负对数似然，直接最优 PPL；MSE 与语言建模目标不完全匹配。
- **梯度效率**：CE 对 logits 的梯度 $\hat y-y$ 为常数量级；MSE 需乘 $J$，在大词表下 **$\mathcal{O}(1/V)$**，学习极慢。
- **优化与稳定**：CE 具有更好的几何与曲率性质；MSE 容易在边界处动力不足。
- **工程生态**：训练/评测/解码工具链都以 CE 为中心。

<br>

### 归一化函数对比

| 方法          | 数学形式                          | 归一化目标       | 输出性质              | 能否解释为概率分布 |
| ------------- | --------------------------------- | ---------------- | --------------------- | ------------------ |
| **Softmax**   | $p_i = e^{z_i}/\sum_j e^{z_j}$    | 总和归一化       | $\ge 0, \sum p_i = 1$ | ✅ 可以             |
| **LayerNorm** | $(x - \mu)/\sigma$                | 调均值=0, 方差=1 | 正负均可，和≠1        | ❌ 不行             |
| **RMSNorm**   | $x / \sqrt{\tfrac{1}{d}\sum x^2}$ | 调 RMS=1         | 正负均可，和≠1        | ❌ 不行             |

<br>

### SFT 7问

#### 1） 为什么要做微调（SFT）？指令数据集如何构建？

**目的**：把“会续写”的预训练模型，变成“**会按指令办事**、会按固定格式/安全规则回答”的助手。
 **构建步骤（示意流程）**：

- **需求拆解**：列出目标能力/场景（对话、摘要、检索问答、代码修复、工具调用、JSON 输出、安全拒绝等）。
- **模板统一**：固定对话格式（如 system/user/assistant），训练与推理保持一致。
- **获取数据**
  - 人工标注的高质量示范（gold demos）。
  - **自举/合成**：self-instruct、用强模型生成→人工审核。
  - 产品日志**脱敏**后的高分样本。
  - 失败样本与“拒绝示范”（安全）。
  - 工具调用/函数调用样本（含成功与失败案例）。
- **清洗与对齐**：去重（文档/段落/n-gram）、语言检测、长度/熵阈、敏感信息过滤、版权/合规审计。
- **分层混合**：按能力域与难度分 bucket，设配比（例如 通用:代码:数学:检索:安全 ≈ 6:2:1:0.5:0.5），后期增加难样本与长输入占比（curriculum）。
- **打标**：只把 **assistant 段**标为监督目标（见 Q7），并给出工具/JSON 的结构化标签。


#### 2）SFT 的核心价值？与继续预训练（CPT）有何不同？

- **SFT（监督微调）**：目标是**行为对齐**与**任务执行**（格式、礼貌、拒绝、工具链），损失是 **CE** 仅作用于 **assistant** 输出。
- **继续预训练（CPT）**：仍是 **CLM**，在新增语料（行业/语言/代码）上“补知识/风格”，不直接教“按指令办事”。
   **选型**：
- 想让模型**更懂领域内容** → 先做 CPT；
- 想让模型**按要求完成任务/安全合规** → 做 SFT；
- 实务中常 **CPT → SFT →（DPO/RL…）** 逐级对齐。


#### 3）为什么“指令多样性”至关重要？举例说明其影响

**原因**：模型要学会“**同一意图的多种表达**”，才能在真实世界的长尾指令中泛化。
 **例**（同一任务“写 Python 排序函数”）：

- “请实现一个函数，对列表升序排序；返回新列表。”
- “用递归重写排序；时间复杂度与边界条件说明。”
- “给我可粘贴到 Jupyter 的最小示例，包含输入/输出样例。”
- “请先给伪代码，再给 Python，可处理重复元素与空输入。”
   若数据只覆盖第一种说法，模型会对其它表述“懵”；多样化训练能显著提升**鲁棒性与任务覆盖**。


#### 4）如何系统评估“有用性（Helpful）”和“无害性（Harmless）”？

**有用性**：

- **离线基准**：MT-Bench/AlignBench、任务专项（Summ，QA，Code/Math）、长上下文（LongBench）。
- **偏好对比**：A/B 对战或 **pairwise** 标注，训练/校准 **Reward Model**，计算 **win-rate**/**helpfulness score**。
- **结构化/工具成功率**：JSON 合法率、工具调用成功率、端到端完成率。
- **可读性/简洁度**：规则打分 + 人评抽检。

**无害性**：

- **安全集**：红队集合（暴力、仇恨、隐私、违法、医疗/金融误导等），统计**安全拒绝正确率**与**误拒率**。
- **自动检测**：toxicity/PII/NSFW 检测器 + 关键词/正则。
- **越狱评估**：jailbreak 成功率、prompt injection 防御率。
- **在线指标**：审核命中率、申诉率、风险分布变化。


#### 5）只用“小而精”SFT 与“中低质大规模”SFT 的取舍？

- **小而精（几千～几万条）**

  - **优点**：风格稳定、指令遵循好、安全边界清晰、可控。
  - **缺点**：**覆盖面窄**，遇到新表述/长尾任务易失效；上限受限。

- **大规模中低质**

  - **优点**：覆盖广、通用性更强、对长尾更友好。
  - **缺点**：噪声与幻觉上升、风格漂移、安全不稳。
     **折中做法**：以“**高质核**（nucleus，规模较小，但**质量极高**、**风格稳定**、**任务定义清楚**的示范数据。常来自人工精标或强模型+严格人工审核。）”为主，配少量筛过的“**广覆盖尾部**”（量更大，**覆盖主题/语言/表达方式的长尾**，质量虽不及“核”，但**已过清洗与粗校**。）；使用 **温度**（**T<1**更“贪心”高质量，**T>1** 更“平均”与探索。）/权重**（例如核:尾部:代码:数学:安全 = 5 : 2 : 1 : 1 : 1）采样 与 **curriculum**，并用强评测/人审把关。

- 课程学习：

  - **按阶段逐步改变数据配比与难度**，让模型先稳，再强。

    一个实用三阶段范式（举例）：

    1. **阶段 A（收敛稳态）**：高质核占比高（如 60–80%），短输入、基础任务、标准格式、安全拒绝。
    2. **阶段 B（能力扩展）**：降低核占比（40–60%），加入更多尾部难例：长上下文、工具失败/修复、复杂推理、多轮对话。
    3. **阶段 C（冲上限）**：核占比 20–40%，加大代码/数学/检索拼接/极长文；引入更严格格式（JSON schema）与对抗样本。


#### 6）SFT 阶段，模型为何能学会预训练没见过的特殊 token（如 `<|user|>`）？

- 这些 token 的**词向量会在 SFT 时被随机初始化**并参与前向；虽然**不在监督端（无直接 CE）**，但它们出现在 **输入序列**里，**通过注意力**影响对下一个输出 token 的预测，因此**会收到梯度反传**（来自 assistant tokens 的损失）。
- 直观上：模型为了“预测正确的下一 token”，需要区分“谁在说话/段落边界/指令位置”，于是把 `<|user|>`、`<|assistant|>` 等学习成**语义分隔/角色提示**的向量。


#### 7）为什么训练时通常只对 **Answer** 计损失、忽略 **Instruction**？

- **避免“复述用户”**：若对 user/system 也计损失，模型会被训练成去“还原”提示，而不是**回应**提示。
- **保持指令不变性**：不同措辞/模板都应得到**稳定输出**；不对指令计损失，能减少对特定措辞的过拟合。
- **符合生成因果性**：CLM 的目标是**预测下一 token（即助手回复）**；输入部分只提供条件，不应成为监督目标。

**注意**：虽然不对 instruction 计损失，但它们仍**参与前向与反传**，其嵌入会被更新（见 Q6）。