# LoRA Introduction

[TOC]



## 什么是 LoRA？

LoRA (Low-rank adaptation)用一个**低秩矩阵分解**去近似权重更新：把原权重 $W$ 的更新限制为 $\Delta W = B A$，其中 $A \in \mathbb{R}^{r \times d_{\text{in}}}$、$B \in \mathbb{R}^{d_{\text{out}} \times r}$、秩 $r \ll \min(d_{\text{in}}, d_{\text{out}})$。训练时**冻结**原始权重 $W_0$，只训练 $A,B$，推理时用
$$
W = W_0 + \alpha \cdot \frac{B A}{s}
$$
其中 $\alpha$ 是缩放系数（常见 8 或 16），$s$ 通常取 $r$ 用于归一化。其中A（$d_{in}\times r$）, B($r \times d_{out}$), 这样

**参数量**从全量 $d_{\text{out}}\!\times\! d_{\text{in}}$ 变成 $r(d_{\text{in}}+d_{\text{out}})$​，显著降低显存与计算。下面提供一个最小可感的 LoRA 例子：把一个**4×5 的线性层**（共 20 个参数）做 LoRA 适配，秩 $r=2$。

- 基座权重 $W_0 \in \mathbb{R}^{4\times 5}$（20 个参数，**冻结**不训练）
- LoRA：$\Delta W = B A$，其中 $A\in\mathbb{R}^{2\times 5}$，$B\in\mathbb{R}^{4\times 2}$
- 缩放：$W = W_0 + \frac{\alpha}{r}\, B A$，取 $\alpha=16, r=2 \Rightarrow \alpha/r=8$
- 训练参数量：$r(d_{in}+d_{out})=2(5+4)=18$（只训练 $A,B$ 这 18 个参数）

**基座权重 $W_0$（4×5）：**
$$
W_0=\begin{bmatrix}
1 & 0 & -1 & 2 & 0.5\\
0 & 1 & 0 & -1 & 1\\
0.5 & -0.5 & 1 & 0 & 0\\
1.5 & 0 & 0 & 1 & -0.5
\end{bmatrix}
$$
**LoRA 矩阵 $A$（2×5）与 $B$（4×2）：**
$$
A=\begin{bmatrix}
0.2 & -0.1 & 0 & 0.3 & -0.2\\
0 & 0.1 & 0.4 & -0.2 & 0.1
\end{bmatrix},\quad
B=\begin{bmatrix}
0.5 & 0.1\\
-0.3 & 0.2\\
0 & -0.4\\
0.2 & 0.3
\end{bmatrix}
$$
AB通过训练得出。这样，其实一共只更新了18个参数将原来20个参数的矩阵都更新了。先算 $\Delta W = BA$（4×5）：
$$
\Delta W=\begin{bmatrix}
\;0.10 & -0.04 & \;\,0.04 & \;\,0.13 & -0.09\\
-0.06 & \;\,0.05 & \;\,0.08 & -0.13 & \;\,0.08\\
\;\,0 & -0.04 & -0.16 & \;\,0.08 & -0.04\\
\;\,0.04 & \;\,0.01 & \;\,0.12 & \;\,0 & -0.01
\end{bmatrix}
$$
得到**适配后权重**（放大 8 倍再加到 $W_0$）：
$$
W = W_0 + 8\Delta W =
\begin{bmatrix}
1.8 & -0.32 & -0.68 & 3.04 & -0.22\\
-0.48 & 1.4 & 0.64 & -2.04 & 1.64\\
0.5 & -0.82 & -0.28 & 0.64 & -0.32\\
1.82 & 0.08 & 0.96 & 1 & -0.58
\end{bmatrix}
$$
在前向阶段，我们假设输入：
$$
x=\begin{bmatrix}1\\-1\\0.5\\0\\2\end{bmatrix}
$$
那么LoRA 输出：$y=W x=\begin{bmatrix}1.34\\1.72\\0.54\\1.06\end{bmatrix}$

<br>

## 为什么用 LoRA？

- **显存友好**：只训练小矩阵，梯度与优化器状态也只为它们分配显存。参数占用全量微调的0.01%-0.5%；我们来看数字，对一个线性层 $W\in \mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$：
  $$
  \text{LoRA 比例} \approx \frac{r(d_{\text{in}}+d_{\text{out}})}{d_{\text{in}}d_{\text{out}}}
  $$
  对 Transformer（常见做法：只给注意力投影层 $W_q,W_k,W_v,W_o$ 加 LoRA，且这些矩阵都是 $d\times d$）：
  $$
  \text{LoRA 比例（仅注意力）}\ \approx\ \frac{4\cdot r(2d)}{4\cdot d^2}\ =\ \frac{2r}{d}
  $$
  若连 FFN 两个大矩阵（$d\times d_{\text{ff}}$ 与 $d_{\text{ff}}\times d$，常见 $d_{\text{ff}}\!\approx\!4d$）也加上，比例大约再增加 $\approx \frac{1.25\,r}{d}$。
   **合计粗略：**
  $$
  \text{注意力+FFN 全加}\ \approx\ \frac{(2+1.25)\,r}{d}\ =\ \frac{3.25\,r}{d}
  $$
  以 7B 级模型常见 $d=4096$ 为例：

  - **只加注意力层**
    - $r=1$: $\frac{2}{4096}\approx 0.049\%$
    - $r=2$: $\approx 0.098\%$
    - $r=8$: $\approx 0.39\%$
  - **注意力+FFN 都加**
    - $r=1$: $\frac{3.25}{4096}\approx 0.079\%$
    - $r=2$: $\approx 0.16\%$
    - $r=8$: $\approx 0.63\%$

  以 70B 级（例如 $d=8192$）：

  - **只加注意力层**

    - $r=2$: $\frac{2\cdot 2}{8192}\approx 0.049\%$
    - $r=8$: $\approx 0.20\%$

    

- **训练效率较高（轻量化）**：全参耗费时间，需要大量资源和时间。在GPT3上，0.01%-0.1%参数的lora微调即可达到95%-99%的性能。

- **多任务/多域切换方便**：同一个基座权重配不同 LoRA 适配器即可热插拔。很多模型可能会出现训练奥的模型在一个新任务上继续学习或者全量微调后，原来的任务性能下降（灾难性遗忘）。LoRA通过冻结原来的矩阵W0，保留了原始能力，可以随时卸载新的训练好的模块，回退回去，不破坏基座能力。

  - **回退可靠**：只要不合并权重（不把 $\Delta W=BA$ 写回 $W_0$），卸载适配器就彻底回到老模型。

    **多适配器并存**：同一基座可维护一个“适配器库”，按任务加载；或做**适配器融合/加权**。AB团队也可以使用LoRA在同一基座下学习和执行不同任务。

    **分阶段训练**：SFT 用适配器 A，RLHF/DPO 用适配器 B，需要时分别加载，互不污染。

- **可与量化结合（QLoRA）**：基座权重量化到 4bit/8bit，保持可训练的 LoRA 层为 FP16/BF16。

- **工程简单**：无需改动推理张量形状，易于并入现有部署。

<br>

## 作用位置与实现要点

典型地给 **自注意力的投影层**（$W_q, W_k, W_v, W_o$）和/或 **MLP/FFN 的输入输出投影**加 LoRA。
常见做法：对某个线性层 $y=W x$ 改为
$$
y=(W_0 + \alpha \tfrac{BA}{s})x = W_0 x + \alpha \tfrac{B(Ax)}{s}
$$
这样实现上只需在前向里加一条旁路（adapter）。

**选择哪些权重加 LoRA（target modules）**：

- 仅 $W_q,W_v$：开销最小，常能拿到 80–95% 的全参效果。（原论文就是这样做的，即分别作用于Q关注点信息，V输出的内容，性价比最高，已经接近全量微调）
- $W_q,W_k,W_v,W_o$：更稳定，尤其是复杂指令或生成质量需求高时。
- 再加 FFN：有助于跨领域风格/知识迁移，但开销上升。（一般不修改，但是取决于任务实际测试效果）

<br>

## 关键超参数（经验起点）

- **rank $r$**：8–64（7B 取 8–16；70B 可到 32–64）。
- **alpha**：等于或略大于 $r$（1-2倍，如 $r=16,\ \alpha=16$ 或 32）。
- **dropout**：0.05–0.1，减少过拟合与“过度适配”。
- **学习率**：LoRA 层 1e-4～2e-4（AdamW/AdamW8bit），基础 LR scheduler 用 cosine/linear decay。
- **权重衰减**：0 或 0.01（LoRA 参数通常不需要大 WD）。
- **微调步数**：和数据规模相关；指令微调常见 1–3 个 epoch。

<br>

## 训练流程（监督微调 SFT 场景）

1. **加载基座模型**（可 4/8bit 量化）。
2. **插入 LoRA 层**到选定的线性层（如 q_proj/v_proj）。
3. **冻结其余参数**，仅优化 LoRA。
4. **数据**：对齐到任务（指令-输出、领域语料等），混入少量高质量样本往往胜过大量噪声。
5. **训练**：混合精度（BF16/FP16），梯度累积配合小显存。
6. **评估与导出**：
   - 导出为“基座 + 适配器”（便于切换）。
   - 或**合并权重**（将 $BA$ 合并回 $W_0$ 生成单一权重，便于纯推理环境）。

<br>

## 改进

### LoRA+

对LoRA的A/B矩阵设置不同的学习率，加速收敛

通常靠近输出的B矩阵学习率$l_B$远高于$l_A$（4-16倍之间），这可能因为神经网络中靠近输出的权重对梯度变化更加名干，需要更大调整，而靠近输入的权重应该更加稳定。

### DoRA

把幅值与方向解耦，往往更稳定。

### QLoRA 简述（量化 + LoRA）

- **做法**：把基座权重量化到 4bit（如 NF4）或 8bit，仅 LoRA 层保持 FP16/BF16 训练。
- **好处**：把 7B/13B 的 SFT/RL 成本压到单卡 24–48GB 可承受范围。
- **注意**：量化感知训练会对学习率更敏感；建议使用 paged optimizers 和更小的全局 batch，梯度裁剪 0.5–1.0。
- 核心是：把**基座模型权重**做 4-bit（nf4）量化并冻结，只训练插在某些线性层上的 **LoRA 低秩矩阵**（rank=r），再配合缩放系数（α）和 LoRA dropout 控制容量/正则。优点：显著降显存（单卡 16 GB 就能训 7B/8B），训练稳定、可拔插回退 。

<br>

## 什么时候不太合适？

- **需要大幅度结构性改变**或**跨模态大迁移**时，仅靠低秩更新可能不足。
- **极低秩**在高复杂任务上会出现欠拟合；增大 $r$ 或覆盖更多层。
- **偏好优化后退化**：RL 阶段可能“毁掉”SFT 能力，可采用多适配器分工或引入 KL/约束技巧。

<br>

## 评估与部署要点

- **评估**：除通用指标外，做 **Catastrophic Forgetting 检查**（原任务集 vs 新任务集）。
- **合并 vs 适配器**：
  - **线上多租户/多域**：保留适配器热插拔。
  - **边缘/嵌入式/纯推理**：合并权重简化依赖与延迟。
- **并行**：LoRA 前向额外开销小；TP/PP/DP 策略与基座一致即可。
- **安全**：对齐任务建议独立适配器，避免互相污染。

<br>

## 常见坑 & 调参清单

- **过拟合**：提高 LoRA dropout、早停、混入少量通用语料。
- **学不动**：稍增 $r$ 或 alpha，放宽 target modules（加上 FFN/out_proj），增大学习率上限。
- **量化不稳**：切到 8bit 先确认收敛，再回 4bit；确保梯度裁剪。
- **推理慢**：合并权重或把 $BAx$ 实现成 fused-op（很多框架已支持）。
- **遗忘**：用多适配器（SFT 与 RL 分开），或在 RL 里加 KL/Ref 模型约束。

<br>

## 代码手撕LoRALinear
可见`lora_mian.py`, LoRA核心代码如下：
```python
class LoRALinear(nn.Module):
    """
    LoRA-化的 Linear：
      y = base(x) + scale * B(A(x)),  其中A: in->r, B: r->out, scale = alpha / r
    - 冻结 base 权重（W0），只训练 A/B（以及可选的 LoRA dropout）
    - 提供 merge/unmerge：把增量合并到 base.weight 中，或恢复为可插拔模式
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        bias: bool = True,
        init_B_to_zero: bool = True,
    ):
        super().__init__()
        assert r > 0, "rank r must be > 0"
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.scale = alpha / r
        self.merged = False

        # 基座
        self.base = nn.Linear(in_features, out_features, bias=bias)
        for p in self.base.parameters():
            p.requires_grad_(False)  # 冻结基座

        # LoRA 旁路
        self.A = nn.Linear(in_features, r, bias=False)
        self.B = nn.Linear(r, out_features, bias=False)

        # 初始化：A 随机小值，B=0 可使初始增量为 0（输出=基座输出）
        nn.init.kaiming_uniform_(self.A.weight, a=5**0.5)
        if init_B_to_zero:
            nn.init.zeros_(self.B.weight)
        else:
            nn.init.kaiming_uniform_(self.B.weight, a=5**0.5)

        self.lora_dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    @property
    def deltaW(self):
        # 返回当前低秩增量矩阵（out x in）
        # B.weight: [out, r], A.weight: [r, in]
        return self.B.weight @ self.A.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            # 已经把 LoRA 合并进 base.weight，直接走 base
            return self.base(x)
        # 可插拔模式：基座 + 低秩旁路
        out = self.base(x)
        out = out + self.scale * self.lora_dropout(self.B(self.A(x)))
        return out

    @torch.no_grad()
    def merge(self):
        """
        把 LoRA 的增量合并进 base.weight：
          base.weight += scale * (B.weight @ A.weight)
        合并后可以仅保留 base 以简化部署（注意将失去“可插拔/可回退”）。
        """
        if self.merged:
            return
        # 合并权重
        self.base.weight += self.scale * (self.B.weight @ self.A.weight)
        # 若有 bias 的 LoRA 变体可在此处合并 bias
        self.merged = True

    @torch.no_grad()
    def unmerge(self):
        """
        从 base.weight 中减去已合并的增量，恢复到“可插拔”模式。
        """
        if not self.merged:
            return
        self.base.weight -= self.scale * (self.B.weight @ self.A.weight)
        self.merged = False
```




## Appendix

### 为什么权重矩阵可以做低秩分解？

1. 从奇异值分解得到的启发-SVD，可以把人以矩阵分解成三个矩阵的乘积

2. LoRA 提出“**适配时权重变化的内在秩很低**”的假设，并用大量实验验证：很多层只要很小的秩（r=1、2、8）就能逼近全量微调效果（但不是相等），说明 $\Delta W$ 的奇异值谱快速衰减、可被低秩近似。

3. **结论**：我们没有一个对所有任务都成立的“ΔW 必低秩”的硬性定理；LoRA更多建立在**强实证** + **低维/低秩结构的理论直觉**上（内在维度、梯度外积结构、线性化/NTK），以及**LoRA 本身的表达力分析**。

   **工程含义**：当 r 较小仍能匹配全参微调的 90%–100% 表现时，就说明该层/该任务下的 $\Delta W$ **在可接受误差内是低秩可近似的**；若表现不够，增大 r、覆盖更多层或采用变体（如自适应秩/局部低秩）即可。

### 和MoE关系？

**核心类比**

- **LoRA = 低秩“插片”**：在固定的基座权重上，加一小块可训练的低秩更新（每层一个小旁路）。**参数很少、训练便宜、推理时仍是密集计算**。更像“给同一台发动机换不同喷嘴”。
- **MoE = 稀疏多专家**：把网络的某些层换成“很多并行专家”，由**门控器（router/gating）**对每个token选择少数几个专家执行。**参数总量巨大，但每个token只激活少量专家，推理是稀疏计算**。更像“多台发动机，按需点火”。

**关键差异一览**

- **目标**
  - LoRA：高效**适配/微调**已有模型到新任务/新域；支持热插拔，缓解遗忘。
  - MoE：在**相同计算预算**下扩大模型**总参数与表达力**，提升上限。
- **参数与计算**
  - LoRA：新增参数量≈$\mathcal{O}(r(d_{in}+d_{out}))$，**训练/显存开销小**；推理仍走原路径（可合并或保留适配器）。
  - MoE：参数量可成倍增长（上亿/百亿级额外专家参数），**训练和系统工程复杂**；推理每token只用少数专家（稀疏）。
- **路由机制**
  - LoRA：**无路由**，一套适配器对所有token生效；若要多域切换，靠“加载不同适配器”。
  - MoE：**有路由**，按token/位置选择专家；需要负载均衡损失、容量管理（capacity factor）、抑制塌缩。
- **遗忘与模块化**
  - LoRA：卸载适配器即可回到基座；多适配器库便于任务切换。
  - MoE：参数共享在专家与路由器中；不是为“回退”设计的，更多是扩容量。
- **部署**
  - LoRA：单基座+多个小适配器文件即可；或离线合并为单权重。
  - MoE：需要**专家并行/跨设备通信**与一致的路由实现；推理调度更复杂。

**什么时候选哪个？**

- **只想快速上新任务/新领域**、算力有限、还想保留回退能力 ⇒ **LoRA**（或 QLoRA）。
- **想要更高的天花板**，能承担工程复杂度与更大内存/带宽 ⇒ **MoE**。
- **多域多风格**：LoRA 更像“多套可插拔滤镜”；MoE 更像“同一模型里学会很多专长并按需调用”。

**也可以“既要又要”的混搭**

- **Mixture of Adapters（MoA）/路由式适配器**：为不同域准备多套 LoRA，由一个轻量门控选择或加权融合 → “LoRA 有了点 MoE 味道”，但仍比真·MoE 简单。
- **MoE + LoRA**：在 MoE 基座上用 LoRA 做下游适配，减少全参再训练成本。
- **分阶段**：先用致密模型+LoRA拿到好起点，再迁到 MoE 架构扩大上限。

**常见陷阱**

- 把 LoRA 当成“动态路由专家”是不对的：**LoRA 不做 token 级选择**。
- 把 MoE 当成“免费增参”也不对：门控稳定性、专家均衡、通信开销都是难点。



### 如果我想要LoRA来复刻一个low rank的大模型可以吗？

不太可以，要分清两件事：

- **LoRA（微调适配）**：在已有基座 $W_0$ 上加一个低秩增量 $\Delta W=BA$；卸载适配器就回到 $W_0$。
- **低秩模型（压缩/复刻）**：直接把**每个线性层本体**用低秩因子化来表示：$W \approx B A$（不再带着 $W_0$）。

且预训练模型的权重矩阵通常含有较高的rank,奇异值较为平滑。

一定要做的**做法：后处理压缩（SVD → 低秩模型）**

1. 取每层全参权重 $W$，做 SVD：$W=U\Sigma V^\top$。
2. 取前 $r$ 个奇异值（能量覆盖率如 90–99%）：$\,U_r,\Sigma_r,V_r$。（r可能会很大）
3. 设 $B = U_r \Sigma_r^{1/2},\ A = \Sigma_r^{1/2} V_r^\top$，得到 $W \approx BA$，参数量 $r(d_{in}+d_{out})$。
4. 用一小段 **蒸馏/微调** 恢复精度（推荐知识蒸馏到 teacher 的 logits）。
    👉 这是最常见的**模型压缩**方式；和 “LoRA” 的数学形态一致，但**没有 $W_0$**，所以是纯低秩模型。

**优点**：一步到位拿到“低秩版权重”；**缺点**：可能有精度回退，需要少量再训练。

或者蒸馏。

