import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# 1) LoRA for Linear layers， used to replace MLP/linear in Attn
# ---------------------------
class LoRALinear(nn.Module):
    """
    LoRA- Linear:
      y = base(x) + scale * B(A(x)),  where A: in->r, B: r->out, scale = alpha / r
    - Frozen base weights（W0），only train A/B（and optional LoRA dropout）
    - provide merge/unmerge：merge the increment to base.weight，or restore to pluggable mode
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

        # base
        self.base = nn.Linear(in_features, out_features, bias=bias)
        for p in self.base.parameters():
            p.requires_grad_(False)  # Frozen base

        # LoRA bypass (two matrices)
        self.A = nn.Linear(in_features, r, bias=False)
        self.B = nn.Linear(r, out_features, bias=False)

        # Initialize: A random small value, B=0 makes the initial increment 0 (output=base output)
        nn.init.kaiming_uniform_(self.A.weight, a=5**0.5)
        if init_B_to_zero:
            nn.init.zeros_(self.B.weight)
        else:
            nn.init.kaiming_uniform_(self.B.weight, a=5**0.5)

        self.lora_dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    @property
    def deltaW(self):
        # Return the current low-rank increment matrix (out x in)
        # B.weight: [out, r], A.weight: [r, in]
        return self.B.weight @ self.A.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            # Already merged LoRA into base.weight，direct base
            return self.base(x)
        # Pluggable mode：base + low-rank bypass (two matrices)
        out = self.base(x)
        out = out + self.scale * self.lora_dropout(self.B(self.A(x)))
        return out

    @torch.no_grad()
    def merge(self):
        """
        Merge the increment of LoRA into base.weight:
          base.weight += scale * (deltaW)
        After merging, only base can be retained to simplify deployment.
        """
        if self.merged:
            return
        # Merge weights
        self.base.weight += self.scale * (self.deltaW)
        # If there is a bias LoRA variant, merge bias here
        self.merged = True

    @torch.no_grad()
    def unmerge(self):
        """
        Subtract the merged increment from base.weight, restore to `pluggable` mode.
        """
        if not self.merged:
            return
        self.base.weight -= self.scale * (self.deltaW)
        self.merged = False

    def trainable_parameters(self):
        """Return the trainable parameters of LoRA (A, B)"""
        return list(self.A.parameters()) + list(self.B.parameters())


# ----------------------------------------
# 2) Tool: replace the Linear with LoRALinear in the model
# ----------------------------------------
def inject_lora(
    model: nn.Module,
    target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "out_proj"),
    r=8,
    alpha=16,
    dropout=0.0,
):
    """
    Replace the nn.Linear with LoRALinear in the model that matches target_modules，
    and copy the original weights to base (keep forward equivalence).
    """
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
            lora_layer = LoRALinear(
                module.in_features,
                module.out_features,
                r=r,
                alpha=alpha,
                dropout=dropout,
                bias=(module.bias is not None),
            )
            # Move to the same device as the original module
            lora_layer = lora_layer.to(module.weight.device)
            # Copy the original weights to base
            lora_layer.base.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                lora_layer.base.bias.data.copy_(module.bias.data)
            setattr(model, name, lora_layer)
        else:
            # Recursively until encountering target_module
            inject_lora(module, target_modules, r, alpha, dropout)
    return model


# ----------------------------------------
# 3) A minimum trainable example
# ----------------------------------------
class TinyMLP(nn.Module):
    def __init__(self, in_dim=16, hidden=32, out_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_dim)   # We do LoRA on this layer
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

def demo_train_lora_vs_full():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Construct the "target" network (teacher), we hope the student network to fit its output
    teacher = TinyMLP().to(device).eval()

    # Student = copy a same structure, but only inject LoRA on fc2, and freeze other parameters
    student = TinyMLP().to(device)
    student = inject_lora(student, target_modules=("fc2",), r=4, alpha=16, dropout=0.05)

    # Freeze all parameters except LoRA
    for n, p in student.named_parameters():
        p.requires_grad_(False)
    # Open the trainable parameters of LoRA
    lora_params = []
    for m in student.modules():
        if isinstance(m, LoRALinear):
            for p in m.trainable_parameters():
                p.requires_grad_(True)
                lora_params.append(p)

    optimizer = torch.optim.AdamW(lora_params, lr=2e-4, weight_decay=0.0)

    # Prepare some random input/teacher output as supervision signal (can also be your data/labels)
    X = torch.randn(2048, 16, device=device)
    with torch.no_grad():
        Y = teacher(X)

    # Train for several steps: only update A/B of LoRA
    student.train()
    for step in range(800):
        idx = torch.randint(0, X.size(0), (64,), device=device)
        x, y = X[idx], Y[idx]
        pred = student(x)
        loss = F.mse_loss(pred, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()

        if (step + 1) % 200 == 0:
            with torch.no_grad():
                val_loss = F.mse_loss(student(X), Y).item()
            print(f"[step {step+1}] train_loss={loss.item():.6f}  val_loss={val_loss:.6f}")

    # Compare: whether the output of unmerged/merged weights is consistent
    with torch.no_grad():
        x_test = torch.randn(5, 16, device=device)
        y1 = student(x_test)              # Pluggable mode
        # Merge后再算
        for m in student.modules():
            if isinstance(m, LoRALinear):
                m.merge()
        y2 = student(x_test)              # Merged mode
        diff = (y1 - y2).abs().max().item()
        print(f"merge consistency max|diff| = {diff:.6e}")  # Should be close to 0

    # If you need to restore to pluggable mode
    for m in student.modules():
        if isinstance(m, LoRALinear):
            m.unmerge()

if __name__ == "__main__":
    demo_train_lora_vs_full()
