import json, re, math, hashlib, os, random
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets, interleave_datasets

MODEL = "unsloth/Qwen3-4B-Instruct-2507"  # Only for annotation/verification, no training dependency
CONTROL_PAT = re.compile(r"(</?think>|<\|im_start\|>|<\|im_end\|>|<\|assistant\|>|<\|user\|>)")
NUM_TOKEN = re.compile(r"N_(\d+)")

def clean_text(s: Optional[str]) -> str:
    if s is None: return ""
    s = s.replace("\r", "").strip()
    # Keep emoji, only remove control symbols
    s = CONTROL_PAT.sub("", s)
    return s

def wrap_messages(q: str, a: str):
    q, a = clean_text(q), clean_text(a)
    if not q or not a: return None
    return {"messages":[{"role":"user","content":q},{"role":"assistant","content":a}]}

# -------------------- Normalization: add source / tags to each sample --------------------
def add_metadata(ds: Dataset, source: str, tags: Optional[List[str]] = None) -> Dataset:
    tags = tags or []
    def _m(ex):
        ex["source"] = source  # Data source (dataset name/domain)
        ex["tags"] = tags      # Any custom tags (domain, question type, language, etc.)
        return ex
    return ds.map(_m)

# ---------- 1) CMATH ----------
def normalize_cmath(path: str) -> Dataset:
    ds = load_dataset("json", data_files=path, split="train")
    def _map(ex):
        q = ex.get("question", "")
        a = str(ex.get("golden", "")).strip()
        return wrap_messages(q, a)
    ds = ds.map(_map, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: x is not None)
    return add_metadata(ds, source="cmath", tags=["math", "zh"])

# ---------- 2) DPO-zh-en-emoji ----------
def normalize_emoji(path: str) -> Dataset:
    ds = load_dataset("json", data_files=path, split="train")
    def _map(ex):
        # Handle DPO format: use prompt as question and chosen response as answer
        q = ex.get("prompt", "")
        chosen = ex.get("chosen", [])
        
        # Extract assistant's response from chosen messages
        a = ""
        if isinstance(chosen, list):
            for msg in chosen:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    a = msg.get("content", "")
                    break
        
        # Fallback: try original format for backward compatibility
        if not a:
            a = ex.get("answer_zh") or ex.get("answer_en") or ""
            
        if len(a) > 3000: a = a[:3000]
        return wrap_messages(q, a)
    ds = ds.map(_map, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: x is not None)
    return add_metadata(ds, source="emoji", tags=["chat", "emoji", "zh_en_mix"])

# ---------- 3) MAWPS CSV ----------
def substitute_numbers(question: str, numbers: str) -> str:
    vals = [x for x in numbers.strip().split() if x]
    def repl(m):
        idx = int(m.group(1))
        return vals[idx] if idx < len(vals) else m.group(0)
    return NUM_TOKEN.sub(repl, question)

def normalize_mawps(path: str) -> Dataset:
    ds = load_dataset("csv", data_files=path, split="train")
    def _map(ex):
        q_raw = ex.get("Question", "")
        nums  = ex.get("Numbers", "") or ""
        q = substitute_numbers(q_raw, nums)
        a = str(ex.get("Answer", "")).strip()
        # Remove 12.0 -> 12
        try:
            f = float(a)
            if math.isfinite(f) and abs(f - round(f)) < 1e-9: a = str(int(round(f)))
        except: pass
        return wrap_messages(q, a)
    ds = ds.map(_map, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: x is not None)
    return add_metadata(ds, source="mawps", tags=["math", "en"])

# -------------------- Deduplication --------------------
def dedup(ds: Dataset) -> Dataset:
    def _hash(ex):
        # Handle cases where 'messages' field might not exist
        if "messages" not in ex:
            # If no messages field, create a hash from all available data
            s = json.dumps(ex, ensure_ascii=False, sort_keys=True)
        else:
            s = json.dumps(ex["messages"], ensure_ascii=False)
        return {"_h": hashlib.md5(s.encode("utf-8")).hexdigest()}
    
    ds = ds.map(_hash, num_proc=1)
    
    # Use filter to remove duplicates
    seen_hashes = set()
    def _filter_duplicates(ex):
        h = ex["_h"]
        if h in seen_hashes:
            return False
        seen_hashes.add(h)
        return True
    
    ds = ds.filter(_filter_duplicates)
    return ds.remove_columns(["_h"])

# -------------------- Mix strategy --------------------
def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(0.0, v) for v in weights.values())
    if s <= 0:  # fallback: equal distribution
        n = len(weights) if weights else 1
        return {k: 1.0 / n for k in weights}
    return {k: max(0.0, v) / s for k, v in weights.items()}

def interleave_mix(dsets: Dict[str, Dataset], weights: Dict[str, float], seed: int = 42):
    """Lightweight mix: sample with probability interleaving until all datasets are exhausted (will not hit the ratio exactly, but close enough)"""
    keys, probs = zip(*_normalize_weights(weights).items())
    ds_list = [dsets[k].shuffle(seed=seed) for k in keys]
    return interleave_datasets(
        ds_list,
        probabilities=list(probs),
        seed=seed,
        stopping_strategy="all_exhausted"  # Use the simplest and safest strategy
    )

def _resample_to_length(ds: Dataset, target_n: int, seed: int = 42) -> Dataset:
    if target_n <= 0:
        return ds.select([])
    n = len(ds)
    if n == 0:
        return ds.select([])
    rnd = random.Random(seed)
    idx = [rnd.randrange(n) for _ in range(target_n)]  # Sample with replacement
    return ds.select(idx)

def resample_mix(
    dsets: Dict[str, Dataset],
    weights: Dict[str, float],
    total_size: Optional[int] = None,
    seed: int = 42,
):
    """Strict ratio: given total size, sample with replacement according to weights, hit the ratio exactly"""
    w = _normalize_weights(weights)
    if total_size is None:
        # Default is 0.9 times the sum of the sizes of the sub-datasets, to avoid extreme upsampling; you can also explicitly pass total_size
        total_size = max(1, int(sum(len(v) for v in dsets.values()) * 0.9))
    plan = {k: int(round(total_size * p)) for k, p in w.items()}
    resampled = [
        _resample_to_length(dsets[k].shuffle(seed=seed + i), plan[k], seed + 17 + i)
        for i, k in enumerate(dsets.keys())
    ]
    return concatenate_datasets(resampled)

# -------------------- Split --------------------
def split_train_valid(ds: Dataset, valid_ratio=0.02, seed=42) -> DatasetDict:
    ds = ds.shuffle(seed=seed)
    n_valid = max(100, int(len(ds) * valid_ratio))
    valid = ds.select(range(n_valid))
    train = ds.select(range(n_valid, len(ds)))
    return DatasetDict(train=train, validation=valid)

# -------------------- Total build entry --------------------
def build(
    cmath_p, emoji_p, mawps_p,
    weights: Dict[str, float] = None,
    mix_strategy: str = "interleave",      # "interleave" | "resample"
    total_size: Optional[int] = None,      # Only valid when mix_strategy == "resample"
    valid_ratio: float = 0.02,
    seed: int = 42,
    separate_save: bool = False,           # Whether to separate save
    separate_dir: str = "processed_datasets/normalized",
) -> Tuple[DatasetDict, Dict[str, Dataset]]:
    # 1) Normalization
    a = normalize_cmath(cmath_p)
    b = normalize_emoji(emoji_p)
    c = normalize_mawps(mawps_p)

    # 2) Separate save (industrial common, for easy re-mixing)
    if separate_save:
        os.makedirs(separate_dir, exist_ok=True)
        a.save_to_disk(os.path.join(separate_dir, "cmath"))
        b.save_to_disk(os.path.join(separate_dir, "emoji"))
        c.save_to_disk(os.path.join(separate_dir, "mawps"))

    # 3) Deduplication (deduplicate each dataset + deduplicate after merging)
    a, b, c = dedup(a), dedup(b), dedup(c)
    pools = {"cmath": a, "emoji": b, "mawps": c}

    # 4) Mix
    weights = weights or {"cmath": 1.0, "emoji": 1.0, "mawps": 1.0}
    if mix_strategy == "resample":
        merged = resample_mix(pools, weights, total_size=total_size, seed=seed)
    else:
        merged = interleave_mix(pools, weights, seed=seed)

    merged = dedup(merged)  # Do it again (Deduplication)
    dsd = split_train_valid(merged, valid_ratio=valid_ratio, seed=seed)
    return dsd, pools

if __name__ == "__main__":
    # -------- Original data paths --------
    cmath_p = "base_datasets/cmath/cmath_dev.jsonl"
    emoji_p = "base_datasets/DPO-zh-en-emoji/merged_dpo_zh_emoji_for_firefly.jsonl"
    mawps_p = "base_datasets/MAWPS/MAWPS.csv"

    # -------- Output directory --------
    out_dir = "processed_datasets/qwen3_sft_mixed"  # The mixed package used for this training
    norm_dir = "processed_datasets/normalized"  # Separate save directory

    # -------- Mix weights (by domain/source)--------
    mix_weights = {
        "cmath": 0.2,   # Chinese math
        "emoji": 0.5,   # Chat/emoji mixed
        "mawps": 0.3,   # English math
    }

    # Select mix strategy:
    # - "interleave": simple and fast, approximate ratio
    # - "resample":   strict hit ratio; can set total_size to control training size
    strategy = "interleave"     # or "resample"
    total_n = None              # e.g., 30000 when strategy == "resample"

    dsd, pools = build(
        cmath_p, emoji_p, mawps_p,
        weights=mix_weights,
        mix_strategy=strategy,
        total_size=total_n,
        valid_ratio=0.02,
        seed=42,
        separate_save=True,          # Separate save (recommended)
        separate_dir=norm_dir,
    )

    # It is recommended to save in Arrow format, which is convenient for next load_from_disk
    os.makedirs(out_dir, exist_ok=True)
    dsd["train"].save_to_disk(os.path.join(out_dir, "train"))
    dsd["validation"].save_to_disk(os.path.join(out_dir, "validation"))

    # If you need to interface with external tools, you can also export JSONL (keep Chinese/emoji)
    dsd["train"].to_json(os.path.join(out_dir, "train.jsonl"), force_ascii=False)
    dsd["validation"].to_json(os.path.join(out_dir, "val.jsonl"), force_ascii=False)

    # Brief information
    print("Per-pool sizes:", {k: len(v) for k, v in pools.items()})
    print(dsd)
