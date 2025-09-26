# pip install --upgrade huggingface_hub
# hf auth login

import os, sys, time, json, math, re, gc, importlib, pkgutil, subprocess
from pathlib import Path
from typing import Any, Dict, Optional
from transformers import AutoModelForCausalLM

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Default configuration for pipeline execution. These values mirror the keys
# injected into ``cfg`` inside ``main`` so that the script can run without
# requiring external argument parsing or environment mutation.
DEFAULT_CFG = {
    "model_id": "google/gemma-3-1b-pt",
    "dataset_id": "yaful/MAGE",
    "save_dir": "./runs",
    "seed": 42,
    "max_length": 256,
    "epochs": 1,
    "batch_size": 16,
    "grad_accum": 1,
    "gradient_checkpointing": False,
    "attn_impl": "sdpa",
    "torch_compile": False,
    "log_interval": 50,
    "max_grad_norm": 1.0,
    "use_lora": True, # QLoRA on by default
    # include MLP projections too (Gemma/Llama style names)
    "target_mods": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    # QLoRA-friendly training defaults
    "lr": 1e-4,
    "weight_decay": 0.0,
    "warmup_ratio": 0.06,
    "max_train": 2000,
    "max_val": 200,
    "wandb": False,
    "eval_every": 0,
    "eval_only": False,
    "warm_start": True,
}

RUNS = [
    {
        "model_id": "google/gemma-3-1b-pt",
        "save_dir": "./runs/gemma-3-1b-pt",
        "epochs": 1,
        "batch_size": 16,
        "gradient_checkpointing": False
    },
    {
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "save_dir": "./runs/llama-3-8b-instruct",
        "epochs": 1,                     # shorter run
        "batch_size": 8,                 # safer for 8B QLoRA
        "grad_accum": 2,                 # keep effective batch size reasonable
        "gradient_checkpointing": True   # reduce memory
    }
]

import contextlib

@contextlib.contextmanager
def _disable_peft_autoload():
    """
    Make transformers think PEFT is unavailable *just while* we load the base model.
    Avoids the 'maybe_adapter_path' type-confusion bug.
    """
    import transformers
    # 1) patch the central utils gate
    from transformers import utils as hf_utils
    _saved_utils = getattr(hf_utils, "is_peft_available", None)
    hf_utils.is_peft_available = lambda: False

    # 2) also patch modules that cached the symbol at import time
    #    (auto_factory / modeling_utils may have their own copy)
    _saved_auto = _saved_modeling = None
    try:
        from transformers.models.auto import auto_factory as af
        _saved_auto = getattr(af, "is_peft_available", None)
        af.is_peft_available = lambda: False
    except Exception:
        pass
    try:
        from transformers import modeling_utils as mu
        _saved_modeling = getattr(mu, "is_peft_available", None)
        mu.is_peft_available = lambda: False
    except Exception:
        pass

    try:
        yield
    finally:
        # restore originals
        if _saved_utils is not None:
            hf_utils.is_peft_available = _saved_utils
        if _saved_auto is not None:
            af.is_peft_available = _saved_auto
        if _saved_modeling is not None:
            mu.is_peft_available = _saved_modeling


def log(msg: str = "", *, prefix: str = "", end: str = "\n"):
    if prefix:
        print(f"{prefix} {msg}", flush=True, end=end)
    else:
        print(msg, flush=True, end=end)

def hline(txt=""):
    log("=" * 80)
    if txt:
        log(txt)
        log("=" * 80)

def ensure_pip():
    try:
        import pip  # noqa
        return True
    except Exception:
        log("[BOOT] ensurepip.bootstrap()", prefix="→")
        import ensurepip
        ensurepip.bootstrap()
        return True

def pip_install(pkgs, index_url=None):
    cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--user", "-U"]
    if index_url:
        cmd += ["--index-url", index_url]
    cmd += pkgs
    env = os.environ.copy()
    # force visible progress
    env["PIP_PROGRESS_BAR"] = "on"
    env.pop("PYTHONNOUSERSITE", None)
    log(" ".join(cmd), prefix="[PIP]")
    # stream output so the user sees progress in real-time
    proc = subprocess.Popen(cmd, env=env)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"pip install failed for: {pkgs}")
    return True

def installed(mod_name: str) -> bool:
    base = mod_name.split(">=")[0].split("[")[0]
    return pkgutil.find_loader(base) is not None
def ensure_core_packages():
    import os, sys, subprocess, importlib, site
    from pathlib import Path

    def p(msg): print(f"[PIP] {msg}", flush=True)

    # --- 0) Force setuptools to own distutils (prevents the _distutils_hack warning/assert) ---
    os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "stdlib")
    # If distutils was already imported by the system Anaconda, purge it so setuptools can re-wire it.
    for name in list(sys.modules):
        if name == "distutils" or name.startswith("distutils."):
            del sys.modules[name]
    try:
        import setuptools  # noqa: F401
        p("Imported setuptools *before* distutils (good).")
    except Exception as e:
        p(f"Could not import setuptools early: {e}")

    # --- 1) Ensure pip exists and upgrade pip/setuptools/wheel in user site first ---
    try:
        import pip  # noqa: F401
    except Exception:
        import ensurepip; ensurepip.bootstrap()
    def pip_install(args, index_url=None):
        cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--user", "-U"]
        if index_url: cmd += ["--index-url", index_url]
        cmd += args
        env = os.environ.copy()
        env["PIP_PROGRESS_BAR"] = "on"
        env.pop("PYTHONNOUSERSITE", None)  # we DO want user site
        p(" ".join(cmd))
        rc = subprocess.call(cmd, env=env)
        if rc != 0:
            raise RuntimeError(f"pip failed for: {args}")

    pip_install(["pip", "setuptools", "wheel", "packaging"])

    # --- 2) Make user site take precedence on sys.path (so our upgrades win over system Anaconda) ---
    try:
        user_site = site.getusersitepackages()
    except Exception:
        user_site = str(Path.home()/".local"/"lib"/f"python{sys.version_info.major}.{sys.version_info.minor}"/"site-packages")
    if user_site in sys.path: sys.path.remove(user_site)
    sys.path.insert(0, user_site)
    p(f"user site prepended: {user_site}")
    p(f"python: {sys.executable}")

    # --- 3) Version-aware installs of the ML stack ---
    try:
        from importlib.metadata import version, PackageNotFoundError
    except Exception:
        from importlib_metadata import version, PackageNotFoundError  # backport
    from packaging.version import Version
    def needs(pkg, minver):
        try: return Version(version(pkg)) < Version(minver)
        except PackageNotFoundError: return True

    # Torch (CUDA 12.4 wheels work on A40)
    if needs("torch", "2.4.0"):
        p("Installing torch/vision/audio (CUDA 12.4)…")
        pip_install(["torch", "torchvision", "torchaudio"], index_url="https://download.pytorch.org/whl/cu124")
    else:
        p("torch OK")

    reqs = {
        "transformers":   "4.56.2",
        "datasets":       "2.20.0",
        "accelerate":     "0.33.0",
        "peft":           "0.17.1",
        "bitsandbytes":   "0.43.1",
        "evaluate":       "0.4.2",
        "scikit-learn":   "1.4.0",
        "huggingface_hub":"0.24.0",
        "sentencepiece":  "0.1.99",
        "tiktoken":       "0.7.0",
        "tqdm":           "4.66.0",
        "wandb":          "0.17.0",
        "pandas":         "2.2.2",
        "pyarrow":        "14.0.2",
    }
    to_install = [f"{k}>={v}" for k, v in reqs.items() if needs(k, v)]
    if to_install:
        p(f"Installing/Upgrading: {to_install}")
        pip_install(to_install)
    else:
        p("All core packages satisfy minimum versions.")

    # --- 4) Purge any already-imported old modules; re-import from user site ---
    for mod in ("transformers", "datasets", "accelerate", "peft", "bitsandbytes"):
        if mod in sys.modules: del sys.modules[mod]

    # --- 5) Snapshot versions and verify BitsAndBytesConfig is available ---
    import transformers, datasets, peft, accelerate, bitsandbytes, torch
    p(f"python        {sys.version.split()[0]}")
    p(f"torch         {torch.__version__}")
    p(f"transformers  {transformers.__version__} @ {Path(transformers.__file__).resolve()}")
    p(f"datasets      {datasets.__version__} @ {Path(datasets.__file__).resolve()}")
    p(f"peft          {peft.__version__} @ {Path(peft.__file__).resolve()}")
    p(f"accelerate    {accelerate.__version__} @ {Path(accelerate.__file__).resolve()}")
    p(f"bitsandbytes  {bitsandbytes.__version__} @ {Path(bitsandbytes.__file__).resolve()}")

    # This covers both new and older Transformers layouts
    try:
        from transformers import BitsAndBytesConfig  # noqa: F401
    except Exception:
        from transformers.utils.quantization_config import BitsAndBytesConfig  # noqa: F401
    p("BitsAndBytesConfig import OK")

def setup_env_and_caches():
    from shutil import disk_usage

    HOME = os.path.expanduser("~")
    os.environ["PATH"] = f"{HOME}/.local/bin:" + os.environ.get("PATH", "")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    tmp = os.environ.get("TMPDIR") or f"/tmp/{os.environ.get('USER', 'user')}"
    hf_home = f"{tmp}/hf_home"
    hf_hub = f"{tmp}/hf_hub"
    hf_ds  = f"{tmp}/hf_datasets"

    for d in (hf_home, hf_hub, hf_ds):
        Path(d).mkdir(parents=True, exist_ok=True)
    os.environ.update({"HF_HOME": hf_home, "HF_HUB_CACHE": hf_hub, "HF_DATASETS_CACHE": hf_ds})

    hline("[ENV] Caches & disk")
    for p in (Path(hf_home), Path(hf_hub), Path(hf_ds)):
        try:
            total, used, free = disk_usage(p)
            log(f"{p}  (free: {free/1e9:.2f} GB, used: {used/1e9:.2f} GB, total: {total/1e9:.2f} GB)")
        except Exception:
            log(f"{p}  (disk usage: n/a)")

    # Enable tqdm-based download progress for HF
    try:
        from huggingface_hub.utils import logging as hf_logging
        hf_logging.set_verbosity_info()
        log("HuggingFace Hub logging set to INFO (will show download progress)", prefix="[HF]")
    except Exception:
        pass

def disable_hf_transfer():
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    try:
        from huggingface_hub import constants, file_download
        constants.HF_HUB_ENABLE_HF_TRANSFER = False
        importlib.reload(file_download)
        log("Fast transfer disabled → standard tqdm progress.", prefix="[HF]")
    except Exception:
        log("Fast transfer disabled via env.", prefix="[HF]")

def gpu_probe(preferred=None):
    import torch, subprocess
    assert torch.cuda.is_available(), "CUDA GPU not visible (check --gres=gpu and partition)."
    name = torch.cuda.get_device_name(0)
    maj, minr = torch.cuda.get_device_capability(0)
    cc = f"{maj}.{minr}"
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    # Use bf16 only on Ampere+ (SM >= 80). Otherwise use fp16. Do NOT trust is_bf16_supported() on pre-Ampere.
    if preferred in ("bf16", "fp16"):
        dtype = preferred
    else:
        dtype = "bf16" if maj >= 8 else "fp16"

    hline("[GPU] CUDA device")
    log(f"Name: {name}")
    log(f"Compute Capability: {cc}")
    log(f"VRAM: {vram:.1f} GB")
    log(f"Preferred precision: {dtype}")
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"]).decode()
        log(out.strip())
    except Exception:
        pass
    return dtype

def set_seed(seed=5541):
    import torch, numpy as np, random as pyrand
    pyrand.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def json_dump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f: json.dump(obj, f, indent=2)

# ----- Persistent run-wide metrics logging (append-only CSV + JSONL) -----
_METRIC_KEYS = [
    "run_id","timestamp","event","phase","epoch","step","batch",
    "loss","loss_avg_window","total_loss","n_steps","acc","f1_macro","auroc","lr",
    "model_id","dataset","batch_size","grad_accum",
    "warm_start_used","warm_notes","eval_tag"
]

def _ensure_perm_logs(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "metrics.csv"
    jsonl_path = root / "metrics.jsonl"
    return csv_path, jsonl_path

def _append_metric(root: Path, row: dict):
    import csv, os
    csv_path, jsonl_path = _ensure_perm_logs(root)
    # Normalize row to canonical schema
    normalized = {k: row.get(k, None) for k in _METRIC_KEYS}
    # JSONL append
    with open(jsonl_path, "a", encoding="utf-8") as jf:
        jf.write(json.dumps(row, ensure_ascii=False) + "\n")
    # CSV append (write header if file doesn't exist)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as cf:
        w = csv.DictWriter(cf, fieldnames=_METRIC_KEYS, extrasaction="ignore")
        if write_header: w.writeheader()
        w.writerow(normalized)


def safe_load_tokenizer(model_id: str):
    import os, json, re, shutil
    from pathlib import Path
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    def _log(msg): log(f"[TOK] {msg}")

    # 1) Local patch dir (separate from HF cache)
    tmp_root = Path(os.environ.get("TMPDIR", "/tmp")) / (os.environ.get("USER", "user") or "user")
    local_tok_dir = tmp_root / f"tokpatch__{re.sub(r'[^a-zA-Z0-9_.-]+', '_', model_id)}"
    local_tok_dir.mkdir(parents=True, exist_ok=True)

    # 2) Pull only tokenizer artifacts into cache; re-runs hit the cache (fast)
    allow = [
        "tokenizer.json",
        "tokenizer.model",
        "vocab.json",            # GPT-2 / BPE
        "merges.txt",            # GPT-2 / BPE
        "special_tokens_map.json",
        "added_tokens.json",
        "tokenizer_config.json",
    ]

    _log(f"snapshot_download allow_patterns={allow}")
    snap_dir = snapshot_download(
        repo_id=model_id,
        allow_patterns=allow,
        cache_dir=os.environ.get("HF_HUB_CACHE")  # if set by your env bootstrap
    )

    # 3) Copy files to our writable patch location
    present = []
    for fname in allow:
        src = Path(snap_dir) / fname
        if src.exists():
            shutil.copy2(src, local_tok_dir / fname)
            present.append(fname)

    # Require at least one real tokenizer *set*; otherwise fall back to HF auto-load
    has_sp = ("tokenizer.json" in present) or ("tokenizer.model" in present)
    has_gpt2_bpe = ("vocab.json" in present) and ("merges.txt" in present)
    if not (has_sp or has_gpt2_bpe):
        _log("no tokenizer files in snapshot; falling back to AutoTokenizer.from_pretrained(model_id)")
        tok = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            trust_remote_code=False,
        )
        if tok.pad_token is None and getattr(tok, "eos_token", None):
            tok.pad_token = tok.eos_token
            _log(f"set pad_token = eos_token ({tok.eos_token})")
        _log("tokenizer ready (fallback path)")
        return tok

    # 4) Sanitize tokenizer_config.json deterministically
    cfg_path = local_tok_dir / "tokenizer_config.json"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            tok_cfg = json.load(f)

        # Normalize/strip broken chat template fields
        if "chat_template" in tok_cfg:
            ct = tok_cfg["chat_template"]
            if isinstance(ct, dict) and isinstance(ct.get("template"), str):
                tok_cfg["chat_template"] = ct["template"]
                _log("patched chat_template: dict → string")
            else:
                tok_cfg.pop("chat_template", None)
                _log("removed invalid chat_template")

        if "chat_template_file" in tok_cfg and not isinstance(tok_cfg["chat_template_file"], str):
            tok_cfg.pop("chat_template_file", None)
            _log("removed invalid chat_template_file")

        # Persist sanitized config
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(tok_cfg, f, ensure_ascii=False)

    else:
        _log("no tokenizer_config.json present (nothing to patch)")

    # 5) Load tokenizer
    tok = AutoTokenizer.from_pretrained(
        str(local_tok_dir),
        use_fast=True,
        trust_remote_code=False,
    )

    # 6) Ensure padding token exists
    if tok.pad_token is None and getattr(tok, "eos_token", None):
        tok.pad_token = tok.eos_token
        _log(f"set pad_token = eos_token ({tok.eos_token})")

    _log("tokenizer ready")
    return tok

from transformers import AutoConfig, AutoModelForCausalLM

def safe_load_backbone(
    model_id: str,
    base_dtype,
    quant,
    *,
    device_map=None,
    trust_remote_code: bool = False,
    attn_impl: Optional[str] = None,
    gradient_checkpointing: bool = False,
):
    if device_map is None:
        device_map = {"": 0}

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if hasattr(cfg, "auto_map"):
        cfg.auto_map = None

    if attn_impl:
        chosen_attn = attn_impl
    else:
        chosen_attn = "eager" if "gemma-3" in model_id.lower() else "sdpa"
    log(f"attn_implementation={chosen_attn}", prefix="[MODEL]")

    # >>> disable PEFT autoload only for this call <<<
    with _disable_peft_autoload():
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=cfg,
            quantization_config=quant,
            torch_dtype=(None if quant else base_dtype),
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            attn_implementation=chosen_attn,
        )

    # training-time prefs
    model.config.output_hidden_states = False
    model.config.use_cache = False
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return model

def load_and_tokenize(cfg: Dict[str, Any], save_root: Path, max_train=0, max_val=0):
    disable_hf_transfer()
    from datasets import load_dataset, ClassLabel, DatasetDict, load_from_disk
    from transformers import AutoTokenizer, DataCollatorWithPadding
    from collections import Counter
    from tqdm.auto import tqdm

    cache_dir = save_root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = re.sub(r"[^a-zA-Z0-9_.-]+", "_", f"{cfg['dataset_id']}__{cfg['model_id']}__L{cfg['max_length']}")
    tok_cache = cache_dir / f"tok_{key}"

    if tok_cache.exists():
        log(f"Loading tokenized dataset from disk → {tok_cache}", prefix="[DATA]")
        ds_tok = load_from_disk(str(tok_cache))
        with open(tok_cache / "meta.json") as f:
            meta = json.load(f)
        id2label = {int(k): v for k, v in meta["id2label"].items()}
        tok = safe_load_tokenizer(cfg["model_id"])
        collator = DataCollatorWithPadding(tokenizer=tok, pad_to_multiple_of=8)
        return ds_tok, collator, id2label

    hline("[DATA] Downloading/reading dataset")
    t0 = time.time()
    ds = load_dataset(cfg["dataset_id"])
    log(f"Splits: {list(ds.keys())}")
    log(f"Train size: {len(ds['train'])}" + (f", Test size: {len(ds['test'])}" if "test" in ds else ""))

    # pick columns (must be before we create/stratify validation)
    cand_text = ["text","content","document","body","sentence","prompt","input","inputs","article"]
    cand_label = ["label","labels","target","class","gold","source"]
    cols = list(ds["train"].column_names)
    text_col = next((c for c in cand_text if c in cols), cols[0])
    label_col = next((c for c in cand_label if c in cols), None)
    assert label_col is not None, f"No label-like column found in {cols}"
    log(f"text_col={text_col} | label_col={label_col}", prefix="[DATA]")

    # ensure validation split (stratified)
    keys = set(ds.keys())
    if "validation" not in keys and "val" not in keys:
        log("No validation split → creating 10% from train (stratified)", prefix="[DATA]")
        split = ds["train"].train_test_split(test_size=0.1, seed=cfg["seed"], stratify_by_column=label_col)
        from datasets import DatasetDict as HF_DatasetDict
        ds = HF_DatasetDict(train=split["train"], validation=split["test"], **({"test": ds["test"]} if "test" in keys else {}))
    elif "val" in keys:
        ds = DatasetDict(train=ds["train"], validation=ds["val"], **({"test": ds["test"]} if "test" in keys else {}))

    # label mapping
    feat = ds["train"].features.get(label_col)
    # Special-case: MAGE uses 0=AI (machine), 1=HUMAN
    if cfg["dataset_id"].lower() in {"yaful/mage", "mage", r"yaful\mage"}:
        id2label = {0: "AI", 1: "HUMAN"}
    elif isinstance(feat, ClassLabel):
        id2label = {i: n.upper() for i, n in enumerate(feat.names)}
    else:
        vals = set(ds["train"][label_col])
        if vals <= {0,1} or vals <= {"0","1"}:
            id2label = {0:"HUMAN", 1:"AI"}
        else:
            low = {str(x).strip().lower() for x in vals}
            assert low <= {"human","real","ai","gpt","machine"}, f"Unrecognized labels: {vals}"
            id2label = {0:"HUMAN", 1:"AI"}
    log(f"id2label={id2label}", prefix="[DATA]")

    def map_label_value(v):
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"0","1"}: return int(s)
            return {"human":0,"real":0,"ai":1,"gpt":1,"machine":1}[s]
        return int(v)

    # subsample (stratified when we can)
    auto_max_train = int(os.environ.get("AUTO_MAX_TRAIN", "20000"))
    auto_max_val   = int(os.environ.get("AUTO_MAX_VAL",   "2000"))
    disable_auto   = os.environ.get("AUTO_SUBSAMPLE", "1") in {"0", "false", "False"}

    def maybe_take_stratified(ds_split, nmax, fallback):
        limit = nmax if (nmax and nmax > 0) else (0 if disable_auto else fallback)
        if limit and len(ds_split) > limit:
            log(f"Auto-subsample (stratified) → {limit} rows (set AUTO_SUBSAMPLE=0 to disable)", prefix="[DATA]")
            try:
                # Use train_test_split with an integer size and stratification to pick a balanced subset
                sub = ds_split.train_test_split(test_size=limit, seed=cfg["seed"], stratify_by_column=label_col)["test"]
                return sub
            except Exception as e:
                log(f"Stratified subsample failed ({e}) → falling back to shuffled head", prefix="[DATA]")
                return ds_split.shuffle(seed=cfg["seed"]).select(range(limit))
        return ds_split

    ds["train"] = maybe_take_stratified(ds["train"], max_train, auto_max_train)
    ds["validation"] = maybe_take_stratified(ds["validation"], max_val, auto_max_val)
    log(f"Using: train={len(ds['train'])}, val={len(ds['validation'])}", prefix="[DATA]")

    tok = safe_load_tokenizer(cfg["model_id"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    max_len = int(cfg["max_length"])

    def tok_fn(batch):
        enc = tok(batch[text_col], truncation=True, max_length=max_len)
        enc["labels"] = [map_label_value(v) for v in batch[label_col]]
        enc["raw_text"] = batch[text_col]
        return enc

    log("Tokenizing …", prefix="[DATA]")
    from datasets import DatasetDict as HF_DatasetDict
    ds_tok = {}
    for split in ds.keys():
        log(f"→ {split} ({len(ds[split])} rows)", prefix="[DATA]")
        ds_tok[split] = ds[split].map(
            tok_fn, batched=True, remove_columns=ds[split].column_names, desc=f"Tokenize[{split}]"
        )
    ds_tok = HF_DatasetDict(**ds_tok)

    from transformers import DataCollatorWithPadding
    collator = DataCollatorWithPadding(tokenizer=tok, pad_to_multiple_of=8)
    y_counts = Counter(ds_tok["train"]["labels"])
    log("Label dist (train): " + str({id2label[k]: v for k, v in sorted(y_counts.items())}), prefix="[DATA]")

    log(f"Saving tokenized dataset → {tok_cache}", prefix="[CACHE]")
    ds_tok.save_to_disk(str(tok_cache))
    json_dump({"id2label": id2label, "text_col": text_col, "label_col": label_col, "max_length": max_len}, tok_cache / "meta.json")
    log(f"Dataset ready in {time.time()-t0:.1f}s", prefix="[DATA]")
    return ds_tok, collator, id2label

def train_eval(cfg: Dict[str, Any]):
    import torch, time
    import torch.nn as nn, torch.nn.functional as F
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    from transformers import get_linear_schedule_with_warmup
    try:
        from transformers import BitsAndBytesConfig
    except Exception:
        from transformers.utils.quantization_config import BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, PeftModel
    import bitsandbytes as bnb
    run_id = cfg.get("run_id") or time.strftime("%Y%m%d-%H%M%S")
    cfg["run_id"] = run_id

    device = torch.device("cuda")
    save_root = Path(cfg["save_dir"]); save_root.mkdir(parents=True, exist_ok=True)
    best_dir = save_root / "best"

    # Data
    ds_tok, collator, id2label = load_and_tokenize(cfg, save_root, cfg["max_train"], cfg["max_val"])
    # drop the non-tensor column from the training split (prevents 'raw_text' from reaching the collator)
    train_ds = ds_tok["train"].remove_columns([c for c in ds_tok["train"].column_names if c == "raw_text"])

    cpu_count = os.cpu_count() or 8
    if cpu_count <= 1:
        loader_workers = 1
    else:
        loader_workers = min(16, max(2, cpu_count))
    dl_common = {
        "pin_memory": True,
        "pin_memory_device": "cuda",
        "num_workers": loader_workers,
        "persistent_workers": loader_workers > 0,
    }
    if loader_workers > 0:
        dl_common["prefetch_factor"] = 4

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=collator,  # DataCollatorWithPadding
        **dl_common,
    )
    
    val_bs = max(1, cfg["batch_size"] * 2)
    # use a copy of validation without 'raw_text' for mid-epoch evals during training
    val_ds_for_train = ds_tok["validation"].remove_columns([c for c in ds_tok["validation"].column_names if c == "raw_text"])
    val_dl = DataLoader(
        val_ds_for_train,
        batch_size=val_bs,
        shuffle=False,
        collate_fn=collator,  # DataCollatorWithPadding
        **dl_common,
    )


    # Early paths
    if cfg["eval_only"]:
        if best_dir.exists():
            log("eval-only requested → evaluating best checkpoint", prefix="[EVAL]")
            return evaluate_and_save(cfg, best_dir, ds_tok, val_dl, collator, id2label)
        else:
            log("eval-only requested but no checkpoint found → exit", prefix="[EVAL]")
            return

    # Precision & GPU info
    if cfg["precision"] is None:
        cfg["precision"] = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    log(f"Precision: {cfg['precision']}", prefix="[TRAIN]")

    # Quantization choice (no regex, no try/except)
    from packaging.version import Version

    base_dtype = torch.bfloat16 if cfg["precision"] == "bf16" else torch.float16
    
    # Normalize version like "0.43.1+cu124" or "0.43.1-foo" → "0.43.1"
    bnb_version_raw = getattr(bnb, "__version__", "0.0.0")
    bnb_version_clean = bnb_version_raw.split("+", 1)[0].split("-", 1)[0]
    
    bnb_ok = Version(bnb_version_clean) >= Version("0.42.0")
    
    if bnb_ok:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=base_dtype,
            bnb_4bit_use_double_quant=True,
        )
        log(f"Quantization: 4-bit NF4 (bitsandbytes {bnb_version_raw})", prefix="[MODEL]")
    else:
        quant = None
        log(f"Quantization: disabled (bitsandbytes {bnb_version_raw} < 0.42.0)", prefix="[MODEL]")


    # Model load + WARM-START (weights only; fresh optimizer/schedule every run)
    t0 = time.time()
    log(f"Loading base model: {cfg['model_id']}", prefix="[MODEL]")
    grad_ckpt = bool(cfg.get("gradient_checkpointing", False))
    base = safe_load_backbone(
        cfg["model_id"],
        base_dtype,
        quant,
        attn_impl=cfg.get("attn_impl"),
        gradient_checkpointing=grad_ckpt,
    )
    from peft import prepare_model_for_kbit_training
    prepare_kwargs = {"use_gradient_checkpointing": grad_ckpt}
    if grad_ckpt:
        prepare_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    base = prepare_model_for_kbit_training(base, **prepare_kwargs)
    if grad_ckpt and hasattr(base, "enable_input_require_grads"):
        base.enable_input_require_grads()

    warm_used = False
    warm_notes = []
    backbone = None

    # Try warm-start from best_dir whenever possible (independent of epochs)
    try:
        if cfg.get("warm_start", True) and best_dir.exists():
            if cfg["use_lora"] and (best_dir / "lora_adapter").exists():
                log("WARM-START: loading previous LoRA adapter", prefix="[WARM]")
                backbone = PeftModel.from_pretrained(base, str(best_dir / "lora_adapter"), is_trainable=True)
                warm_used = True
            elif (not cfg["use_lora"]) and (best_dir / "backbone").exists():
                log("WARM-START: loading previous finetuned backbone", prefix="[WARM]")
                backbone = AutoModelForCausalLM.from_pretrained(
                    str(best_dir / "backbone"),
                    torch_dtype=base_dtype,
                    device_map={"": 0}
                )
                backbone.config.output_hidden_states = False
                backbone.config.use_cache = False
                backbone = prepare_model_for_kbit_training(backbone, **prepare_kwargs)
                warm_used = True
            else:
                warm_notes.append("no_compatible_artifact_in_best_dir")
        else:
            warm_notes.append("best_dir_missing_or_warm_start_disabled")
    except Exception as e:
        warm_notes.append(f"warm_start_error:{type(e).__name__}")

    if backbone is None:
        # Scratch path: inject new LoRA if configured
        backbone = base
        if cfg["use_lora"]:
            log("Injecting NEW LoRA adapters (scratch)", prefix="[LoRA]")
            from peft import LoraConfig, get_peft_model
            lcfg = LoraConfig(
                r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                target_modules=cfg["target_mods"].split(","), bias="none", task_type="CAUSAL_LM",
            )
            backbone = get_peft_model(base, lcfg)
        log("WARM-START: NOT USED", prefix="[WARM]")
    else:
        log("WARM-START: USED", prefix="[WARM]")

    if hasattr(backbone, "print_trainable_parameters"):
        backbone.print_trainable_parameters()

    # Lightweight classifier on pooled last hidden state
    class SeqClassifier(nn.Module):
        def __init__(self, backbone, hidden, num_labels=2, dropout=0.1):
            super().__init__()
            self.backbone = backbone
            self.drop = nn.Dropout(dropout)
            self.cls = nn.Linear(hidden, num_labels)
            self.num_labels = num_labels
    
        def forward(self, input_ids=None, attention_mask=None, labels=None):
            base_model = getattr(self.backbone, "base_model", self.backbone)
            encoder = getattr(base_model, "model", base_model)  # prefer bare transformer if present
            out = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )
    
            if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                h = out.last_hidden_state
            elif getattr(out, "hidden_states", None) is not None:
                h = out.hidden_states[-1]
            else:
                # ultra-conservative fallback (shouldn’t be hit)
                h = self.backbone.get_input_embeddings()(input_ids)
    
            if attention_mask is None:
                pooled = h[:, -1]
            else:
                mask = attention_mask.unsqueeze(-1).type_as(h)
                pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
    
            logits = self.cls(self.drop(pooled))
            loss = F.cross_entropy(logits, labels.long()) if labels is not None else None
            return {"loss": loss, "logits": logits}


    hidden_size = getattr(backbone.config, "hidden_size", None) or getattr(backbone.config, "hidden_dim", None)
    assert hidden_size, "Could not infer hidden size."
    model = SeqClassifier(backbone, hidden_size, num_labels=len(id2label)).to(device)

    # Try to warm-start classifier head if label mapping matches
    try:
        if warm_used and (best_dir / "classifier.pt").exists() and (best_dir / "id2label.json").exists():
            with open(best_dir / "id2label.json") as f:
                prev_id2label = {int(k): v for k, v in json.load(f).items()}
            if prev_id2label == id2label:
                head_state = torch.load(best_dir / "classifier.pt", map_location="cpu")
                model.cls.load_state_dict(head_state["state_dict"], strict=True)
                log("WARM-START: loaded previous classifier head", prefix="[WARM]")
            else:
                warm_notes.append("label_mapping_changed→fresh_classifier_head")
                log("WARM-START: label mapping changed → new classifier head", prefix="[WARM]")
        elif warm_used:
            warm_notes.append("no_classifier_head_found")
    except Exception as e:
        warm_notes.append(f"classifier_load_error:{type(e).__name__}")

    # Persist warm-start decision for audit
    json_dump({"used": warm_used, "notes": warm_notes, "source": str(best_dir)}, save_root / "warm_start_status.json")
    _append_metric(save_root, {
        "run_id": run_id,
        "timestamp": time.time(),
        "event": "warm_start_decision",
        "phase": "setup",
        "epoch": 0,
        "step": 0,
        "warm_start_used": bool(warm_used),
        "warm_notes": ";".join(warm_notes) if warm_notes else ""
    })
    log(f"Model ready in {time.time()-t0:.1f}s | hidden_size={hidden_size}", prefix="[MODEL]")
    if warm_notes:
        log("WARM-START notes: " + ", ".join(warm_notes), prefix="[WARM]")

    if cfg.get("torch_compile", True) and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead", dynamic=True, fullgraph=False)
            log("torch.compile enabled", prefix="[OPT]")
        except Exception as e:
            log(f"torch.compile disabled: {type(e).__name__}: {e}", prefix="[OPT]")

    # Optim & sched
    steps_per_epoch = math.ceil(len(train_dl) / cfg["grad_accum"])
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup_steps = int(total_steps * cfg["warmup_ratio"])
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in trainable_params)
    opt = None
    if total_trainable < 30_000_000:
        try:
            opt = torch.optim.AdamW(
                trainable_params,
                lr=cfg["lr"],
                weight_decay=cfg["weight_decay"],
                fused=True,
            )
            log("Optimizer: AdamW (fused)", prefix="[OPT]")
        except (TypeError, RuntimeError, ValueError):
            opt = torch.optim.AdamW(
                trainable_params,
                lr=cfg["lr"],
                weight_decay=cfg["weight_decay"],
            )
            log("Optimizer: AdamW", prefix="[OPT]")
    if opt is None:
        opt = bnb.optim.PagedAdamW8bit(
            trainable_params,
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            betas=(0.9, 0.95),
            eps=1e-8
        )
        log("Optimizer: PagedAdamW8bit", prefix="[OPT]")
    sch = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)
    from torch.amp import GradScaler
    scaler = GradScaler('cuda', enabled=(cfg["precision"] == "fp16"))


    # W&B
    if cfg["wandb"]:
        import wandb
        wandb.init(project="deepfake-detect-gemma3", config=cfg, name="gemma3-1b-pt_qLoRA_seqcls")

    # Eval util
    @torch.no_grad()
    def evaluate(model: nn.Module, loader) -> Dict[str, Any]:
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        model.eval()
        total_loss, nb = 0.0, 0
        ys, yps, yhats = [], [], []
        pbar = tqdm(loader, desc="Validating", unit="batch", leave=False)
        import numpy as np
        for batch in pbar:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None: attention_mask = attention_mask.to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            from torch.amp import autocast
            with autocast('cuda', dtype=(torch.bfloat16 if cfg["precision"] == "bf16" else torch.float16)):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out["loss"]
                logits = out["logits"].float()
            prob_ai = torch.softmax(logits, dim=-1)[:, 1]
            pred = logits.argmax(-1)
            total_loss += float(loss.item()); nb += 1
            ys.append(labels.cpu()); yps.append(prob_ai.cpu()); yhats.append(pred.cpu())
            # live validation metrics on processed batches so far
            try:
                y_sofar = torch.cat(ys).numpy(); p_sofar = torch.cat(yps).numpy(); yhat_sofar = torch.cat(yhats).numpy()
                acc_sofar = accuracy_score(y_sofar, yhat_sofar)
                f1_sofar = f1_score(y_sofar, yhat_sofar, average="macro")
                try: auc_sofar = roc_auc_score(y_sofar, p_sofar)
                except Exception: auc_sofar = float("nan")
                pbar.set_postfix({
                    "loss": f"{total_loss/max(1,nb):.4f}",
                    "acc": f"{acc_sofar:.3f}",
                    "f1": f"{f1_sofar:.3f}",
                    "auc": f"{auc_sofar:.3f}",
                })
            except Exception:
                pbar.set_postfix({"loss": f"{total_loss/max(1,nb):.4f}"})
        import numpy as np
        y = torch.cat(ys).numpy(); p = torch.cat(yps).numpy(); yhat = torch.cat(yhats).numpy()
        acc = accuracy_score(y, yhat); f1m = f1_score(y, yhat, average="macro")
        try: auc = roc_auc_score(y, p)
        except Exception: auc = float("nan")
        return {"loss": total_loss / max(1, nb), "acc": acc, "f1_macro": f1m, "auroc": auc}

    # Train
    hline("[TRAIN] Start")
    log(f"epochs={cfg['epochs']} | steps/epoch≈{steps_per_epoch} | total_steps={total_steps} | warmup={warmup_steps}", prefix="[TRAIN]")
    best_f1 = -1.0
    t0 = time.time()
    global_step = 0
    n_steps_seen = 0
    total_loss_sum = 0.0
    eval_every = int(cfg.get("eval_every", 0)) or 0  # 0 = disabled

    # helper to run validation + save best
    def _do_eval(tag: str, cur_epoch: int):
        nonlocal best_f1
        model.eval()
        valm = evaluate(model, val_dl)
        elapsed = (time.time() - t0) / 60.0
        log(
            f"step={global_step} | {tag} | val_loss={valm['loss']:.4f} | "
            f"acc={valm['acc']:.4f} | f1_macro={valm['f1_macro']:.4f} | auroc={valm['auroc']:.4f} | time={elapsed:.1f}m",
            prefix="[EVAL]"
        )
        _append_metric(save_root, {
            "run_id": run_id,
            "timestamp": time.time(),
            "event": "eval",
            "phase": "eval",
            "epoch": int(cur_epoch),
            "step": int(global_step),
            "eval_tag": str(tag),
            "loss": float(valm["loss"]),
            "acc": float(valm["acc"]),
            "f1_macro": float(valm["f1_macro"]),
            "auroc": (None if (valm["auroc"] != valm["auroc"]) else float(valm["auroc"])),
            "warm_start_used": bool(warm_used)
        })
        if cfg["wandb"]:
            import wandb
            wandb.log({f"val/{k}": v for k, v in valm.items()} | {"time/min": elapsed, "global_step": global_step})

        # Save best by F1
        if valm["f1_macro"] > best_f1:
            best_f1 = valm["f1_macro"]
            best_dir.mkdir(parents=True, exist_ok=True)
            if cfg["use_lora"]:
                model.backbone.save_pretrained(str(best_dir / "lora_adapter"))
            else:
                base.save_pretrained(str(best_dir / "backbone"))
            torch.save(
                {"state_dict": model.cls.state_dict(), "hidden_size": hidden_size, "num_labels": model.num_labels},
                best_dir / "classifier.pt",
            )
            json_dump(id2label, best_dir / "id2label.json")
            json_dump(cfg, best_dir / "cfg.json")
            log(f"Saved new best → {best_dir} (f1_macro={best_f1:.4f})", prefix="[SAVE]")

        torch.cuda.empty_cache(); gc.collect()
        model.train()

    log_interval = max(1, int(cfg.get("log_interval", 50)))
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_examples = 0
        interval_steps = 0
        batches = tqdm(train_dl, desc=f"Epoch {epoch}/{cfg['epochs']}", unit="batch")

        opt.zero_grad(set_to_none=True)

        for step, batch in enumerate(batches, start=1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None: attention_mask = attention_mask.to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=(torch.bfloat16 if cfg["precision"] == "bf16" else torch.float16)):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                raw_loss = float(out["loss"].detach().item())
                loss = out["loss"] / cfg["grad_accum"]
                logits_detached = out["logits"].detach()
                preds = logits_detached.argmax(-1)

            if cfg["precision"] == "fp16":
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % cfg["grad_accum"] == 0:
                if cfg["precision"] == "fp16":
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                    scaler.step(opt); scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                    opt.step()
                sch.step()
                opt.zero_grad(set_to_none=True)

            running_loss += raw_loss
            running_correct += (preds == labels).sum().item()
            running_examples += labels.numel()
            interval_steps += 1
            global_step += 1
            n_steps_seen += 1
            total_loss_sum += raw_loss
            _append_metric(save_root, {
                "run_id": run_id,
                "timestamp": time.time(),
                "event": "train_step",
                "phase": "train",
                "epoch": epoch,
                "step": int(global_step),
                "batch": int(step),
                "loss": raw_loss,
                "lr": float(sch.get_last_lr()[0]),
                "model_id": cfg["model_id"],
                "dataset": cfg["dataset_id"],
                "batch_size": cfg["batch_size"],
                "grad_accum": cfg["grad_accum"],
                "warm_start_used": bool(warm_used)
            })

            if step % log_interval == 0:
                denom = max(1, interval_steps)
                avg_loss = running_loss / denom
                acc = running_correct / max(1, running_examples)
                batches.set_postfix({
                    "train_loss": f"{avg_loss:.4f}",
                    "acc": f"{acc:.3f}",
                    "lr": f"{sch.get_last_lr()[0]:.2e}"
                })
                if cfg["wandb"]:
                    import wandb
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/lr": sch.get_last_lr()[0],
                        "train/acc": acc,
                        "train/step": (epoch-1)*steps_per_epoch + step,
                        "global_step": global_step,
                    })
                running_loss = 0.0
                running_correct = 0
                running_examples = 0
                interval_steps = 0

            # ---- mid-epoch evaluation every N batches ----
            if eval_every and (global_step % eval_every == 0):
                _do_eval("mid-epoch", epoch)

        # always eval at epoch end (covers tail that didn't hit eval_every)
        _do_eval(f"epoch={epoch} end", epoch)

    log("Training complete.", prefix="[TRAIN]")
    _append_metric(save_root, {
        "run_id": run_id,
        "timestamp": time.time(),
        "event": "train_end",
        "phase": "train",
        "epoch": int(cfg["epochs"]),
        "step": int(global_step),
        "total_loss": float(total_loss_sum),
        "n_steps": int(n_steps_seen),
        "warm_start_used": bool(warm_used)
    })
    evaluate_and_save(cfg, best_dir, ds_tok, val_dl, collator, id2label)


def evaluate_and_save(cfg, best_dir: Path, ds_tok, val_dl, collator, id2label):
    import torch, numpy as np
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    try:
        from transformers import BitsAndBytesConfig
    except Exception:
        from transformers.utils.quantization_config import BitsAndBytesConfig
    from peft import PeftModel, prepare_model_for_kbit_training
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    assert best_dir.exists(), f"Best dir not found: {best_dir}"
    device = torch.device("cuda")
    base_dtype = torch.bfloat16 if cfg["precision"] == "bf16" else torch.float16
    # Light quant reload
    try:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=base_dtype, bnb_4bit_use_double_quant=True)
    except Exception:
        quant = None

    log("Reloading best checkpoint …", prefix="[EVAL]")

    if cfg["use_lora"]:
        base = safe_load_backbone(
            cfg["model_id"],
            base_dtype,
            quant,
            attn_impl=cfg.get("attn_impl"),
            gradient_checkpointing=False,
        )
        backbone = PeftModel.from_pretrained(base, str(best_dir / "lora_adapter"), is_trainable=True)
        # Keep numerically sensitive ops (e.g., LayerNorm) in fp32 for k-bit eval
        backbone = prepare_model_for_kbit_training(backbone, use_gradient_checkpointing=False)
    else:
        backbone_dir = best_dir / "backbone"
        if backbone_dir.exists():
            backbone = AutoModelForCausalLM.from_pretrained(
                str(backbone_dir),
                torch_dtype=base_dtype,
                device_map={"": 0}
            )
        else:
            print("NO FINE-TUNING APPLIED")
            backbone = safe_load_backbone(
                cfg["model_id"],
                base_dtype,
                quant,
                attn_impl=cfg.get("attn_impl"),
                gradient_checkpointing=False,
            )
        backbone.config.output_hidden_states = False
        backbone.config.use_cache = False
        # Same stabilization even without LoRA
        backbone = prepare_model_for_kbit_training(backbone, use_gradient_checkpointing=False)

    # Debug: attention implementation + sample LayerNorm parameter dtypes
    try:
        attn_impl_dbg = getattr(backbone.config, "attn_implementation", getattr(backbone.config, "_attn_implementation", "unknown"))
    except Exception:
        attn_impl_dbg = "unknown"
    log(f"[EVAL] compute_dtype={base_dtype} | attn_impl={attn_impl_dbg}", prefix="")
    try:
        ln_dtypes = []
        for name, mod in backbone.named_modules():
            if "norm" in name.lower():
                for p in mod.parameters(recurse=False):
                    ln_dtypes.append(str(p.dtype)); break
            if len(ln_dtypes) >= 2:
                break
        if ln_dtypes:
            log(f"[EVAL] sample LayerNorm dtypes: {ln_dtypes}", prefix="")
    except Exception:
        pass

    import torch.nn as nn
    class SeqClassifier(nn.Module):
        def __init__(self, backbone, hidden, num_labels=2):
            super().__init__()
            self.backbone = backbone
            self.cls = nn.Linear(hidden, num_labels)
            self.num_labels = num_labels
    
        def forward(self, input_ids=None, attention_mask=None):
            base_model = getattr(self.backbone, "base_model", self.backbone)
            encoder = getattr(base_model, "model", base_model)
            out = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )
    
            if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                h = out.last_hidden_state
            elif getattr(out, "hidden_states", None) is not None:
                h = out.hidden_states[-1]
            else:
                h = self.backbone.get_input_embeddings()(input_ids)
    
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).type_as(h)
                pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            else:
                pooled = h[:, -1]
            return self.cls(pooled)


    head_state = torch.load(best_dir / "classifier.pt", map_location=device)
    hidden_size = int(head_state.get("hidden_size") or getattr(backbone.config, "hidden_size", None) or getattr(backbone.config, "hidden_dim"))
    num_labels = int(head_state.get("num_labels", 2))

    model = SeqClassifier(backbone, hidden_size, num_labels=num_labels).to(device)
    model.cls.load_state_dict(head_state["state_dict"], strict=True)
    model.eval()

    # Build an eval DataLoader that preserves raw_text for error analysis
    def collate_keep_text(features):
        # keep originals for error analysis
        texts = [f.get("raw_text", "") for f in features]
        # pass only tensor-able fields to the HF collator
        TENSOR_KEYS = {"input_ids", "attention_mask", "labels", "token_type_ids"}
        clean = [{k: v for k, v in f.items() if k in TENSOR_KEYS} for f in features]
        batch = collator(clean)
        batch["raw_text"] = texts
        return batch

    eval_ds = ds_tok["validation"]
    cpu_count = os.cpu_count() or 8
    if cpu_count <= 1:
        eval_workers = 1
    else:
        eval_workers = min(16, max(2, cpu_count))
    eval_dl_kwargs = {
        "pin_memory": True,
        "pin_memory_device": "cuda",
        "num_workers": eval_workers,
        "persistent_workers": eval_workers > 0,
    }
    if eval_workers > 0:
        eval_dl_kwargs["prefetch_factor"] = 4

    loader = DataLoader(
        eval_ds,
        batch_size=max(1, cfg["batch_size"]*2),
        shuffle=False,
        collate_fn=collate_keep_text,
        **eval_dl_kwargs,
    )
    log(f"[EVAL] Using split=validation | n={len(eval_ds)} | batch_size={max(1, cfg['batch_size']*2)}", prefix="")

    ys, yps, yhats, texts_all = [], [], [], []
    pbar = tqdm(loader, desc="Eval(best)", unit="batch", leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        raw_text = batch.get("raw_text", [])
        from torch.amp import autocast
        # Disable autocast for fp16 (pre-Ampere) to avoid instability; keep it on for bf16.
        with torch.no_grad(), autocast('cuda', dtype=base_dtype, enabled=(cfg["precision"] == "bf16")):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).float()

        # Debug & sanitize logits
        if not torch.isfinite(logits).all():
            n_nan = int(torch.isnan(logits).sum().item())
            n_pos = int(torch.isposinf(logits).sum().item())
            n_neg = int(torch.isneginf(logits).sum().item())
            log(f"[EVAL] Detected invalid logits → nan={n_nan}, +inf={n_pos}, -inf={n_neg}", prefix="")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)

        prob_ai = torch.softmax(logits, dim=-1)[:, 1]
        if not torch.isfinite(prob_ai).all():
            n_bad = int((~torch.isfinite(prob_ai)).sum().item())
            log(f"[EVAL] Detected {n_bad} non-finite probabilities → fixing with nan_to_num+clamp", prefix="")
            prob_ai = torch.nan_to_num(prob_ai, nan=0.5, posinf=1.0, neginf=0.0)
        prob_ai = prob_ai.clamp(0.0, 1.0)

        pred = logits.argmax(-1)
        ys.append(labels.cpu()); yps.append(prob_ai.cpu()); yhats.append(pred.cpu()); texts_all.extend(raw_text)

    y = torch.cat(ys).numpy()
    p = torch.cat(yps).numpy()
    yhat = torch.cat(yhats).numpy()

    # Print class counts for the evaluated split
    uniq, cnts = np.unique(y, return_counts=True)
    counts_map = {int(k): int(v) for k, v in zip(uniq, cnts)}
    label_names = {k: id2label.get(int(k), str(k)) for k in counts_map.keys()}
    log(f"[EVAL] Class counts (validation): " + str({label_names[k]: counts_map[k] for k in counts_map}), prefix="")

    # Metrics (guard AUC when single-class) — ARGMAX baseline
    acc = accuracy_score(y, yhat)
    f1m = f1_score(y, yhat, average="macro")
    if len(uniq) == 2:
        try:
            auc = roc_auc_score(y, p)
        except Exception:
            auc = float("nan")
    else:
        log("[EVAL] Only one class present in labels; ROC-AUC is undefined. Increase max_val or ensure stratification.", prefix="")
        auc = float("nan")

    log(f"[VAL/ARGMAX] acc={acc:.4f} | f1_macro={f1m:.4f} | auroc={auc:.4f}", prefix="[BEST]")
    _append_metric(best_dir.parent, {
        "run_id": cfg.get("run_id"),
        "timestamp": time.time(),
        "event": "eval_best",
        "phase": "eval",
        "epoch": None,
        "step": None,
        "acc": float(acc),
        "f1_macro": float(f1m),
        "auroc": (None if (auc != auc) else float(auc))
    })

    # ---- Threshold sweep on validation (maximize macro-F1 over P(class==1)) ----
    ts = np.linspace(0.0, 1.0, 101)
    best_t, best_f1 = 0.5, -1.0
    best_acc = None
    for t in ts:
        yhat_t = (p >= t).astype(int)
        f1_t = f1_score(y, yhat_t, average="macro")
        if f1_t > best_f1:
            best_f1 = f1_t
            best_t = float(t)
            best_acc = accuracy_score(y, yhat_t)

    threshold_info = {
        "prob_is_class_1": True,  # p = softmax(logits)[:, 1]
        "class1_name": id2label.get(1, "1"),
        "best_threshold": best_t,
        "val_f1_macro_at_best": float(best_f1),
        "val_acc_at_best": float(best_acc),
        "search_grid": {"start": 0.0, "stop": 1.0, "num": 101},
        "note": "If your positive class is AI and mapped to label 0, remember p is P(class==1)."
    }
    json_dump(threshold_info, best_dir / "threshold.json")
    log(f"[VAL/THRESH] t*={best_t:.2f} | f1_macro={best_f1:.4f} | acc={best_acc:.4f}", prefix="[BEST]")

    # Error analysis: print up to 20 misclassified examples (by argmax) with confidence p(class==1)
    mis_idx = np.where(yhat != y)[0].tolist()
    log(f"[EVAL] Misclassified examples: {len(mis_idx)}", prefix="")
    max_print = min(20, len(mis_idx))
    for i in range(max_print):
        j = mis_idx[i]
        gold = int(y[j]); pred_lbl = int(yhat[j]); conf = float(p[j])
        gold_name = id2label.get(gold, str(gold))
        pred_name = id2label.get(pred_lbl, str(pred_lbl))
        snippet = texts_all[j].replace("\n", " ")
        if len(snippet) > 300:
            snippet = snippet[:300]
        log(f"[ERR] gold={gold_name} pred={pred_name} p(class==1)={conf:.4f} | text={snippet}", prefix="")

    # Save predictions and both metric flavors (argmax + thresholded)
    pred_dir = best_dir / "preds"; pred_dir.mkdir(parents=True, exist_ok=True)
    np.save(pred_dir / "val_labels.npy", y)
    np.save(pred_dir / "val_prob_ai.npy", p)  # NOTE: p is P(class==1)
    np.save(pred_dir / "val_pred.npy", yhat)  # argmax predictions

    # Argmax baseline
    json_dump({"acc": acc, "f1_macro": f1m, "auroc": auc, "class_counts": counts_map}, pred_dir / "metrics.json")
    # Calibrated at best threshold
    json_dump({
        "threshold": best_t,
        "acc": float(best_acc),
        "f1_macro": float(best_f1)
    }, pred_dir / "metrics_thresh.json")

    log(f"Saved preds + metrics (+threshold) → {pred_dir}", prefix="[SAVE]")

    # Also evaluate extra splits (test / OOD) with the same guards
    extra = [k for k in ds_tok.keys() if k not in ("train", "validation")]
    for split in extra:
        ds_split = ds_tok[split]
        loader = DataLoader(
            ds_split,
            batch_size=max(1, cfg["batch_size"]*2),
            shuffle=False,
            collate_fn=collate_keep_text,
            **eval_dl_kwargs,
        )
        ys, yps, yhats = [], [], []
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device, non_blocking=True)
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=base_dtype, enabled=(cfg["precision"] == "bf16")):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).float()
            if not torch.isfinite(logits).all():
                logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)
            prob_ai = torch.softmax(logits, dim=-1)[:, 1]
            if not torch.isfinite(prob_ai).all():
                prob_ai = torch.nan_to_num(prob_ai, nan=0.5, posinf=1.0, neginf=0.0)
            prob_ai = prob_ai.clamp(0.0, 1.0)
            pred = logits.argmax(-1)
            ys.append(batch["labels"].cpu()); yps.append(prob_ai.cpu()); yhats.append(pred.cpu())

        import numpy as np
        y = torch.cat(ys).numpy(); p = torch.cat(yps).numpy(); yhat = torch.cat(yhats).numpy()
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        # ARGMAX baseline
        acc = accuracy_score(y, yhat); f1m = f1_score(y, yhat, average="macro")
        uniq, _ = np.unique(y, return_counts=True)
        if len(uniq) == 2:
            try:
                auc = roc_auc_score(y, p)
            except Exception:
                auc = float("nan")
        else:
            auc = float("nan")

        # THRESHOLDED metrics (use validation-derived threshold if available)
        thr_path = best_dir / "threshold.json"
        if thr_path.exists():
            thr = json.load(open(thr_path))["best_threshold"]
        else:
            thr = 0.5
        yhat_thr = (p >= float(thr)).astype(int)
        acc_thr = accuracy_score(y, yhat_thr)
        f1_thr = f1_score(y, yhat_thr, average="macro")

        # Save both flavors
        out_dir = best_dir / "preds"
        json_dump({"acc": acc, "f1_macro": f1m, "auroc": auc}, out_dir / f"{split}_metrics.json")
        json_dump({"threshold": float(thr), "acc": float(acc_thr), "f1_macro": float(f1_thr)}, out_dir / f"{split}_metrics_thresh.json")


def maybe_login(do_login: bool):
    if not do_login: return
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        log("--login requested but no $HUGGINGFACE_TOKEN/$HF_TOKEN found; skipping.", prefix="[HF]")
        return
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        log("Hugging Face login ok.", prefix="[HF]")
    except Exception as e:
        log(f"HF login failed: {e}", prefix="[HF]")

def main():
    # 1) Very first: bootstrap deps, env, caches
    ensure_core_packages()
    setup_env_and_caches()
    disable_hf_transfer()

    # 2) GPU probe + precision
    precision = gpu_probe()

    # 3) Multi-run: Gemma first, then LLaMA-3-8B-Instruct
    run_list = RUNS if "RUNS" in globals() else [DEFAULT_CFG]
    for i, overrides in enumerate(run_list, start=1):
        cfg = {"project": "deepfake-detect", "precision": precision, **DEFAULT_CFG, **overrides}
        cfg["run_id"] = cfg.get("run_id") or time.strftime("%Y%m%d-%H%M%S")

        hline(f"[CFG] Run {i} → {cfg['model_id']}")
        log(json.dumps(cfg, indent=2))

        # Seed per run (keeps comparability, still deterministic)
        set_seed(cfg["seed"])

        # Train/Eval this run
        try:
            train_eval(cfg)
        except AssertionError as e:
            log(f"Assertion error: {e}", prefix="[ERR]")
        except Exception as e:
            log(f"Unhandled exception: {e}", prefix="[ERR]")
            raise

if __name__ == "__main__":
    main()
