# pip install --upgrade huggingface_hub
# hf auth login

import os, sys, time, json, math, re, gc, importlib, pkgutil, subprocess
from pathlib import Path
from typing import Any, Dict
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
    "batch_size": 2,
    "grad_accum": 1,
    "max_grad_norm": 1.0,
    "use_lora": True,  # QLoRA on by default
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
    "resume": False,
}

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

def safe_load_backbone(model_id: str, base_dtype, quant, *, device_map=None, trust_remote_code: bool = False):
    if device_map is None:
        device_map = {"": 0}

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if hasattr(cfg, "auto_map"):
        cfg.auto_map = None

    # >>> disable PEFT autoload only for this call <<<
    with _disable_peft_autoload():
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=cfg,
            quantization_config=quant,
            torch_dtype=(None if quant else base_dtype),
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            # Gemma-3 prefers eager attention for training
            attn_implementation="eager",
        )

    # training-time prefs
    model.config.output_hidden_states = True
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
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
        collator = DataCollatorWithPadding(tokenizer=tok)
        return ds_tok, collator, id2label

    hline("[DATA] Downloading/reading dataset")
    t0 = time.time()
    ds = load_dataset(cfg["dataset_id"])
    log(f"Splits: {list(ds.keys())}")
    log(f"Train size: {len(ds['train'])}" + (f", Test size: {len(ds['test'])}" if "test" in ds else ""))

    # ensure validation split
    keys = set(ds.keys())
    if "validation" not in keys and "val" not in keys:
        log("No validation split → creating 10% from train", prefix="[DATA]")
        split = ds["train"].train_test_split(test_size=0.1, seed=cfg["seed"])
        from datasets import DatasetDict as HF_DatasetDict
        ds = HF_DatasetDict(train=split["train"], validation=split["test"], **({"test": ds["test"]} if "test" in keys else {}))
    elif "val" in keys:
        ds = DatasetDict(train=ds["train"], validation=ds["val"], **({"test": ds["test"]} if "test" in keys else {}))

    # pick columns
    cand_text = ["text","content","document","body","sentence","prompt","input","inputs","article"]
    cand_label = ["label","labels","target","class","gold","source"]
    cols = list(ds["train"].column_names)
    text_col = next((c for c in cand_text if c in cols), cols[0])
    label_col = next((c for c in cand_label if c in cols), None)
    assert label_col is not None, f"No label-like column found in {cols}"
    log(f"text_col={text_col} | label_col={label_col}", prefix="[DATA]")

    feat = ds["train"].features.get(label_col)
    if isinstance(feat, ClassLabel):
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

    # subsample
    auto_max_train = int(os.environ.get("AUTO_MAX_TRAIN", "20000"))
    auto_max_val   = int(os.environ.get("AUTO_MAX_VAL",   "2000"))
    disable_auto   = os.environ.get("AUTO_SUBSAMPLE", "1") in {"0", "false", "False"}
    
    def maybe_take(ds_split, nmax, fallback):
        limit = nmax if (nmax and nmax > 0) else (0 if disable_auto else fallback)
        if limit and len(ds_split) > limit:
            log(f"Auto-subsample → {limit} rows (set AUTO_SUBSAMPLE=0 to disable)", prefix="[DATA]")
            return ds_split.shuffle(seed=cfg["seed"]).select(range(limit))
        return ds_split
    
    ds["train"] = maybe_take(ds["train"], max_train, auto_max_train)
    ds["validation"] = maybe_take(ds["validation"], max_val, auto_max_val)
    log(f"Using: train={len(ds['train'])}, val={len(ds['validation'])}", prefix="[DATA]")

    tok = safe_load_tokenizer(cfg["model_id"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    max_len = int(cfg["max_length"])

    def tok_fn(batch):
        enc = tok(batch[text_col], truncation=True, max_length=max_len)
        enc["labels"] = [map_label_value(v) for v in batch[label_col]]
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
    collator = DataCollatorWithPadding(tokenizer=tok)
    from collections import Counter
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

    device = torch.device("cuda")
    save_root = Path(cfg["save_dir"]); save_root.mkdir(parents=True, exist_ok=True)
    best_dir = save_root / "best"

    # Data
    ds_tok, collator, id2label = load_and_tokenize(cfg, save_root, cfg["max_train"], cfg["max_val"])
    train_dl = DataLoader(
        ds_tok["train"],
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=collator,
        pin_memory=True,
        num_workers=0,
        persistent_workers=False,
    )

    val_bs = max(1, cfg["batch_size"] * 2)
    val_dl = DataLoader(
        ds_tok["validation"],
        batch_size=val_bs,
        shuffle=False,
        collate_fn=collator,
        pin_memory=True,
        num_workers=0,
        persistent_workers=False,
    )

    # Early paths
    if cfg["eval_only"] and not best_dir.exists():
        log("eval-only requested but no checkpoint found → exit", prefix="[EVAL]")
        return
    if cfg["resume"] and best_dir.exists() and not cfg["eval_only"]:
        log("best checkpoint exists and --resume set → evaluate only", prefix="[EVAL]")
        return evaluate_and_save(cfg, best_dir, ds_tok, val_dl, collator, id2label)

    # Precision & GPU info
    import torch as _t
    if cfg["precision"] is None:
        cfg["precision"] = "bf16" if _t.cuda.is_bf16_supported() else "fp16"
    _t.set_float32_matmul_precision("medium")
    log(f"Precision: {cfg['precision']}", prefix="[TRAIN]")

    # Quantization choice (no regex, no try/except)
    from packaging.version import Version
    
    base_dtype = _t.bfloat16 if cfg["precision"] == "bf16" else _t.float16
    
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


    # Model load (progress printed by HF/transformers)
    t0 = time.time()
    log(f"Loading base model: {cfg['model_id']}", prefix="[MODEL]")
    base = safe_load_backbone(cfg["model_id"], base_dtype, quant)
    
    base.config.output_hidden_states = True
    base.config.use_cache = False
    if hasattr(base, "gradient_checkpointing_enable"):
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    from peft import prepare_model_for_kbit_training
    base = prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    # ensure inputs require grads for gradient checkpointing + k-bit training
    if hasattr(base, "enable_input_require_grads"):
        base.enable_input_require_grads()
    hidden_size = getattr(base.config, "hidden_size", None) or getattr(base.config, "hidden_dim", None)
    assert hidden_size, "Could not infer hidden size."
    log(f"Model loaded in {time.time()-t0:.1f}s | hidden_size={hidden_size}", prefix="[MODEL]")

    # LoRA adapters
    backbone = base
    if cfg["use_lora"]:
        log("Injecting LoRA adapters", prefix="[LoRA]")
        from peft import LoraConfig, get_peft_model
        lcfg = LoraConfig(
            r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
            target_modules=cfg["target_mods"].split(","), bias="none", task_type="CAUSAL_LM",
        )
        backbone = get_peft_model(base, lcfg)
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
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                                output_hidden_states=True, return_dict=True)

            h = out.hidden_states[-1]
            if attention_mask is None:
                pooled = h[:, -1]
            else:
                mask = attention_mask.unsqueeze(-1).type_as(h)
                pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            logits = self.cls(self.drop(pooled))
            loss = F.cross_entropy(logits, labels.long()) if labels is not None else None
            return {"loss": loss, "logits": logits}

    model = SeqClassifier(backbone, hidden_size, num_labels=2).to(device)

    # Optim & sched
    steps_per_epoch = math.ceil(len(train_dl) / cfg["grad_accum"])
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup_steps = int(total_steps * cfg["warmup_ratio"])
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = bnb.optim.PagedAdamW8bit(
        trainable_params,
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        betas=(0.9, 0.95),
        eps=1e-8
    )
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
        import numpy as np, torch as _t
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
                y_sofar = _t.cat(ys).numpy(); p_sofar = _t.cat(yps).numpy(); yhat_sofar = _t.cat(yhats).numpy()
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
        import numpy as np, torch as _t
        y = _t.cat(ys).numpy(); p = _t.cat(yps).numpy(); yhat = _t.cat(yhats).numpy()
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
    eval_every = int(cfg.get("eval_every", 0)) or 0  # 0 = disabled

    # helper to run validation + save best
    def _do_eval(tag: str):
        nonlocal best_f1
        model.eval()
        valm = evaluate(model, val_dl)
        elapsed = (time.time() - t0) / 60.0
        log(
            f"step={global_step} | {tag} | val_loss={valm['loss']:.4f} | "
            f"acc={valm['acc']:.4f} | f1_macro={valm['f1_macro']:.4f} | auroc={valm['auroc']:.4f} | time={elapsed:.1f}m",
            prefix="[EVAL]"
        )
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

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running = 0.0
        ys_run, yps_run, yhats_run = [], [], []  # epoch-to-date for live train metrics
        batches = tqdm(train_dl, desc=f"Epoch {epoch}/{cfg['epochs']}", unit="batch")

        opt.zero_grad(set_to_none=True)

        for step, batch in enumerate(batches, start=1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None: attention_mask = attention_mask.to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=(torch.bfloat16 if cfg["precision"] == "bf16" else torch.float16)):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out["loss"] / cfg["grad_accum"]
                logits = out["logits"].detach().float()
                prob_ai = torch.softmax(logits, dim=-1)[:, 1]
                pred = logits.argmax(-1)
            # accumulate for live train metrics
            ys_run.append(labels.detach().cpu())
            yps_run.append(prob_ai.detach().cpu())
            yhats_run.append(pred.detach().cpu())

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

            running += float(loss.item())
            global_step += 1

            if step % (cfg["grad_accum"] * 5) == 0:
                avg = running / (cfg["grad_accum"] * 5)
                import torch as _t, numpy as _np
                from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
                y_tr = _t.cat(ys_run).numpy()
                p_tr = _t.cat(yps_run).numpy()
                yhat_tr = _t.cat(yhats_run).numpy()
                acc_tr = accuracy_score(y_tr, yhat_tr)
                f1_tr = f1_score(y_tr, yhat_tr, average="macro")
                try: auc_tr = roc_auc_score(y_tr, p_tr)
                except Exception: auc_tr = float("nan")
                batches.set_postfix({
                    "train_loss": f"{avg:.4f}",
                    "acc": f"{acc_tr:.3f}",
                    "f1": f"{f1_tr:.3f}",
                    "auc": f"{auc_tr:.3f}",
                    "lr": f"{sch.get_last_lr()[0]:.2e}"
                })
                if cfg["wandb"]:
                    import wandb
                    wandb.log({
                        "train/loss": avg,
                        "train/lr": sch.get_last_lr()[0],
                        "train/acc": acc_tr,
                        "train/f1_macro": f1_tr,
                        "train/auroc": auc_tr,
                        "train/step": (epoch-1)*steps_per_epoch + step,
                        "global_step": global_step,
                    })
                running = 0.0

            # ---- mid-epoch evaluation every N batches ----
            if eval_every and (global_step % eval_every == 0):
                _do_eval("mid-epoch")

        # always eval at epoch end (covers tail that didn't hit eval_every)
        _do_eval(f"epoch={epoch} end")

    log("Training complete.", prefix="[TRAIN]")
    evaluate_and_save(cfg, best_dir, ds_tok, val_dl, collator, id2label)

def evaluate_and_save(cfg, best_dir: Path, ds_tok, val_dl, collator, id2label):
    import torch, numpy as np
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    try:
        from transformers import BitsAndBytesConfig
    except Exception:
        from transformers.utils.quantization_config import BitsAndBytesConfig
    from peft import PeftModel
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
        base = safe_load_backbone(cfg["model_id"], base_dtype, quant)
        base.config.output_hidden_states = True; base.config.use_cache = False
        backbone = PeftModel.from_pretrained(base, str(best_dir / "lora_adapter"))  # str() avoids HFValidationError
    else:
        backbone_dir = best_dir / "backbone"
        if backbone_dir.exists():
            # Reload the finetuned backbone we saved
            backbone = AutoModelForCausalLM.from_pretrained(
                str(backbone_dir),
                torch_dtype=base_dtype,
                device_map={"": 0}
            )
        else:
            # Fallback: use the base model from the hub (won’t include finetuning)
            print("NO FINE-TUNING APPLIED")
            backbone = safe_load_backbone(cfg["model_id"], base_dtype, quant)
        backbone.config.output_hidden_states = True
        backbone.config.use_cache = False


    import torch.nn as nn, json as _json
    class SeqClassifier(nn.Module):
        def __init__(self, backbone, hidden, num_labels=2):
            super().__init__()
            self.backbone = backbone
            self.cls = nn.Linear(hidden, num_labels)
            self.num_labels = num_labels
        def forward(self, input_ids=None, attention_mask=None):
            out = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            h = out.hidden_states[-1]
            mask = attention_mask.unsqueeze(-1).type_as(h) if attention_mask is not None else None
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0) if mask is not None else h[:, -1]
            return self.cls(pooled)

    # Load classifier metadata and weights saved during training
    head_state = torch.load(best_dir / "classifier.pt", map_location=device)
    hidden_size = int(head_state.get("hidden_size") or getattr(backbone.config, "hidden_size", None) or getattr(backbone.config, "hidden_dim"))
    num_labels = int(head_state.get("num_labels", 2))

    model = SeqClassifier(backbone, hidden_size, num_labels=num_labels).to(device)
    model.cls.load_state_dict(head_state["state_dict"], strict=True)
    model.eval()

    loader = val_dl if isinstance(val_dl, DataLoader) else DataLoader(
        ds_tok["validation"],
        batch_size=max(1, cfg["batch_size"]*2),
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        persistent_workers=False,
    )

    ys, yps, yhats = [], [], []
    pbar = tqdm(loader, desc="Eval(best)", unit="batch", leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None: attention_mask = attention_mask.to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        from torch.amp import autocast
        with torch.no_grad(), autocast('cuda', dtype=base_dtype):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).float()
        prob_ai = torch.softmax(logits, dim=-1)[:, 1]
        pred = logits.argmax(-1)
        ys.append(labels.cpu()); yps.append(prob_ai.cpu()); yhats.append(pred.cpu())

    y = torch.cat(ys).numpy(); p = torch.cat(yps).numpy(); yhat = torch.cat(yhats).numpy()
    acc = accuracy_score(y, yhat); f1m = f1_score(y, yhat, average="macro")
    try: auc = roc_auc_score(y, p)
    except Exception: auc = float("nan")
    log(f"acc={acc:.4f} | f1_macro={f1m:.4f} | auroc={auc:.4f}", prefix="[BEST]")

    pred_dir = best_dir / "preds"; pred_dir.mkdir(parents=True, exist_ok=True)
    np.save(pred_dir / "val_labels.npy", y)
    np.save(pred_dir / "val_prob_ai.npy", p)
    np.save(pred_dir / "val_pred.npy", yhat)
    json_dump({"acc": acc, "f1_macro": f1m, "auroc": auc}, pred_dir / "metrics.json")
    log(f"Saved preds + metrics → {pred_dir}", prefix="[SAVE]")

    # --- Also evaluate extra splits present (test / OOD) ---
    from torch.utils.data import DataLoader
    extra = [k for k in ds_tok.keys() if k not in ("train", "validation")]
    for split in extra:
        loader = DataLoader(ds_tok[split], batch_size=max(1, cfg["batch_size"]*2),
                            shuffle=False, collate_fn=collator)
        ys, yps, yhats = [], [], []
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None: attention_mask = attention_mask.to(device, non_blocking=True)
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=base_dtype):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).float()
            prob_ai = torch.softmax(logits, dim=-1)[:, 1]
            pred = logits.argmax(-1)
            ys.append(batch["labels"].cpu()); yps.append(prob_ai.cpu()); yhats.append(pred.cpu())
        import numpy as _np, torch as _t
        y = _t.cat(ys).numpy(); p = _t.cat(yps).numpy(); yhat = _t.cat(yhats).numpy()
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        acc = accuracy_score(y, yhat); f1m = f1_score(y, yhat, average="macro")
        try: auc = roc_auc_score(y, p)
        except Exception: auc = float("nan")
        json_dump({"acc": acc, "f1_macro": f1m, "auroc": auc}, best_dir / "preds" / f"{split}_metrics.json")


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

    # 3) Config
    cfg = {"project": "deepfake-detect-gemma3", "precision": precision, **DEFAULT_CFG}
    hline("[CFG]")
    log(json.dumps(cfg, indent=2))

    # 5) Seed
    set_seed(cfg["seed"])

    # 6) Train/Eval
    try:
        train_eval(cfg)
    except AssertionError as e:
        log(f"Assertion error: {e}", prefix="[ERR]")
    except Exception as e:
        log(f"Unhandled exception: {e}", prefix="[ERR]")
        raise

if __name__ == "__main__":
    main()
