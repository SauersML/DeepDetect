import os, sys, json, math, time, random, gc, re, argparse, subprocess, importlib
from pathlib import Path
from typing import Dict, Any, Optional

# ---------------------------
# [0] Paths, caches, env hardening
# ---------------------------
HOME = os.path.expanduser("~")
os.environ["PATH"] = f"{HOME}/.local/bin:" + os.environ.get("PATH", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

TMPDIR = os.environ.get("TMPDIR") or f"/tmp/{os.environ.get('USER','user')}"
HF_HOME = f"{TMPDIR}/hf_home"
HF_HUB_CACHE = f"{TMPDIR}/hf_hub"
HF_DATASETS_CACHE = f"{TMPDIR}/hf_datasets"
for d in (HF_HOME, HF_HUB_CACHE, HF_DATASETS_CACHE):
    Path(d).mkdir(parents=True, exist_ok=True)
os.environ.update({
    "HF_HOME": HF_HOME,
    "HF_HUB_CACHE": HF_HUB_CACHE,
    "HF_DATASETS_CACHE": HF_DATASETS_CACHE,
})

def log(*a, **k): print(*a, **k, flush=True)

# ---------------------------
# [1] CLI
# ---------------------------
def build_argparser():
    p = argparse.ArgumentParser(description="Gemma-3 1B PT • MAGE (AI-text detect) with (Q)LoRA")
    p.add_argument("--model-id", default="google/gemma-3-1b-pt")
    p.add_argument("--dataset-id", default="yaful/MAGE")
    p.add_argument("--save-dir", default="./outputs/gemma3-1b-pt-mage")
    p.add_argument("--seed", type=int, default=5541)
    p.add_argument("--precision", choices=["bf16","fp16"], default=None, help="auto-detect if None")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--use-lora", action="store_true", default=True)
    p.add_argument("--no-lora", dest="use_lora", action="store_false")
    p.add_argument("--target-mods", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--max-train", type=int, default=0, help="0 = use full train")
    p.add_argument("--max-val", type=int, default=0, help="0 = use full val")
    p.add_argument("--login", action="store_true", help="call huggingface_hub.login() using $HUGGINGFACE_TOKEN")
    p.add_argument("--resume", action="store_true", help="skip train if best checkpoint exists")
    p.add_argument("--wandb", action="store_true", help="enable Weights & Biases logging")
    p.add_argument("--eval-only", action="store_true", help="load best and run eval/predict only")
    return p

# ---------------------------
# [2] Torch / GPU probe
# ---------------------------
def gpu_probe():
    import torch
    assert torch.cuda.is_available(), "CUDA GPU not visible."
    name = torch.cuda.get_device_name(0)
    cc = ".".join(map(str, torch.cuda.get_device_capability(0)))
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    use_bf16 = torch.cuda.is_bf16_supported()
    dtype = "bf16" if use_bf16 else "fp16"
    log(f"[GPU] {name} | CC {cc} | VRAM~{vram:.1f} GB | preferred_precision={dtype}")
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        if out.returncode == 0: log(out.stdout.strip())
    except Exception:
        pass
    return dtype

# ---------------------------
# [3] Kill hf_transfer in-process (avoids needing hf_transfer pkg)
# ---------------------------
def disable_hf_transfer():
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    try:
        import huggingface_hub
        from huggingface_hub import constants, file_download
        constants.HF_HUB_ENABLE_HF_TRANSFER = False
        importlib.reload(file_download)
        log("[HF] Fast transfer disabled (patched module + env).")
    except Exception:
        log("[HF] Fast transfer disabled via env; hub not yet imported.")

# ---------------------------
# [4] Login (optional)
# ---------------------------
def maybe_login(do_login: bool):
    if not do_login: return
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        log("! --login requested but $HUGGINGFACE_TOKEN not set. Skipping.")
        return
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        log("✓ Hugging Face login ok.")
    except Exception as e:
        log(f"! Hugging Face login failed: {e}")

# ---------------------------
# [5] Seed & small utils
# ---------------------------
def set_seed(seed=5541):
    import torch, numpy as np, random as pyrand
    pyrand.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def json_dump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f: json.dump(obj, f, indent=2)

# ---------------------------
# [6] Data load + cache (tokenized)
# ---------------------------
def load_and_tokenize(cfg: Dict[str, Any], save_root: Path, max_train=0, max_val=0):
    disable_hf_transfer()
    from datasets import load_dataset, ClassLabel, DatasetDict, load_from_disk
    from transformers import AutoTokenizer, DataCollatorWithPadding
    from collections import Counter

    cache_dir = save_root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Use on-disk tokenized cache keyed by (dataset, model, maxlen)
    key = re.sub(r"[^a-zA-Z0-9_.-]+", "_", f"{cfg['dataset_id']}__{cfg['model_id']}__L{cfg['max_length']}")
    tok_cache = cache_dir / f"tok_{key}"

    if tok_cache.exists():
        log(f"[DATA] Loading tokenized dataset from disk → {tok_cache}")
        ds_tok = load_from_disk(str(tok_cache))
        # pick metadata back up
        with open(tok_cache / "meta.json") as f:
            meta = json.load(f)
        id2label = {int(k): v for k, v in meta["id2label"].items()}
        collator = DataCollatorWithPadding(tokenizer=AutoTokenizer.from_pretrained(cfg["model_id"]))
        return ds_tok, collator, id2label

    log(f"[DATA] Loading raw dataset: {cfg['dataset_id']}")
    ds = load_dataset(cfg["dataset_id"])

    # ensure validation split
    keys = set(ds.keys())
    if "validation" not in keys and "val" not in keys:
        log("[DATA] No validation split → creating 10% from train")
        split = ds["train"].train_test_split(test_size=0.1, seed=cfg["seed"])
        ds = DatasetDict(train=split["train"], validation=split["test"], **({"test": ds["test"]} if "test" in keys else {}))
    elif "val" in keys:
        ds = DatasetDict(train=ds["train"], validation=ds["val"], **({"test": ds["test"]} if "test" in keys else {}))

    # pick columns
    cand_text = ["text","content","document","body","sentence","prompt","input","inputs","article"]
    cand_label = ["label","labels","target","class","gold","source"]
    cols = list(ds["train"].column_names)
    text_col = next((c for c in cand_text if c in cols), cols[0])
    label_col = next((c for c in cand_label if c in cols), None)
    assert label_col is not None, f"No label-like column found in {cols}"
    feat = ds["train"].features.get(label_col)

    # normalize labels to {0: HUMAN, 1: AI}
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

    def map_label_value(v):
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"0","1"}: return int(s)
            return {"human":0,"real":0,"ai":1,"gpt":1,"machine":1}[s]
        return int(v)

    # subsample (optional)
    def maybe_take(ds_split, nmax):
        if nmax and len(ds_split) > nmax:
            return ds_split.shuffle(seed=cfg["seed"]).select(range(nmax))
        return ds_split

    ds["train"] = maybe_take(ds["train"], max_train)
    ds["validation"] = maybe_take(ds["validation"], max_val)

    # tokenizer + map (with progress)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model_id"], use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    max_len = int(cfg["max_length"])

    def tok_fn(batch):
        enc = tok(batch[text_col], truncation=True, max_length=max_len)
        enc["labels"] = [map_label_value(v) for v in batch[label_col]]
        return enc

    log("[DATA] Tokenizing …")
    ds_tok = {}
    for split in ds.keys():
        log(f"  ↳ {split} ({len(ds[split])} rows)")
        ds_tok[split] = ds[split].map(
            tok_fn, batched=True, remove_columns=ds[split].column_names, desc=f"Tokenize[{split}]"
        )
    from datasets import DatasetDict as HF_DatasetDict
    ds_tok = HF_DatasetDict(**ds_tok)

    # collator & stats
    from transformers import DataCollatorWithPadding
    collator = DataCollatorWithPadding(tokenizer=tok)
    from collections import Counter
    y_counts = Counter(ds_tok["train"]["labels"])
    log(f"[DATA] Label dist (train): " + str({id2label[k]: v for k, v in sorted(y_counts.items())}))

    # persist tokenized dataset
    log(f"[CACHE] Saving tokenized dataset → {tok_cache}")
    ds_tok.save_to_disk(str(tok_cache))
    json_dump({"id2label": id2label, "text_col": text_col, "label_col": label_col, "max_length": max_len}, tok_cache / "meta.json")
    return ds_tok, collator, id2label

# ---------------------------
# [7] Model, (Q)LoRA, training
# ---------------------------
def train_eval(cfg: Dict[str, Any]):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    from transformers import (
        AutoModelForCausalLM, BitsAndBytesConfig, get_linear_schedule_with_warmup
    )
    from peft import LoraConfig, get_peft_model, PeftModel
    import bitsandbytes as bnb
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    device = torch.device("cuda")
    save_root = Path(cfg["save_dir"]); save_root.mkdir(parents=True, exist_ok=True)
    best_dir = save_root / "best"

    # data
    ds_tok, collator, id2label = load_and_tokenize(cfg, save_root, cfg["max_train"], cfg["max_val"])
    train_dl = DataLoader(ds_tok["train"], batch_size=cfg["batch_size"], shuffle=True, collate_fn=collator, pin_memory=True)
    val_bs = max(1, cfg["batch_size"] * 2)
    val_dl = DataLoader(ds_tok["validation"], batch_size=val_bs, shuffle=False, collate_fn=collator, pin_memory=True)

    # early exit if eval-only or resume:
    if cfg["eval_only"] and not best_dir.exists():
        log("! --eval-only requested but no best checkpoint found. Exiting."); return
    if cfg["resume"] and best_dir.exists() and not cfg["eval_only"]:
        log("✓ Best checkpoint exists and --resume set. Skipping training and running eval.")
        return evaluate_and_save(cfg, best_dir, ds_tok, val_dl, collator, id2label)

    # precision
    auto_prec = gpu_probe()
    if cfg["precision"] is None: cfg["precision"] = auto_prec
    torch.set_float32_matmul_precision("medium")

    # bitsandbytes availability check
    bnb_ok = False
    try:
        ver = tuple(int(x) for x in re.findall(r"\d+", bnb.__version__)[:3])
        bnb_ok = ver >= (0, 42, 0)  # safe for 4-bit + PEFT adapter injection
        log(f"[BNB] bitsandbytes {bnb.__version__} | 4-bit LoRA {'enabled' if bnb_ok else 'disabled (fallback)'}")
    except Exception:
        log("[BNB] bitsandbytes version parse failed; disabling 4-bit.")

    # try load in 4-bit, else 8-bit, else full precision
    quant = None
    base_dtype = torch.bfloat16 if cfg["precision"] == "bf16" else torch.float16
    if bnb_ok:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=base_dtype, bnb_4bit_use_double_quant=True)
    elif hasattr(bnb.nn, "Linear8bitLt"):
        quant = BitsAndBytesConfig(load_in_8bit=True)

    log(f"[MODEL] Loading {cfg['model_id']} (quant={'4-bit' if (quant and quant.load_in_4bit) else '8-bit' if (quant and getattr(quant,'load_in_8bit',False)) else 'none'}, dtype={cfg['precision']})")
    t_load = time.time()
    try:
        base = AutoModelForCausalLM.from_pretrained(
            cfg["model_id"],
            quantization_config=quant,
            dtype=base_dtype if quant is None else None,
            device_map={"": 0},
        )
    except Exception as e:
        log(f"! 1st attempt failed ({e}). Retrying without quantization…")
        base = AutoModelForCausalLM.from_pretrained(
            cfg["model_id"], dtype=base_dtype, device_map={"": 0}
        )
    base.config.output_hidden_states = True
    base.config.use_cache = False
    if hasattr(base, "gradient_checkpointing_enable"):
        base.gradient_checkpointing_enable()
    hidden_size = getattr(base.config, "hidden_size", None) or getattr(base.config, "hidden_dim", None)
    assert hidden_size, "Could not infer hidden size from model config."
    log(f"    ↳ loaded in {time.time()-t_load:.1f}s | hidden_size={hidden_size}")

    # LoRA
    backbone = base
    if cfg["use_lora"]:
        lora_cfg = LoraConfig(
            r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
            target_modules=cfg["target_mods"].split(","), bias="none", task_type="CAUSAL_LM",
        )
        log("[LoRA] Injecting adapters …")
        try:
            backbone = get_peft_model(base, lora_cfg)
            backbone.print_trainable_parameters()
        except AttributeError as e:
            # fallback: no quantization injection oddities
            log(f"! PEFT inject hit {type(e).__name__}: {e} → retry with dequantized base.")
            base = AutoModelForCausalLM.from_pretrained(cfg["model_id"], dtype=base_dtype, device_map={"":0})
            base.config.output_hidden_states = True
            base.config.use_cache = False
            if hasattr(base, "gradient_checkpointing_enable"):
                base.gradient_checkpointing_enable()
            backbone = get_peft_model(base, lora_cfg)
            backbone.print_trainable_parameters()

    # light seq-classifier head on pooled last hidden state
    class SeqClassifier(nn.Module):
        def __init__(self, backbone, hidden, num_labels=2, dropout=0.1):
            super().__init__()
            self.backbone = backbone
            self.drop = nn.Dropout(dropout)
            self.cls = nn.Linear(hidden, num_labels)
            self.num_labels = num_labels
        def forward(self, input_ids=None, attention_mask=None, labels=None):
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                                output_hidden_states=True, use_cache=False, return_dict=True)
            h = out.hidden_states[-1]  # [B,T,H]
            if attention_mask is None:
                pooled = h[:, -1]
            else:
                mask = attention_mask.unsqueeze(-1).type_as(h)
                pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            logits = self.cls(self.drop(pooled))
            loss = F.cross_entropy(logits, labels.long()) if labels is not None else None
            return {"loss": loss, "logits": logits}

    model = SeqClassifier(backbone, hidden_size, num_labels=2).to(device)

    # optim + sched
    steps_per_epoch = math.ceil(len(train_dl) / cfg["grad_accum"])
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup_steps = int(total_steps * cfg["warmup_ratio"])
    opt = bnb.optim.PagedAdamW8bit(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], betas=(0.9, 0.95), eps=1e-8
    )
    sch = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg["precision"] == "fp16"))

    # (optional) W&B
    if cfg["wandb"]:
        import wandb
        wandb.init(project="deepfake-detect-gemma3", config=cfg, name="gemma3-1b-pt_qLoRA_seqcls")

    # eval util
    @torch.no_grad()
    def evaluate(model: nn.Module, loader) -> Dict[str, Any]:
        model.eval()
        total_loss, nb = 0.0, 0
        ys, yps, yhats = [], [], []
        pbar = tqdm(loader, desc="Validating", unit="batch", leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None: attention_mask = attention_mask.to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=(torch.bfloat16 if cfg["precision"] == "bf16" else torch.float16)):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out["loss"]
                logits = out["logits"].float()
            prob_ai = torch.softmax(logits, dim=-1)[:, 1]
            pred = logits.argmax(-1)
            total_loss += float(loss.item())
            nb += 1
            ys.append(labels.cpu()); yps.append(prob_ai.cpu()); yhats.append(pred.cpu())
            pbar.set_postfix({"loss": f"{total_loss/max(1,nb):.4f}"})
        import numpy as np
        y = torch.cat(ys).numpy()
        p = torch.cat(yps).numpy()
        yhat = torch.cat(yhats).numpy()
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        acc = accuracy_score(y, yhat)
        f1m = f1_score(y, yhat, average="macro")
        try:
            auc = roc_auc_score(y, p)
        except Exception:
            auc = float("nan")
        return {"loss": total_loss / max(1, nb), "acc": acc, "f1_macro": f1m, "auroc": auc}

    # train
    log(f"[TRAIN] epochs={cfg['epochs']} | steps/epoch≈{steps_per_epoch} | total_steps={total_steps} | warmup={warmup_steps}")
    best_f1 = -1.0
    t0 = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running = 0.0
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

            if cfg["precision"] == "fp16":
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % cfg["grad_accum"] == 0:
                if cfg["precision"] == "fp16":
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                    scaler.step(opt); scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                    opt.step()
                sch.step()
                opt.zero_grad(set_to_none=True)

            running += float(loss.item())
            if step % (cfg["grad_accum"] * 5) == 0:
                avg = running / (cfg["grad_accum"] * 5)
                batches.set_postfix({"train_loss": f"{avg:.4f}", "lr": f"{sch.get_last_lr()[0]:.2e}"})
                if cfg["wandb"]:
                    import wandb; wandb.log({"train/loss": avg, "train/lr": sch.get_last_lr()[0]})
                running = 0.0

        # eval
        valm = evaluate(model, val_dl)
        elapsed = (time.time() - t0) / 60.0
        log(f"[EVAL] epoch={epoch} | val_loss={valm['loss']:.4f} | acc={valm['acc']:.4f} | f1_macro={valm['f1_macro']:.4f} | auroc={valm['auroc']:.4f} | time={elapsed:.1f}m")
        if cfg["wandb"]:
            import wandb; wandb.log({f"val/{k}": v for k, v in valm.items()} | {"time/min": elapsed})

        # save best by F1
        if valm["f1_macro"] > best_f1:
            best_f1 = valm["f1_macro"]
            best_dir.mkdir(parents=True, exist_ok=True)
            # 1) save LoRA
            if isinstance(model.backbone, PeftModel):
                model.backbone.save_pretrained(best_dir / "lora_adapter")
            else:
                backbone.save_pretrained(best_dir / "lora_adapter")
            # 2) save classifier head
            torch.save(
                {"state_dict": model.cls.state_dict(), "hidden_size": hidden_size, "num_labels": model.num_labels},
                best_dir / "classifier.pt",
            )
            # 3) maps + cfg
            json_dump(id2label, best_dir / "id2label.json")
            json_dump(cfg, best_dir / "cfg.json")
            log(f"✓ Saved new best → {best_dir} (f1_macro={best_f1:.4f})")

        torch.cuda.empty_cache(); gc.collect()

    log("🎉 Training complete.")
    evaluate_and_save(cfg, best_dir, ds_tok, val_dl, collator, id2label)

# ---------------------------
# [8] Reload best, evaluate, and dump predictions
# ---------------------------
def evaluate_and_save(cfg, best_dir: Path, ds_tok, val_dl, collator, id2label):
    import torch, json
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
    from peft import PeftModel
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm
    import numpy as np

    assert best_dir.exists(), f"Best dir not found: {best_dir}"
    device = torch.device("cuda")

    # dtype
    import torch as _torch
    base_dtype = _torch.bfloat16 if cfg["precision"] == "bf16" else _torch.float16

    # light quant for reload (ok to skip)
    try:
        from bitsandbytes import __version__ as _v
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=base_dtype, bnb_4bit_use_double_quant=True)
    except Exception:
        quant = None

    # model reload
    base = AutoModelForCausalLM.from_pretrained(cfg["model_id"], quantization_config=quant, dtype=None if quant else base_dtype, device_map={"":0})
    base.config.output_hidden_states = True; base.config.use_cache = False
    peft = PeftModel.from_pretrained(base, best_dir / "lora_adapter")
    # reconstruct classifier
    head_state = torch.load(best_dir / "classifier.pt", map_location="cpu")
    hidden_size = head_state.get("hidden_size")
    num_labels = head_state.get("num_labels", 2)

    import torch.nn as nn, torch.nn.functional as F
    class SeqClassifier(nn.Module):
        def __init__(self, backbone, hidden, num_labels=2):
            super().__init__(); self.backbone = backbone; self.cls = nn.Linear(hidden, num_labels)
        def forward(self, input_ids=None, attention_mask=None):
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False, return_dict=True)
            h = out.hidden_states[-1]; mask = attention_mask.unsqueeze(-1).type_as(h) if attention_mask is not None else None
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0) if mask is not None else h[:, -1]
            return self.cls(pooled)

    model = SeqClassifier(peft, hidden_size, num_labels).to(device)
    model.cls.load_state_dict(head_state["state_dict"])
    model.eval()

    # make/test loader (use same val_dl)
    if isinstance(val_dl, DataLoader):
        loader = val_dl
    else:
        loader = DataLoader(ds_tok["validation"], batch_size=max(1, cfg["batch_size"]*2), shuffle=False, collate_fn=collator)

    # eval metrics + write preds
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    ys, yps, yhats = [], [], []
    pbar = tqdm(loader, desc="Eval(best)", unit="batch", leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None: attention_mask = attention_mask.to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=base_dtype):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).float()
        prob_ai = torch.softmax(logits, dim=-1)[:, 1]
        pred = logits.argmax(-1)
        ys.append(labels.cpu()); yps.append(prob_ai.cpu()); yhats.append(pred.cpu())

    y = torch.cat(ys).numpy(); p = torch.cat(yps).numpy(); yhat = torch.cat(yhats).numpy()
    acc = accuracy_score(y, yhat); f1m = f1_score(y, yhat, average="macro")
    try: auc = roc_auc_score(y, p)
    except Exception: auc = float("nan")
    log(f"[BEST] acc={acc:.4f} | f1_macro={f1m:.4f} | auroc={auc:.4f}")

    # dump predictions
    pred_dir = best_dir / "preds"; pred_dir.mkdir(parents=True, exist_ok=True)
    np.save(pred_dir / "val_labels.npy", y)
    np.save(pred_dir / "val_prob_ai.npy", p)
    np.save(pred_dir / "val_pred.npy", yhat)
    json_dump({"acc": acc, "f1_macro": f1m, "auroc": auc}, pred_dir / "metrics.json")
    log(f"✓ Saved preds + metrics → {pred_dir}")

# ---------------------------
# [9] main
# ---------------------------
def main():
    args = build_argparser().parse_args()
    prec = gpu_probe()
    precision = args.precision or prec

    cfg = {
        "project": "deepfake-detect-gemma3",
        "model_id": args.model_id,
        "dataset_id": args.dataset_id,
        "save_dir": args.save_dir,
        "seed": args.seed,
        "precision": precision,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "use_lora": args.use_lora,
        "target_mods": args.target_mods,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "max_train": args.max_train,
        "max_val": args.max_val,
        "wandb": args.wandb,
        "eval_only": args.eval_only,
        "resume": args.resume,
    }

    # print config
    log("[CFG]\n" + json.dumps(cfg, indent=2))

    disable_hf_transfer()
    maybe_login(args.login)
    set_seed(cfg["seed"])

    # versions snapshot
    try:
        import transformers, datasets, peft, accelerate, huggingface_hub, bitsandbytes
        log(f"[VERS] transformers {transformers.__version__} | datasets {datasets.__version__} | peft {peft.__version__} | accelerate {accelerate.__version__} | hub {huggingface_hub.__version__} | bnb {bitsandbytes.__version__}")
    except Exception as e:
        log(f"[VERS] Could not snapshot versions: {e}")

    if args.eval_only or args.resume:
        # Will run eval only if checkpoint is present; otherwise continue to train.
        try:
            train_eval(cfg)
        except AssertionError as e:
            log(f"Eval path hit: {e}")
    else:
        train_eval(cfg)

if __name__ == "__main__":
    main()
