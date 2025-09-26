import os
import re
import json
import csv
import time
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score
)

HYPERBOLIC_API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJzYXVlcnNsYWJzQGdtYWlsLmNvbSIsImlhdCI6MTc1ODg1MTM4OH0.UIIxFK6JiR01VWm3u5WPu3toSdXYnqrWYKlqHHoxKk4"

BASE_URL_CHAT = "https://api.hyperbolic.xyz/v1/chat/completions"
BASE_URL_COMP = "https://api.hyperbolic.xyz/v1/completions"

SAVE_DIR       = "./runs"
DATASET_ID     = "yaful/MAGE"
TRAIN_MODEL_ID = "google/gemma-3-1b-pt"   # tokenizer fallback if needed
MAX_LENGTH     = 256

N_EVAL         = 150     # trials/examples
SEED           = 42
MAX_TEXT_CHARS = 6000

# Concurrency + timeouts
MAX_CONCURRENT = 5  # up to 5 in-flight calls
HTTP_TIMEOUT   = 120  # seconds

ENABLE_KIMI = True
ENABLE_L405B_BASE = False

# Model specs (copy user's temp/top_p; raise max_tokens to 12k)
MODELS: List[Dict[str, Any]] = [
    {
        "name": "moonshotai/Kimi-K2-Instruct",
        "endpoint": "chat",
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": 12000
    },
    {
        "name": "meta-llama/Meta-Llama-3.1-405B",
        "endpoint": "comp",
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 12000
    },
]

# Apply toggles
if not ENABLE_KIMI:
    MODELS = [m for m in MODELS if m["name"] != "moonshotai/Kimi-K2-Instruct"]
if not ENABLE_L405B_BASE:
    MODELS = [m for m in MODELS if m["name"] != "meta-llama/Meta-Llama-3.1-405B"]

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {HYPERBOLIC_API_KEY}",
}

STRICT_PROMPT = (
    "You are a binary detector of AI-generated text.\n"
    "First, briefly reason in <scratchpad>…</scratchpad>.\n"
    "Then output exactly one final tag on its own line:\n"
    "<answer>HUMAN</answer> or <answer>AI</answer>.\n\n"
    "Text:\n<<<\n{TEXT}\n>>>\n"
    "Output nothing after the </answer> tag."
)

# -------------------------------
# DATA LOADING
# -------------------------------
CAND_TEXT  = ["text","content","document","body","sentence","prompt","input","inputs","article"]
CAND_LABEL = ["label","labels","target","class","gold","source"]

def _tok_cache_path(save_dir: str, dataset_id: str, model_id: str, max_length: int) -> Path:
    key = re.sub(r"[^a-zA-Z0-9_.-]+", "_", f"{dataset_id}__{model_id}__L{max_length}")
    return Path(save_dir) / "cache" / f"tok_{key}"

def _auto_columns(ds) -> Tuple[str, str]:
    cols = ds.column_names
    text_col = next((c for c in CAND_TEXT if c in cols), cols[0])
    label_col = next((c for c in CAND_LABEL if c in cols), None)
    if label_col is None:
        # heuristic: tiny set of label-like tokens in sample
        sample_n = min(1000, len(ds))
        for c in cols:
            sample = list(map(str, ds[c][: sample_n ]))
            uniq = set(s.strip().lower() for s in sample)
            if len(uniq) <= 6 and uniq <= {"0","1","human","ai","gpt","real","machine"}:
                label_col = c
                break
    if label_col is None:
        raise RuntimeError(f"Could not infer label column from {cols}")
    return text_col, label_col

def _label_to_int(v: Any) -> int:
    s = str(v).strip().lower()
    if s in {"0","human","real"}: return 0
    if s in {"1","ai","gpt","machine"}: return 1
    raise ValueError(f"Unrecognized label value: {v}")

def load_validation_texts(n_eval: int, seed: int) -> Tuple[List[str], List[int], Dict[int, str]]:
    # Avoid dill/pickle of ColabKernelApp by disabling caching & keeping everything in RAM
    from datasets import load_dataset, disable_caching
    disable_caching()
    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets")  # harmless; not used when caching is disabled

    rng = random.Random(seed)

    # Non-streaming, in-memory load
    ds_all = load_dataset(DATASET_ID, keep_in_memory=True)

    # Prefer validation → test → train
    for cand in ("validation", "test", "train"):
        if cand in ds_all:
            ds = ds_all[cand]
            break
    else:
        raise RuntimeError(f"No usable split found in {DATASET_ID}")

    # Columns
    cols = ds.column_names
    text_col  = next((c for c in CAND_TEXT  if c in cols), cols[0])
    label_col = "label" if "label" in cols else next((c for c in CAND_LABEL if c in cols), None)
    if label_col is None:
        raise RuntimeError(f"Could not infer label column from {cols}")

    # Normalize labels to {0:HUMAN, 1:AI}; MAGE is 0=AI,1=HUMAN → invert
    def norm_label(v: Any) -> int:
        s = str(v).strip().lower()
        if DATASET_ID.lower() == "yaful/mage":
            iv = int(v) if s in {"0","1"} else (0 if s in {"ai","gpt","machine"} else 1)
            return 0 if iv == 1 else 1
        return 0 if s in {"0","human","real"} else 1

    # Balanced sample
    idxs = list(range(len(ds))); rng.shuffle(idxs)
    target_pos = max(1, n_eval // 2); target_neg = n_eval - target_pos
    pos, neg = [], []
    for i in idxs:
        t = str(ds[text_col][i])
        if len(t) > MAX_TEXT_CHARS: t = t[:MAX_TEXT_CHARS] + " …"
        y = norm_label(ds[label_col][i])
        (pos if y == 1 else neg).append((t, y))
        if len(pos) >= target_pos and len(neg) >= target_neg:
            break

    pooled = (pos[:target_pos] + neg[:target_neg]) or (pos + neg)
    rng.shuffle(pooled); pooled = pooled[:n_eval]

    texts  = [t for t, _ in pooled]
    labels = [y for _, y in pooled]
    id2label = {0: "HUMAN", 1: "AI"}
    return texts, labels, id2label

# -------------------------------
# API HELPERS (NO RETRIES)
# -------------------------------
def _prompt_for(text: str) -> str:
    return STRICT_PROMPT.format(TEXT=text)

def _post_once(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        r = requests.post(url, headers=HEADERS, json=payload, timeout=HTTP_TIMEOUT)
        if r.status_code == 200:
            return r.json()
        return {"error": True, "status": r.status_code, "text": r.text}
    except requests.RequestException as e:
        return {"error": True, "exception": str(e)}

def call_kimi(text: str, temperature: float, top_p: float, max_tokens: int) -> Tuple[str, Dict[str, Any]]:
    payload = {
        "messages": [{"role": "user", "content": _prompt_for(text)}],
        "model": "moonshotai/Kimi-K2-Instruct",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stop": ["</answer>"],
        "stream": False
    }
    data = _post_once(BASE_URL_CHAT, payload)
    try:
        out = data["choices"][0]["message"]["content"]
    except Exception:
        out = ""
    return out, data

def call_llama_base(text: str, temperature: float, top_p: float, max_tokens: int) -> Tuple[str, Dict[str, Any]]:
    payload = {
        "prompt": _prompt_for(text),
        "model": "meta-llama/Meta-Llama-3.1-405B",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stop": ["</answer>"],
        "stream": False
    }
    data = _post_once(BASE_URL_COMP, payload)
    try:
        out = data["choices"][0]["text"]
    except Exception:
        out = ""
    return out, data

# -------------------------------
# PERMISSIVE PARSER (NO FALLBACK)
# -------------------------------
ANSWER_TAG_RE = re.compile(r"<\s*answer\s*>\s*(human|ai)\s*<\s*/\s*answer\s*>", re.I)

def parse_label_or_none(output: str) -> Optional[str]:
    """
    Very permissive, but NO fallback default:
      1) Try <answer>HUMAN|AI</answer> (case-insensitive, allow spaces).
      2) Else: look anywhere for HUMAN or AI tokens; pick the LAST occurrence (if any).
      3) If nothing found → return None (the example will be skipped in metrics).
    """
    if not output:
        return None

    low = output.lower()

    # 1) explicit tag
    m = ANSWER_TAG_RE.search(low)
    if m:
        return m.group(1).upper()

    # 2) look for tokens anywhere; pick last occurrence
    positions: List[Tuple[int, str]] = []
    positions += [(m.start(), "HUMAN") for m in re.finditer(r"\bhuman\b", low)]
    positions += [(m.start(), "AI")    for m in re.finditer(r"\bai\b", low)]
    if positions:
        positions.sort(key=lambda x: x[0])
        return positions[-1][1]

    # 3) none found
    return None

# -------------------------------
# EVALUATION (BATCHED CONCURRENCY)
# -------------------------------
def _call_model(spec: Dict[str, Any], text: str) -> Tuple[str, Dict[str, Any]]:
    if spec["endpoint"] == "chat":
        return call_kimi(text, spec["temperature"], spec["top_p"], spec["max_tokens"])
    else:
        return call_llama_base(text, spec["temperature"], spec["top_p"], spec["max_tokens"])

def eval_model(
    spec: Dict[str, Any],
    texts: List[str],
    gold: List[int],
    id2label: Dict[int, str],
    out_dir: Path
):
    name = spec["name"]
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)

    print(f"\n=== Evaluating {name} on {len(texts)} examples ===", flush=True)

    # Run in batches with up to MAX_CONCURRENT calls at a time
    n = len(texts)
    raw_outputs: List[Optional[str]] = [None] * n
    raw_jsons:   List[Optional[Dict[str, Any]]] = [None] * n

    for start in range(0, n, MAX_CONCURRENT):
        end = min(start + MAX_CONCURRENT, n)
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as ex:
            futures = {
                ex.submit(_call_model, spec, texts[i]): i
                for i in range(start, end)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    raw, data = fut.result()
                except Exception as e:
                    raw, data = "", {"error": True, "exception": f"executor:{type(e).__name__}:{e}"}
                raw_outputs[i] = raw
                raw_jsons[i]   = data

    # Print RAW output for every call in order
    for i in range(n):
        raw = raw_outputs[i] or ""
        print(f"\n[{name}] Example {i+1}/{n} RAW OUTPUT:\n{raw}\n", flush=True)
        if raw.strip() == "":
            try:
                pretty = json.dumps(raw_jsons[i], indent=2, ensure_ascii=False)
            except Exception:
                pretty = str(raw_jsons[i])
            print(f"[{name}] Example {i+1} RAW JSON (empty/blank text):\n{pretty}\n", flush=True)

    # Parse labels (None if undecidable)
    parsed_labels: List[Optional[str]] = [parse_label_or_none(o or "") for o in raw_outputs]
    y_pred_opt: List[Optional[int]] = [ (0 if lab == "HUMAN" else 1) if lab in ("HUMAN","AI") else None
                                        for lab in parsed_labels ]

    # Filter out undecidable (None) for metrics
    keep_idx = [i for i, yp in enumerate(y_pred_opt) if yp is not None]
    skipped  = [i for i in range(n) if i not in keep_idx]

    if len(keep_idx) == 0:
        acc = float("nan"); f1m = float("nan"); auroc = float("nan")
        cm = [[0,0],[0,0]]
        report = "No valid predictions (all undecidable)."
        used_n = 0
    else:
        y_true = [gold[i] for i in keep_idx]
        y_pred = [y_pred_opt[i] for i in keep_idx]
        y_score = y_pred  # binary 0/1 for AUROC (no logprobs)

        used_n = len(y_true)
        acc  = accuracy_score(y_true, y_pred)
        f1m  = f1_score(y_true, y_pred, average="macro", zero_division=0)
        try:
            if len(set(y_true)) == 2:
                auroc = roc_auc_score(y_true, y_score)
            else:
                auroc = float("nan")
        except Exception:
            auroc = float("nan")
        cm   = confusion_matrix(y_true, y_pred, labels=[0,1]).tolist()
        try:
            report = classification_report(
                y_true, y_pred, labels=[0,1],
                target_names=[id2label[0], id2label[1]],
                digits=4, zero_division=0
            )
        except Exception:
            report = "classification_report failed."

    # Save artifacts
    out_dir.mkdir(parents=True, exist_ok=True)
    mpath = out_dir / f"external_llm_metrics__{safe}.json"
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": name,
            "n_requested": n,
            "n_used_for_metrics": used_n,
            "n_skipped": len(skipped),
            "skipped_indices": skipped,
            "accuracy": (None if acc != acc else float(acc)),
            "f1_macro": (None if f1m != f1m else float(f1m)),
            "auroc": (None if auroc != auroc else float(auroc)),
            "confusion_matrix": cm,
            "classification_report": report
        }, f, indent=2, ensure_ascii=False)

    ppath = out_dir / f"external_llm_preds__{safe}.csv"
    with open(ppath, "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(["id","gold_int","gold_name","pred_int","pred_name","raw_output","text"])
        for i in range(n):
            yp = y_pred_opt[i]
            pred_name = (id2label[yp] if yp is not None else "")
            w.writerow([
                i,
                gold[i],
                id2label[gold[i]],
                ("" if yp is None else yp),
                pred_name,
                (raw_outputs[i] or ""),
                texts[i]
            ])

    # Console summary
    print(f"\n--- {name} SUMMARY ---")
    print(f"  requested={n} | used={used_n} | skipped={len(skipped)} -> {skipped}")
    print(f"  acc={(acc if acc==acc else float('nan')):.4f} | f1_macro={(f1m if f1m==f1m else float('nan')):.4f} | auroc={(auroc if auroc==auroc else float('nan')):.4f}")
    print("  confusion_matrix [[TN, FP],[FN, TP]]:")
    print(f"  {cm}\n")
    print(report)
    print(f"  Saved metrics → {mpath}")
    print(f"  Saved preds   → {ppath}")

# -------------------------------
# MAIN
# -------------------------------
def main():
    random.seed(SEED); np.random.seed(SEED)
    texts, labels, id2label = load_validation_texts(N_EVAL, SEED)
    print(f"[DATA] Loaded {len(texts)} examples. Class names: {id2label}")

    # Avoid blanks
    texts = ["EMPTY" if (t is None or str(t).strip()=="") else str(t) for t in texts]

    out_dir = Path(SAVE_DIR) / "api_baselines"
    for spec in MODELS:
        eval_model(spec, texts, labels, id2label, out_dir)

if __name__ == "__main__":
    main()
