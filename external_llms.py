import os, re, time, json, csv, random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
import numpy as np
from datasets import load_from_disk, load_dataset
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score

HYPERBOLIC_API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJzYXVlcnNsYWJzQGdtYWlsLmNvbSIsImlhdCI6MTc1ODg1MTM4OH0.UIIxFK6JiR01VWm3u5WPu3toSdXYnqrWYKlqHHoxKk4"

BASE_URL_CHAT = "https://api.hyperbolic.xyz/v1/chat/completions"
BASE_URL_COMP = "https://api.hyperbolic.xyz/v1/completions"

SAVE_DIR    = "./runs"
DATASET_ID  = "yaful/MAGE"
TRAIN_MODEL = "google/gemma-3-1b-pt"   # tokenizer fallback to decode cached ids
MAX_LENGTH  = 256

N_EVAL      = 5     # trials/examples
SEED        = 42
MAX_TEXT_CHARS = 6000

MODELS = [
    {"name": "moonshotai/Kimi-K2-Instruct",      "endpoint": "chat"},
    {"name": "meta-llama/Meta-Llama-3.1-405B",   "endpoint": "comp"},
]

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
# DATA LOADING (prefer cached split)
# -------------------------------
def _tok_cache_path(save_dir: str, dataset_id: str, model_id: str, max_length: int) -> Path:
    key = re.sub(r"[^a-zA-Z0-9_.-]+", "_", f"{dataset_id}__{model_id}__L{max_length}")
    return Path(save_dir) / "cache" / f"tok_{key}"

CAND_TEXT  = ["text","content","document","body","sentence","prompt","input","inputs","article"]
CAND_LABEL = ["label","labels","target","class","gold","source"]

def _auto_columns(ds) -> Tuple[str, str]:
    cols = ds.column_names
    text_col = next((c for c in CAND_TEXT if c in cols), cols[0])
    label_col = next((c for c in CAND_LABEL if c in cols), None)
    if label_col is None:
        for c in cols:
            sample = list(map(str, ds[c][: min(1000, len(ds)) ]))
            uniq = set(s.strip().lower() for s in sample)
            if len(uniq) <= 6 and uniq <= {"0","1","human","ai","gpt","real","machine"}:
                label_col = c; break
    if label_col is None:
        raise RuntimeError(f"Could not infer label column from {cols}")
    return text_col, label_col

def _label_to_int(v: Any) -> int:
    s = str(v).strip().lower()
    if s in {"0","human","real"}: return 0
    if s in {"1","ai","gpt","machine"}: return 1
    raise ValueError(f"Unrecognized label value: {v}")

def load_validation_texts(n_eval: int, seed: int) -> Tuple[List[str], List[int], Dict[int,str]]:
    rng = random.Random(seed)

    # Try cached tokenized split with raw_text first
    cache_dir = _tok_cache_path(SAVE_DIR, DATASET_ID, TRAIN_MODEL, MAX_LENGTH)
    if cache_dir.exists():
        ds_tok = load_from_disk(str(cache_dir))
        if "validation" not in ds_tok:
            raise RuntimeError("Cached tokenized dataset found, but missing 'validation' split.")
        val = ds_tok["validation"]
        if "raw_text" not in val.column_names:
            tokenizer = AutoTokenizer.from_pretrained(TRAIN_MODEL, use_fast=True, trust_remote_code=False)
            def _decode(batch):
                texts = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
                return {"raw_text": texts}
            val = val.map(_decode, batched=True)
        texts = list(val["raw_text"])
        labels = [int(x) for x in val["labels"]]
    else:
        ds = load_dataset(DATASET_ID)
        split = "validation" if "validation" in ds else ("val" if "val" in ds else "test")
        if split:
            val = ds[split]
        else:
            val = ds["train"].train_test_split(test_size=0.1, seed=seed)["test"]
        text_col, label_col = _auto_columns(val)
        texts  = list(val[text_col])
        labels = [_label_to_int(v) for v in val[label_col]]

    # Build a small stratified sample of exactly n_eval where possible
    idx_pos = [i for i, y in enumerate(labels) if y == 1]
    idx_neg = [i for i, y in enumerate(labels) if y == 0]
    rng.shuffle(idx_pos); rng.shuffle(idx_neg)
    half = max(1, n_eval // 2)
    if idx_pos and idx_neg:
        chosen = (idx_pos[:half] + idx_neg[:(n_eval - half)])
        rng.shuffle(chosen)
    else:
        chosen = list(range(min(n_eval, len(labels))))
    chosen = chosen[:n_eval]

    # Trim very long texts for safety
    texts_out  = [ (str(texts[i]) if len(str(texts[i])) <= MAX_TEXT_CHARS else str(texts[i])[:MAX_TEXT_CHARS] + " …")
                   for i in chosen ]
    labels_out = [ labels[i] for i in chosen ]
    id2label = {0:"HUMAN", 1:"AI"}
    return texts_out, labels_out, id2label

# -------------------------------
# API HELPERS
# -------------------------------
def _backoff(attempt: int) -> float:
    return min(16.0, 0.5 * (2 ** (attempt - 1)))

def _post(url: str, payload: Dict[str, Any], timeout: int = 60, tries: int = 6) -> Dict[str, Any]:
    for k in range(1, tries + 1):
        try:
            r = requests.post(url, headers=HEADERS, json=payload, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429,500,502,503,504):
                delay = _backoff(k)
                print(f"[API] {r.status_code} → retrying in {delay:.1f}s")
                time.sleep(delay); continue
            print(f"[API] {r.status_code}: {r.text[:400]}")
            break
        except requests.RequestException as e:
            delay = _backoff(k)
            print(f"[API] Exception {e} → retrying in {delay:.1f}s")
            time.sleep(delay)
    return {"error": True, "note": "request_failed"}

def _prompt_for(text: str) -> str:
    return STRICT_PROMPT.format(TEXT=text)

def call_kimi(text: str) -> Tuple[str, Dict[str, Any]]:
    payload = {
        "messages": [{"role": "user", "content": _prompt_for(text)}],
        "model": "moonshotai/Kimi-K2-Instruct",
        "max_tokens": 128,     # ↑ allow full <scratchpad> + <answer>
        "temperature": 0.0,
        "top_p": 1.0,
        "stop": ["</answer>"], # hint to cut right after tag (if supported)
        "stream": False
    }
    data = _post(BASE_URL_CHAT, payload)
    try:
        out = data["choices"][0]["message"]["content"]
    except Exception:
        out = ""
    return out, data

def call_llama(text: str) -> Tuple[str, Dict[str, Any]]:
    payload = {
        "prompt": _prompt_for(text),
        "model": "meta-llama/Meta-Llama-3.1-405B",
        "max_tokens": 64,      # enough to emit scratchpad + <answer>
        "temperature": 0.0,
        "top_p": 1.0,
        "stop": ["</answer>"], # hint to stop right after the final tag
        "stream": False
    }
    data = _post(BASE_URL_COMP, payload)
    try:
        out = data["choices"][0]["text"]
    except Exception:
        out = ""
    return out, data

# -------------------------------
# PERMISSIVE PARSER (NO INVALIDS)
# -------------------------------
ANSWER_TAG_RE = re.compile(r"<\s*answer\s*>\s*(human|ai)\s*<\s*/\s*answer\s*>", re.I)

def parse_label(output: str) -> str:
    """
    Very permissive:
      1) Prefer <answer>HUMAN|AI</answer> (case-insensitive, allow spaces).
      2) Else: find last occurrence of 'human' or 'ai' anywhere; choose whichever appears last,
         biased toward the region after the word 'answer' if present.
      3) If nothing found, return 'AI' (ensures no invalids).
    """
    if not output:
        return "AI"

    low = output.lower()

    # 1) explicit tag
    m = ANSWER_TAG_RE.search(low)
    if m:
        return m.group(1).upper()

    # 2) heuristic around 'answer'
    anchor = low.rfind("answer")
    cand = []
    for mm in re.finditer(r"\bhuman\b", low):
        cand.append((mm.start(), "HUMAN"))
    for mm in re.finditer(r"\bai\b", low):
        cand.append((mm.start(), "AI"))

    if cand:
        if anchor != -1:
            after = [c for c in cand if c[0] >= anchor]
            if after:
                after.sort(key=lambda x: x[0])
                return after[0][1]
        cand.sort(key=lambda x: x[0])
        return cand[-1][1]

    return "AI"

def eval_model(spec: Dict[str,str], texts: List[str], gold: List[int], id2label: Dict[int,str], out_dir: Path):
    name = spec["name"]; endpoint = spec["endpoint"]
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)

    preds_str: List[str] = []
    labels_pred: List[int] = []

    print(f"\n=== Evaluating {name} on {len(texts)} examples ===")
    for i, t in enumerate(texts, start=1):
        if endpoint == "chat":
            raw, raw_json = call_kimi(t)
        else:
            raw, raw_json = call_llama(t)

        # Print RAW text output
        print(f"\n[{name}] Example {i}/{len(texts)} RAW OUTPUT:\n{raw}\n", flush=True)

        # If empty, also dump the raw JSON to help debug
        if (not raw) or (raw.strip() == ""):
            try:
                pretty = json.dumps(raw_json, indent=2, ensure_ascii=False)
            except Exception:
                pretty = str(raw_json)
            print(f"[{name}] Example {i} RAW JSON (empty/blank text):\n{pretty}\n", flush=True)

        lab = parse_label(raw)            # 'HUMAN' | 'AI'
        yhat = 0 if lab == "HUMAN" else 1
        preds_str.append(raw)
        labels_pred.append(yhat)

    # Binary scores for AUROC (no logprobs): AI→1, HUMAN→0
    y_true = gold
    y_score = labels_pred
    y_pred  = labels_pred

    # Metrics
    acc  = accuracy_score(y_true, y_pred)
    f1m  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        uniq = set(y_true)
        auroc = roc_auc_score(y_true, y_score) if len(uniq) == 2 else float("nan")
    except Exception:
        auroc = float("nan")
    cm   = confusion_matrix(y_true, y_pred, labels=[0,1]).tolist()
    report = classification_report(
        y_true, y_pred, labels=[0,1],
        target_names=[id2label[0], id2label[1]],
        digits=4, zero_division=0
    )

    # Save artifacts
    out_dir.mkdir(parents=True, exist_ok=True)
    mpath = out_dir / f"external_llm_metrics__{safe}.json"
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": name,
            "n": len(texts),
            "accuracy": float(acc),
            "f1_macro": float(f1m),
            "auroc": (None if (auroc != auroc) else float(auroc)),
            "confusion_matrix": cm,
            "classification_report": report
        }, f, indent=2, ensure_ascii=False)

    ppath = out_dir / f"external_llm_preds__{safe}.csv"
    with open(ppath, "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(["id","gold_int","gold_name","pred_int","pred_name","raw_output","text"])
        for i, (g, yp, raw, tx) in enumerate(zip(gold, y_pred, preds_str, texts)):
            w.writerow([i, g, id2label[g], yp, id2label[yp], raw, tx])

    # Console summary
    print(f"\n--- {name} SUMMARY ---")
    print(f"  n={len(texts)} | acc={acc:.4f} | f1_macro={f1m:.4f} | auroc={(auroc if auroc==auroc else float('nan')):.4f}")
    print("  confusion_matrix [[TN, FP],[FN, TP]]:")
    print(f"  {cm}")
    print("\n" + report)
    print(f"  Saved metrics → {mpath}")
    print(f"  Saved preds   → {ppath}")

def main():
    random.seed(SEED); np.random.seed(SEED)
    texts, labels, id2label = load_validation_texts(N_EVAL, SEED)
    print(f"[DATA] Loaded {len(texts)} examples. Class names: {id2label}")

    # Avoid empty strings
    texts = ["EMPTY" if (t is None or str(t).strip()=="") else str(t) for t in texts]

    out_dir = Path(SAVE_DIR) / "api_baselines"
    for spec in MODELS:
        eval_model(spec, texts, labels, id2label, out_dir)

if __name__ == "__main__":
    main()
