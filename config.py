# === A40: bootstrap & sanity ===
import os, sys, json, random
from pathlib import Path

# [1/8] PATH & user-site so %pip --user imports work in this kernel
HOME = os.path.expanduser("~")
os.environ["PATH"] = f"{HOME}/.local/bin:" + os.environ.get("PATH","")
import site
USER_SITE = site.getusersitepackages()
if USER_SITE not in sys.path: sys.path.insert(0, USER_SITE)
print("[1/8] PATH ok →", os.environ["PATH"].split(":")[0])
print("[1/8] USER_SITE →", USER_SITE)

# [2/8] Install deps (quiet)
print("[2/8] Installing deps …")
%pip -q install --user --no-warn-script-location -U \
  transformers>=4.44 datasets>=2.20 peft>=0.11 accelerate>=0.33 \
  bitsandbytes evaluate scikit-learn wandb huggingface_hub

# [3/8] Place caches on fast node-local storage (no deprecated TRANSFORMERS_CACHE)
TMPDIR = os.environ.get("TMPDIR") or f"/tmp/{os.environ.get('USER','user')}"
for d in ("hf_home","hf_hub","hf_datasets"): Path(f"{TMPDIR}/{d}").mkdir(parents=True, exist_ok=True)
os.environ.update({
    "HF_HOME": f"{TMPDIR}/hf_home",
    "HF_HUB_CACHE": f"{TMPDIR}/hf_hub",          # modern hub cache
    "HF_DATASETS_CACHE": f"{TMPDIR}/hf_datasets",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
})
print(f"[3/8] Caches → HF_HOME={os.environ['HF_HOME']} | HF_HUB_CACHE={os.environ['HF_HUB_CACHE']}")

# [4/8] GPU probe + driver snapshot
import torch
assert torch.cuda.is_available(), "CUDA GPU not visible."
name = torch.cuda.get_device_name(0)
cc   = ".".join(map(str, torch.cuda.get_device_capability(0)))
vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
use_bf16 = torch.cuda.is_bf16_supported()
dtype = "bf16" if use_bf16 else "fp16"
print(f"[4/8] GPU={name} | CC {cc} | VRAM~{vram:.1f} GB | precision={dtype}")
!nvidia-smi

# [5/8] Hugging Face login (needed to accept Gemma license & pull weights)
from huggingface_hub import notebook_login
print("[5/8] Opening Hugging Face login… (accept google/gemma-3-1b-it license in browser once)")
notebook_login()

# [6/8] Repro + minimal run config
def set_seed(seed=5541):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
set_seed(5541)

CFG = {
    "project": "deepfake-detect-gemma3",
    "model_id": "google/gemma-3-1b-it",
    "dataset_id": "yaful/MAGE",                # Deepfake/AI-text detection
    "task": "sequence_classification",
    "num_labels": 2,
    "max_length": 1024,
    "precision": dtype,
    "use_lora": True,
    "lora": {"r": 16, "alpha": 32, "dropout": 0.1, "target_modules": ["q_proj","v_proj"]},
    "train": {"epochs": 3, "per_device_batch_size": 4, "grad_accum_steps": 8,
              "lr": 1e-4, "weight_decay": 0.01, "warmup_ratio": 0.06, "max_grad_norm": 1.0},
    "save_dir": "./outputs/gemma3-1b-it-mage",
}
print("[6/8] Config:\n" + json.dumps(CFG, indent=2))

# [7/8] Verify imports & print versions (now that user-site is on sys.path)
import transformers, datasets, peft, accelerate, evaluate, sklearn, huggingface_hub
print(f"[7/8] Versions → transformers {transformers.__version__} | datasets {datasets.__version__} | "
      f"peft {peft.__version__} | accelerate {accelerate.__version__} | hub {huggingface_hub.__version__}")

# [8/8] Final sanity
print("✅ Setup complete. Next: load tokenizer/model, preprocess MAGE, and start CustomTrainer.")
