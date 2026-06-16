"""Central configuration for the LCF reproduction.

All paths default to a SCRATCH directory OUTSIDE the Obsidian vault so that
multi-GB model caches / hidden-state tensors never get synced by iCloud/git.
Override with the LCF_SCRATCH env var on the H100 box.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path

# --- Where heavy artifacts live (NOT inside the vault) -----------------------
REPO_DIR = Path(__file__).resolve().parent
SCRATCH = Path(os.environ.get("LCF_SCRATCH", REPO_DIR / "_scratch"))
SCRATCH.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    # --- Base LLM ------------------------------------------------------------
    model_name: str = os.environ.get("LCF_MODEL", "meta-llama/Llama-2-7b-chat-hf")
    d_model: int = 4096          # Llama2-7b / Llama3-8b / Mistral-7b hidden size
    n_layers: int = 32
    load_in_4bit: bool = os.environ.get("LCF_4BIT", "1") != "0"  # set LCF_4BIT=0 for fp16

    # --- LCF architecture (from paper + supplementary) -----------------------
    # Projectors: two-layer MLP with projection dims [2048, 1024]
    proj_hidden: int = 2048
    proj_out: int = 1024         # dimensionality of content / logic space
    # Decoder MLP: dims [1024 -> 2048 -> d_model]
    dec_hidden: int = 2048
    contrastive_tau: float = 0.2  # temperature (paper uses tau, value unspecified)

    # --- Hidden-state extraction --------------------------------------------
    layer_lo: int = 10           # extract/modify within [layer_lo, layer_hi)
    layer_hi: int = 30
    layers_per_pair: int = 2     # supplementary: 2 random layers per identical-token pair
    tap_points: tuple = ("attn", "mlp")  # tap attention output and MLP output

    # --- Inference modification ---------------------------------------------
    n_modify_layers: int = 10    # modify the 10 most "distinctive" taps
    eta_generation: float = 0.5  # modification magnitude (conclusion generation)
    eta_identification: float = 4.5  # (fallacy identification)

    # --- Training ------------------------------------------------------------
    lr: float = 1e-4         # 1e-3 (paper) diverges with cross-attn decoder; 1e-4 + grad-clip is stable
    epochs: int = 15
    batch_size: int = 256
    weight_decay: float = 0.0
    seed: int = 42

    # --- Data split (by scenario / proposition) -----------------------------
    n_train_scenarios: int = 45  # 45*12 = 540
    n_val_scenarios: int = 5     # 5*12  =  60
    n_test_scenarios: int = 17   # 17*12 = 204

    # --- Paths ---------------------------------------------------------------
    lfud_csv: Path = REPO_DIR / "LFUD.csv"
    data_dir: Path = REPO_DIR / "data"           # small json splits (tracked-ok)
    hidden_dir: Path = SCRATCH / "hidden"        # extracted reps (gitignored)
    ckpt_dir: Path = SCRATCH / "checkpoints"
    results_dir: Path = REPO_DIR / "results"     # json metrics + figures
    # model cache honors HF_HOME so smoke + full runs share one download
    hf_cache: Path = Path(os.environ["HF_HOME"]) if os.environ.get("HF_HOME") else SCRATCH / "hf_cache"

    def __post_init__(self):
        for p in (self.data_dir, self.hidden_dir, self.ckpt_dir, self.results_dir, self.hf_cache):
            Path(p).mkdir(parents=True, exist_ok=True)


CFG = Config()

# 12 logical fallacy types in LFUD (for category analysis)
FALLACY_TYPES = [
    "faulty generalization", "false causality", "circular reasoning", "ad populum",
    "false dilemma", "fallacy of relevance", "ad hominem", "appeal to emotion",
    "fallacy of extension", "fallacy of credibility", "intentional fallacy",
    "deductive fallacy",
]
