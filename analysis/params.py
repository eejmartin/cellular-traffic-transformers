"""Optional: count trainable parameters per model by rebuilding each model
from a saved config JSON (one per model per run). Requires torch and the
repository's models/ package; degrades gracefully when unavailable.
"""

import argparse
import glob
import importlib
import json
import os
import sys
import warnings

from models_registry import discover_run_models, label_for, model_key_to_label, repo_root


def count_params(run_dirs, run_names=None):
    """{run_name: {label: n_trainable_params}} for every model in every run.

    Returns an empty dict when torch (or a model import) is unavailable.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        print("params: torch not available — skipping parameter counts")
        return {}

    root = repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    key_to_module = model_key_to_label(root)
    run_names = run_names or [os.path.basename(os.path.abspath(d).rstrip("/")) for d in run_dirs]

    out = {}
    warnings.filterwarnings("ignore")
    for run_dir, run_name in zip(run_dirs, run_names):
        counts = {}
        for model_key in discover_run_models(run_dir):
            cfgs = sorted(glob.glob(os.path.join(run_dir, model_key, "configs", "*.json")))
            if not cfgs:
                continue
            try:
                with open(cfgs[0], encoding="utf-8") as f:
                    cfg = json.load(f)
                module = importlib.import_module(
                    "models." + key_to_module.get(model_key, model_key))
                model = module.Model(argparse.Namespace(**cfg))
                counts[label_for(model_key)] = sum(
                    p.numel() for p in model.parameters() if p.requires_grad)
            except Exception as exc:  # any model may fail to rebuild — report, move on
                print(f"params: could not rebuild {model_key} from {cfgs[0]}: {exc}")
        if counts:
            out[run_name] = counts
    return out
