"""Discovery of model names, labels and colors.

Nothing here is hardcoded to the four current models: run folders are scanned
for whichever model subfolders exist, and display labels come from the
``models/`` package file names via the MODEL_CLASSES mapping in
``exp/exp_main.py`` (e.g. run key 'TST' -> models/PatchTST.py -> label
'PatchTST'). A new model added to models/ and MODEL_CLASSES is picked up
automatically.
"""

import glob
import os
import re

# Fixed categorical palette (colorblind-validated ordering). Colors are
# assigned to models by sorted label, so the same model always gets the same
# color across figures and runs.
_PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
            "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def known_model_files(root=None):
    """Model names as defined by the file names in models/ (without .py)."""
    root = root or repo_root()
    stems = [os.path.splitext(os.path.basename(p))[0]
             for p in glob.glob(os.path.join(root, "models", "*.py"))]
    return sorted(s for s in stems if not s.startswith("__"))


def model_key_to_label(root=None):
    """Map run-folder model keys to models/ file names.

    Parsed from the MODEL_CLASSES dict in exp/exp_main.py, e.g.
    ``'TST': PatchTST`` -> {'TST': 'PatchTST'}. Keys without an entry fall
    back to themselves.
    """
    root = root or repo_root()
    mapping = {}
    exp_main = os.path.join(root, "exp", "exp_main.py")
    try:
        with open(exp_main, encoding="utf-8") as f:
            src = f.read()
        block = re.search(r"MODEL_CLASSES\s*=\s*\{(.*?)\}", src, re.S)
        if block:
            valid = set(known_model_files(root))
            for key, val in re.findall(r"['\"](\w+)['\"]\s*:\s*(\w+)", block.group(1)):
                mapping[key] = val if val in valid else key
    except OSError:
        pass
    return mapping


def discover_run_models(run_dir):
    """Model subfolders of a result_data_* run directory (those that actually
    produced a result.txt)."""
    models = []
    for entry in sorted(os.listdir(run_dir)):
        sub = os.path.join(run_dir, entry)
        if os.path.isdir(sub) and os.path.exists(os.path.join(sub, "result.txt")):
            models.append(entry)
    if not models:
        raise FileNotFoundError(
            f"No model subfolders with result.txt found in {run_dir!r} — "
            "expected the layout result_data_<ts>/<model>/result.txt")
    return models


def label_for(model_key, root=None):
    return model_key_to_label(root).get(model_key, model_key)


def assign_colors(labels):
    """Stable label -> color assignment (sorted order, fixed palette)."""
    ordered = sorted(set(labels))
    if len(ordered) > len(_PALETTE):
        print(f"Warning: {len(ordered)} models > {len(_PALETTE)} palette slots; "
              "colors will repeat — consider comparing fewer models per figure.")
    return {lab: _PALETTE[i % len(_PALETTE)] for i, lab in enumerate(ordered)}
