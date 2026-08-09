# -*- coding: utf-8 -*-
"""Transfer-learning comparison figures for the thesis (chapter 10.12).

Uses analysis/compare_runs.py to pair each baseline run with its
transfer-learning counterpart; outputs land in the figures folder under
cmp_bs/ and cmp_big/ and are referenced from the thesis markdown.
"""
import os, sys

# --- CONFIG -----------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MT = os.path.join(REPO, "master_thesis")        # runs live here
FIGURES = os.path.join(MT, "src", "figures")    # output folder
# ----------------------------------------------------------------------
sys.path.insert(0, os.path.join(REPO, "analysis"))

from compare_runs import compare

PAIRS = {
    # figures subfolder -> (baseline run, transfer run)
    "cmp_bs": ("result_data_20260705_131934", "result_data_20260706_112518"),
    "cmp_big": ("result_data_20260706_072702", "result_data_20260706_130958"),
}

for sub, (base, var) in PAIRS.items():
    compare(os.path.join(MT, base), os.path.join(MT, var),
            base_name="без трансфер", var_name="со трансфер",
            out_dir=os.path.join(FIGURES, sub), lang="mk",
            data_root=os.path.join(REPO, "data"))

print("transfer figs done")
