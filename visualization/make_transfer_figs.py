# -*- coding: utf-8 -*-
"""Transfer-learning comparison figures for the thesis (chapter 10.12).

Uses analysis/compare_runs.py to pair each baseline run with its
transfer-learning counterpart; outputs land in the figures folder under
cmp_bs/ and cmp_big/ and are referenced from the thesis markdown.
"""
import os, sys

# --- CONFIG -----------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MT = os.path.join(REPO, "master_thesis_final")  # v2 campaign runs
FIGURES = os.path.join(REPO, "master_thesis", "src", "figures")   # output folder
# ----------------------------------------------------------------------
sys.path.insert(0, os.path.join(REPO, "analysis"))

from compare_runs import compare

PAIRS = {
    # figures subfolder -> (baseline run, transfer run)
    "cmp_bs": ("result_data_bs_s2021_20260809_072721", "result_data_bs_transfer-sequential_s2021_20260809_201853"),
    "cmp_big": ("result_data_big_data_l100_s2021_20260809_084717", "result_data_big_data_l100_transfer-sequential_s2021_20260809_215939"),
}

for sub, (base, var) in PAIRS.items():
    compare(os.path.join(MT, base), os.path.join(MT, var),
            base_name="без трансфер", var_name="со трансфер",
            out_dir=os.path.join(FIGURES, sub), lang="mk",
            data_root=os.path.join(REPO, "data"))

print("transfer figs done")
