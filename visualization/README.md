# Visualization toolkit

Figure-generation and document-build scripts for presenting analysis results
— originally written for the master's thesis, reusable for any future runs,
datasets or reports. Generic result analysis (parsing, aggregate charts,
run comparison) lives in `analysis/`; these scripts build on it and add the
presentation layer: schematic diagrams, dataset showcases, curated
per-station figures and the thesis docx build.

Each script has a marked `CONFIG` block at the top (paths, run IDs, output
folder) — edit it to point at new runs or a different output location. By
default everything reads from and writes into the author's local thesis
working folder (`master_thesis/`, not part of this repository), so point the
CONFIG paths at your own run folders when reusing.

## Scripts

| Script | What it makes | Inputs |
|---|---|---|
| `make_diagrams.py` | schematic diagrams: mobile network, O-RAN, research pipeline, sliding window, train/val/test split, per-model architecture schematics (Macedonian labels) | none (self-contained; one example CSV from `data/`) |
| `make_charts.py` | dataset showcase figures (daily cycles, station categories, 48h big_data series) + aggregate comparison charts (RSE/MAE bars, boxplots, wins, category/tech breakdown, parameter counts) | metrics CSV (from `analysis/run_analysis.py`), `data/`, run configs (for parameter counts) |
| `make_pred_figs.py` | true-vs-predicted grids and overlays for selected stations + copies of the pipeline's original plots | `pred.npy`/`true.npy` in the run folders |
| `make_transfer_figs.py` | baseline-vs-transfer pairwise comparison figures (wraps `analysis/compare_runs.py`) | two run pairs (baseline, transfer) |
| `make_strategy_figs.py` | 3-way transfer comparison (baseline / sequential / combined) per model, both datasets | baseline + sequential + combined runs |
| `build_docx.py` | builds `master_thesis/магистерски_труд.docx` from the markdown in `master_thesis/src/thesis/` following the FEIT урнек: A4 + sections (roman front matter / arabic body), Heading 1-3 with automatic multilevel numbering (literal numbers stripped at build time), live TOC / list-of-figures / list-of-tables fields (`<!--toc-->`/`<!--lof-->`/`<!--lot-->` markers in part0.md), Caption style with SEQ numbering, citation footnotes auto-converted to numbered `[n]` references matched against the bibliography, `$...$` → Word equations. Open the result in Word and press Ctrl+A, F9 to populate all fields. | thesis markdown + generated figures |

## Rebuild the thesis document

```bash
pip install pypandoc-binary python-docx      # one-time
python visualization/make_diagrams.py
python visualization/make_charts.py          # torch needed only for the params figure
python visualization/make_pred_figs.py
python visualization/make_transfer_figs.py
python visualization/make_strategy_figs.py
python visualization/build_docx.py           # -> master_thesis/магистерски_труд.docx
```

Only `build_docx.py` is needed when just the text changed.

## Reuse for a new analysis

1. Run the trainings (`run.py`, optionally `-t`) — each produces a
   `result_data_<timestamp>/` folder.
2. Regenerate the metrics CSV:
   `python analysis/run_analysis.py <run1> <run2> --names bs big -o master_thesis`
   (or point `METRICS_CSV` in `make_charts.py` elsewhere; the CSV needs the
   columns run/model/station/mse/mae/rse/category).
3. Update the run IDs in the `CONFIG` blocks of `make_charts.py`,
   `make_pred_figs.py`, `make_transfer_figs.py`, `make_strategy_figs.py`.
4. Rerun the scripts above. If the thesis text should reflect the new runs,
   also update the run IDs mentioned in `master_thesis/src/thesis/part4.md`
   (§8.11) and `part5.md` (§10.1, §10.12).

## Where the thesis content lives

The thesis **text** (`master_thesis/src/thesis/part0.md … part6.md`) and the
generated **figures** (`master_thesis/src/figures/`) stay inside
`master_thesis/`, next to the docx and the training runs — that folder is
deliberately untracked. This folder (`visualization/`) contains only code, so
the tooling survives in git even though the thesis assets do not. Back up
`master_thesis/` separately if you care about the text sources.
