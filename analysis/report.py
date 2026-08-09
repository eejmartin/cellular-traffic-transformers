"""Markdown report generation: aggregate tables, findings and figure links."""

import os

STRINGS = {
    "en": dict(
        title="# Results analysis\n",
        runs_header="Analyzed runs: {runs}. Models: {models}.\n",
        summary="## Aggregate results — {run}\n",
        cols=["model", "stations", "MSE mean", "MSE median", "MAE mean",
              "MAE median", "RSE mean", "RSE median", "RSE≥1 (%)", "wins"],
        findings="## Key findings\n",
        best="- **{run}**: best model by mean RSE is **{model}** "
             "(RSE {rse:.3f}, lowest MSE on {wins} of {n} stations).",
        naive_warn="  Note: mean RSE ≥ 1 — on average predictions do not beat "
                   "the naive mean; treat automation with care.",
        groups="## Breakdown by station group — {run}\n",
        figures="## Figures\n",
        stations_note="Prediction figures are drawn for representative "
                      "stations (best / median / worst by median RSE): {stations}.\n",
    ),
    "mk": dict(
        title="# Анализа на резултатите\n",
        runs_header="Анализирани извршувања: {runs}. Модели: {models}.\n",
        summary="## Агрегирани резултати — {run}\n",
        cols=["модел", "станици", "MSE просек", "MSE медијана", "MAE просек",
              "MAE медијана", "RSE просек", "RSE медијана", "RSE≥1 (%)", "победи"],
        findings="## Клучни наоди\n",
        best="- **{run}**: најдобар модел по просечен RSE е **{model}** "
             "(RSE {rse:.3f}, најнизок MSE на {wins} од {n} станици).",
        naive_warn="  Забелешка: просечниот RSE ≥ 1 — прогнозите во просек не "
                   "ја надминуваат наивната средина; внимателно со автоматизација.",
        groups="## Преглед по група на станици — {run}\n",
        figures="## Слики\n",
        stations_note="Фигурите со прогнози се за репрезентативни станици "
                      "(најдобра / средна / најлоша по медијански RSE): {stations}.\n",
    ),
}


def _md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out) + "\n"


def write_report(df, summary, out_dir, figure_paths, station_map, lang="en",
                 fname="analysis_report.md"):
    t = STRINGS[lang]
    runs = list(dict.fromkeys(df["run"]))
    models = sorted(df["label"].unique())
    lines = [t["title"]]
    lines.append(t["runs_header"].format(runs=", ".join(runs), models=", ".join(models)))

    for run in runs:
        lines.append(t["summary"].format(run=run))
        sub = summary[summary.run == run]
        rows = [[r.label, r.stations,
                 f"{r.mse_mean:,.1f}", f"{r.mse_median:,.1f}",
                 f"{r.mae_mean:.2f}", f"{r.mae_median:.2f}",
                 f"{r.rse_mean:.3f}", f"{r.rse_median:.3f}",
                 f"{100 * r.rse_ge_1:.0f}", r.wins]
                for r in sub.itertuples()]
        lines.append(_md_table(t["cols"], rows))

    lines.append(t["findings"])
    for run in runs:
        sub = summary[summary.run == run].sort_values("rse_mean")
        best = sub.iloc[0]
        n = df[df.run == run]["station"].nunique()
        lines.append(t["best"].format(run=run, model=best.label,
                                      rse=best.rse_mean, wins=best.wins, n=n))
        if best.rse_mean >= 1:
            lines.append(t["naive_warn"])
    lines.append("")

    for run in runs:
        sub = df[df.run == run]
        if sub["group"].nunique() > 1:
            lines.append(t["groups"].format(run=run))
            piv = sub.groupby(["group", "label"])["rse"].mean().unstack().round(3)
            lines.append(_md_table([""] + list(piv.columns),
                                   [[g] + [f"{v:.3f}" for v in piv.loc[g]] for g in piv.index]))

    if station_map:
        parts = [f"{run}: {', '.join(sts)}" for run, sts in station_map.items()]
        lines.append(t["stations_note"].format(stations="; ".join(parts)))

    lines.append(t["figures"])
    for p in figure_paths:
        if p:
            rel = os.path.basename(p)
            lines.append(f"![{rel}]({rel})\n")

    path = os.path.join(out_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path
