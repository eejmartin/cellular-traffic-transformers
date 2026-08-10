"""Aggregate the full thesis campaign matrix under master_thesis_final/.

Discovers and classifies every run folder produced by the phase-2 scripts —
base campaigns (5 seeds x 2 sets), transfer regimes (sequential/ordered/
grouped/federated/combined), ablations, cold-start folds, stability samples
and the CPU baselines — and writes one tidy CSV per thesis table plus a
printed summary. Safe to run mid-campaign: sections whose folders are
missing are skipped and appear on the next invocation.

Outputs (into <root>/matrix_<ts>/):
    01_base_seeds.csv      set x model: mean +/- std over seeds (RSE, MAE, ...)
    02_regimes.csv         set x model x regime: RSE, delta vs none, % improved
    03_wilcoxon.csv        paired Wilcoxon tests with Holm correction
    04_ablation_<name>.csv one per ablation family
    05_coldstart.csv       zero-shot / fine-tune / from-scratch per model
    06_stability.csv       primary vs random Set 2 samples
    07_baselines.csv       naive/seasonal/linear vs the models (win rates)

Usage: python analysis/campaign_matrix.py [--root master_thesis_final]
"""

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.metrics_parser import parse_run  # noqa: E402

EXT_FIELDS = ('smape', 'wape', 'mase', 'me', 'peak_mae', 'under_rate', 'peak_under_rate')

RUN_RE = re.compile(
    r'^result_data_(?P<ds>bs|big_data_l100)'
    r'(?:_transfer-(?P<mode>[a-z]+)(?:-g\d+)?(?:-r\d+-e\d+)?)?'
    r'_s(?P<seed>\d+)_\d{8}_\d{6}$')


def attach_ext(df, run_dir):
    """Merge metrics_ext.json values into the parsed frame."""
    ext_rows = []
    for model in df['model'].unique():
        for setting in df.loc[df.model == model, 'setting']:
            p = os.path.join(run_dir, model, 'results', setting, 'metrics_ext.json')
            row = {'model': model, 'setting': setting}
            if os.path.exists(p):
                with open(p) as f:
                    row.update({k: v for k, v in json.load(f).items() if k in EXT_FIELDS})
            ext_rows.append(row)
    return df.merge(pd.DataFrame(ext_rows), on=['model', 'setting'], how='left')


def load_std(run_dir):
    df = attach_ext(parse_run(run_dir), run_dir)
    return df


def classify(root):
    """Walk root, classify folders. Returns dict of lists of (meta, path)."""
    out = {'base': [], 'transfer': [], 'ablation': [], 'coldstart': [],
           'stability': [], 'baselines': []}
    for p in sorted(glob.glob(os.path.join(root, '*'))):
        name = os.path.basename(p)
        if not os.path.isdir(p) or name in ('old_runs', 'ablation_data') \
                or name.endswith('_retest') or name.startswith('hp_search') \
                or '_l1_' in name:
            continue
        m = RUN_RE.match(name)
        if m:
            meta = {'set': 'Set1' if m.group('ds') == 'bs' else 'Set2',
                    'mode': m.group('mode') or 'none', 'seed': int(m.group('seed'))}
            out['transfer' if m.group('mode') else 'base'].append((meta, p))
            continue
        m = re.match(r'^ablation_(?P<tag>.+)_s(?P<seed>\d+)_\d{8}_\d{6}$', name)
        if m:
            out['ablation'].append(({'tag': m.group('tag'), 'seed': int(m.group('seed'))}, p))
            continue
        m = re.match(r'^coldstart_k(?P<k>\d+)_s(?P<seed>\d+)_\d{8}_\d{6}$', name)
        if m:
            out['coldstart'].append(({'k': int(m.group('k')), 'seed': int(m.group('seed'))}, p))
            continue
        m = re.match(r'^stability_sample(?P<i>\d+)_s(?P<seed>\d+)_\d{8}_\d{6}$', name)
        if m:
            out['stability'].append(({'i': int(m.group('i')), 'seed': int(m.group('seed'))}, p))
            continue
        if name.startswith('baselines_'):
            out['baselines'].append(({}, p))
    return out


def holm(pvals):
    """Holm step-down adjusted p-values."""
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def station_frame(entries, extra_cols):
    frames = []
    for meta, p in entries:
        try:
            df = load_std(p)
        except Exception as e:
            print(f'  skip {os.path.basename(p)}: {e}')
            continue
        for k, v in meta.items():
            df[k] = v
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument('--root', default='master_thesis_final')
    cli.add_argument('-o', '--out', default=None)
    args = cli.parse_args()

    out_dir = args.out or os.path.join(args.root, 'matrix_' + time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(out_dir, exist_ok=True)
    cat = classify(args.root)
    for k, v in cat.items():
        print(f'{k}: {len(v)} folders')

    # ---- 01 base: mean +/- std over seeds --------------------------------
    base = station_frame(cat['base'], ['set', 'seed'])
    if not base.empty:
        per_seed = base.groupby(['set', 'model', 'seed']).agg(
            rse=('rse', 'mean'), mae=('mae', 'mean'), smape=('smape', 'mean'),
            wape=('wape', 'mean'), mase=('mase', 'mean'),
            under=('under_rate', 'mean')).reset_index()
        t01 = per_seed.groupby(['set', 'model']).agg(
            seeds=('seed', 'nunique'),
            rse_mean=('rse', 'mean'), rse_std=('rse', 'std'),
            mae_mean=('mae', 'mean'), mae_std=('mae', 'std'),
            smape_mean=('smape', 'mean'), wape_mean=('wape', 'mean'),
            mase_mean=('mase', 'mean'), under_mean=('under', 'mean')).round(4)
        t01.to_csv(os.path.join(out_dir, '01_base_seeds.csv'))
        print('\n== 01 base (mean over seeds) ==')
        print(t01[['seeds', 'rse_mean', 'rse_std', 'mase_mean']].to_string())

    # ---- 02 regimes ------------------------------------------------------
    tr = station_frame(cat['transfer'], ['set', 'mode', 'seed'])
    if not tr.empty and not base.empty:
        allr = pd.concat([base.assign(mode='none'), tr], ignore_index=True)
        per = allr.groupby(['set', 'model', 'mode', 'seed'])['rse'].mean() \
                  .groupby(['set', 'model', 'mode']).agg(['mean', 'std', 'count'])
        rows = []
        for (s, mdl), g in per.groupby(['set', 'model']):
            none = g.loc[(s, mdl, 'none'), 'mean'] if (s, mdl, 'none') in g.index else np.nan
            for (_, _, mode), r in g.iterrows():
                # paired improvement share on the common seed
                imp = np.nan
                if mode != 'none':
                    a = allr[(allr.set == s) & (allr.model == mdl) &
                             (allr['mode'] == 'none') & (allr.seed == 2021)]
                    b = allr[(allr.set == s) & (allr.model == mdl) &
                             (allr['mode'] == mode) & (allr.seed == 2021)]
                    m = a.merge(b, on='station', suffixes=('_0', '_1'))
                    if len(m):
                        imp = float((m.rse_1 < m.rse_0).mean())
                rows.append({'set': s, 'model': mdl, 'mode': mode,
                             'rse_mean': r['mean'], 'rse_std': r['std'],
                             'seeds': int(r['count']), 'delta_vs_none': r['mean'] - none,
                             'improved_share': imp})
        t02 = pd.DataFrame(rows).round(4)
        t02.to_csv(os.path.join(out_dir, '02_regimes.csv'), index=False)
        print('\n== 02 regimes (mean RSE) ==')
        print(t02.pivot_table(index=['set', 'model'], columns='mode',
                              values='rse_mean').round(3).to_string())

        # ---- 03 Wilcoxon: regime vs none, per set x model ----------------
        try:
            from scipy.stats import wilcoxon
            wrows = []
            for (s, mdl, mode), _ in tr.groupby(['set', 'model', 'mode']):
                a = base[(base.set == s) & (base.model == mdl) & (base.seed == 2021)]
                b = tr[(tr.set == s) & (tr.model == mdl) & (tr['mode'] == mode) &
                       (tr.seed == 2021)]
                m = a.merge(b, on='station', suffixes=('_0', '_1'))
                if len(m) < 10:
                    continue
                stat, p = wilcoxon(m.rse_0, m.rse_1)
                wrows.append({'set': s, 'model': mdl, 'contrast': f'{mode} vs none',
                              'n': len(m), 'median_delta': float((m.rse_1 - m.rse_0).median()),
                              'p_raw': p})
            if wrows:
                t03 = pd.DataFrame(wrows)
                t03['p_holm'] = holm(t03['p_raw'].values)
                t03['significant_0.05'] = t03['p_holm'] < 0.05
                t03 = t03.round(5)
                t03.to_csv(os.path.join(out_dir, '03_wilcoxon.csv'), index=False)
                print(f'\n== 03 wilcoxon: {len(t03)} contrasts, '
                      f'{int(t03["significant_0.05"].sum())} significant after Holm ==')
        except ImportError:
            print('scipy missing — skipping Wilcoxon')

    # ---- 04 ablations ----------------------------------------------------
    for meta, p in cat['ablation']:
        try:
            df = load_std(p)
        except Exception as e:
            print(f'  skip ablation {meta["tag"]}: {e}')
            continue
        t = df.groupby('model').agg(rse_mean=('rse', 'mean'),
                                    rse_median=('rse', 'median'),
                                    mae_mean=('mae', 'mean'),
                                    n=('station', 'size')).round(4)
        t.to_csv(os.path.join(out_dir, f'04_ablation_{meta["tag"]}.csv'))
    if cat['ablation']:
        print(f'\n== 04 ablations: {len(cat["ablation"])} campaigns exported ==')

    # ---- 05 cold-start ---------------------------------------------------
    for meta, p in cat['coldstart']:
        rows = []
        for fold_dir in sorted(glob.glob(os.path.join(p, 'fold*'))):
            for md in sorted(glob.glob(os.path.join(fold_dir, '*_zeroshot'))) + \
                      sorted(glob.glob(os.path.join(fold_dir, '*_finetune'))):
                model, mode = os.path.basename(md).rsplit('_', 1)
                rt = os.path.join(md, 'result.txt')
                if not os.path.exists(rt):
                    continue
                with open(rt) as f:
                    lines = [l.strip() for l in f if l.strip()]
                for i in range(0, len(lines) - 1, 2):
                    mm = re.match(r'mse:([\d.e+-]+), mae:([\d.e+-]+), rse:([\d.enan+-]+)',
                                  lines[i + 1])
                    if mm:
                        rows.append({'fold': os.path.basename(fold_dir), 'model': model,
                                     'mode': mode, 'setting': lines[i],
                                     'mse': float(mm.group(1)), 'mae': float(mm.group(2)),
                                     'rse': float(mm.group(3))})
        if rows:
            cs = pd.DataFrame(rows)
            t05 = cs.groupby(['model', 'mode']).agg(
                rse_mean=('rse', 'mean'), rse_median=('rse', 'median'),
                n=('rse', 'size')).round(4)
            t05.to_csv(os.path.join(out_dir, '05_coldstart.csv'))
            print('\n== 05 cold-start ==')
            print(t05.to_string())

    # ---- 06 stability ----------------------------------------------------
    st = station_frame(cat['stability'], ['i', 'seed'])
    if not st.empty:
        t06 = st.groupby(['i', 'model']).agg(rse_mean=('rse', 'mean'),
                                             rse_median=('rse', 'median')).round(4)
        t06.to_csv(os.path.join(out_dir, '06_stability.csv'))
        print('\n== 06 stability samples ==')
        print(t06.to_string())

    # ---- 07 baselines + win rates ---------------------------------------
    for _, p in cat['baselines']:
        bl = pd.read_csv(os.path.join(p, 'baseline_metrics.csv'))
        t07 = bl.groupby(['set', 'baseline']).agg(
            rse_mean=('rse', 'mean'), rse_median=('rse', 'median'),
            mase_mean=('mase', 'mean')).round(4)
        # model-vs-naive win rate on seed 2021 base runs
        if not base.empty:
            naive = bl[bl.baseline == 'naive'][['set', 'station', 'rse']] \
                .rename(columns={'rse': 'rse_naive'})
            b21 = base[base.seed == 2021].merge(naive, on=['set', 'station'])
            wins = b21.groupby(['set', 'model']) \
                .apply(lambda g: (g.rse < g.rse_naive).mean(), include_groups=False) \
                .rename('beats_naive_share').round(3)
            wins.to_csv(os.path.join(out_dir, '07_beats_naive.csv'))
            print('\n== 07 share of stations where model beats naive (seed 2021) ==')
            print(wins.to_string())
        t07.to_csv(os.path.join(out_dir, '07_baselines.csv'))

    print('\nsaved ->', out_dir)


if __name__ == '__main__':
    main()
