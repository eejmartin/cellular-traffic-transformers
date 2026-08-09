"""Transfer-learning strategies for run.py.

Three upgrades over the plain sequential warm-start chain (-t):

  ordered    stations sorted so that consecutive stations have similar
             normalized daily traffic profiles (greedy nearest-neighbour
             chain, started at the highest-volume station) — the warm-start
             always comes from a similar station.
  grouped    stations clustered by daily profile (KMeans); one independent
             transfer chain per cluster, so dissimilar stations never share
             weights. Each cluster is internally similarity-ordered.
  federated  FedAvg pretraining: R rounds of {local training on every
             station from shared global weights -> sample-weighted averaging
             of the resulting weights}. The final global weights become the
             warm-start for every station's normal fine-tuning (no chaining).

Profiles are computed on the training split only (first `train_ratio` of
each series) so ordering/grouping never sees validation or test data.
"""

import copy
import os

import numpy as np
import pandas as pd


# --------------------------------------------------------------- profiles

def _profile(file_path, train_ratio=0.7):
    """(24-dim z-normalized mean-by-hour `users` profile, mean volume)."""
    df = pd.read_csv(os.path.join(file_path['root_path'], file_path['data_path']))
    n = int(len(df) * train_ratio)
    users = df['users'].values[:n].astype(float)
    hours = pd.to_datetime(df['time_hour']).dt.hour.values[:n]
    prof = np.zeros(24)
    for h in range(24):
        vals = users[hours == h]
        prof[h] = vals.mean() if len(vals) else 0.0
    std = prof.std()
    shape = (prof - prof.mean()) / std if std > 0 else np.zeros(24)
    return shape, float(users.mean())


def station_features(file_paths, train_ratio=0.7):
    shapes, volumes = [], []
    for fp in file_paths:
        s, v = _profile(fp, train_ratio)
        shapes.append(s)
        volumes.append(v)
    return np.asarray(shapes), np.asarray(volumes)


# --------------------------------------------------------------- ordering

def order_by_similarity(file_paths, train_ratio=0.7):
    """Greedy nearest-neighbour chain over daily profiles.

    Starts at the highest-volume station (most signal to learn from) and
    always continues with the most similar remaining station, so every
    warm-start comes from a close neighbour in profile space.
    """
    if len(file_paths) <= 2:
        return list(file_paths)
    shapes, volumes = station_features(file_paths, train_ratio)
    current = int(np.argmax(volumes))
    order = [current]
    remaining = set(range(len(file_paths))) - {current}
    while remaining:
        dists = {j: float(np.linalg.norm(shapes[current] - shapes[j])) for j in remaining}
        current = min(dists, key=dists.get)
        order.append(current)
        remaining.remove(current)
    return [file_paths[i] for i in order]


# --------------------------------------------------------------- grouping

def group_stations(file_paths, n_groups, train_ratio=0.7):
    """KMeans clusters over daily profiles; each cluster similarity-ordered.

    Returns a list of station lists (largest cluster first). run.py resets
    the transfer checkpoint between clusters, so each cluster is its own
    independent warm-start chain.
    """
    if len(file_paths) <= n_groups:
        return [[fp] for fp in file_paths]
    from sklearn.cluster import KMeans
    shapes, _ = station_features(file_paths, train_ratio)
    k = min(n_groups, len(file_paths))
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(shapes)
    groups = []
    for g in range(k):
        members = [file_paths[i] for i in range(len(file_paths)) if labels[i] == g]
        if members:
            groups.append(order_by_similarity(members, train_ratio))
    groups.sort(key=len, reverse=True)
    return groups


# --------------------------------------------------------------- federated

def federated_pretrain(args_list, rounds=5, local_epochs=5):
    """FedAvg over all stations; returns the global state_dict.

    Every round, each station trains `local_epochs` epochs from the current
    global weights on its own training split; the new global weights are the
    sample-count-weighted average of all station weights. Non-float buffers
    are taken from the first station. The same model instance is reused for
    all stations (weights swapped in/out), so memory stays flat.
    """
    import torch
    from torch import nn, optim
    from exp.exp_main import Exp_Main
    from data_provider.data_factory import data_provider

    base = copy.deepcopy(args_list[0])
    base.transfer = False           # never warm-start the federated model itself
    exp = Exp_Main(base)
    global_state = copy.deepcopy(exp.model.state_dict())
    criterion = nn.MSELoss()

    loaders = [data_provider(a, 'train')[1] for a in args_list]

    for r in range(rounds):
        states, weights = [], []
        for a, loader in zip(args_list, loaders):
            exp.model.load_state_dict(global_state)
            optimizer = optim.Adam(exp.model.parameters(), lr=a.learning_rate)
            exp.model.train()
            for _ in range(local_epochs):
                for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
                    batch_x = batch_x.float().to(exp.device)
                    batch_y = batch_y.float().to(exp.device)
                    batch_x_mark = batch_x_mark.float().to(exp.device)
                    batch_y_mark = batch_y_mark.float().to(exp.device)
                    dec_inp = torch.zeros_like(batch_y[:, -a.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :a.label_len, :], dec_inp],
                                        dim=1).float().to(exp.device)
                    outputs = exp._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if a.features == 'MS' else 0
                    loss = criterion(outputs[:, -a.pred_len:, f_dim:],
                                     batch_y[:, -a.pred_len:, f_dim:])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            states.append({k: v.detach().cpu().clone()
                           for k, v in exp.model.state_dict().items()})
            weights.append(len(loader.dataset))

        total = float(sum(weights))
        global_state = {}
        for key, ref in states[0].items():
            if ref.dtype.is_floating_point:
                global_state[key] = sum(s[key] * (w / total)
                                        for s, w in zip(states, weights))
            else:
                global_state[key] = ref
        print(f'federated: round {r + 1}/{rounds} averaged over '
              f'{len(args_list)} stations')

    return global_state
