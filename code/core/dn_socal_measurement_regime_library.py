import csv
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path.home() / 'Desktop' / '配电网实验_临时' / 'real_data_preview_samples'
OUT = Path.home() / 'Desktop' / '配电网实验_临时' / 'socal_measurement_regime_library.txt'


def read_csv(path):
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def parse_ts(ts):
    return datetime.fromisoformat(ts)


def nearest_second_key(dt):
    return int(round(dt.timestamp()))


def kmeans(x, k=4, iters=30, seed=0):
    rng = np.random.RandomState(seed)
    n = x.shape[0]
    idx = rng.choice(n, size=k, replace=False)
    centers = x[idx].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        d2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(d2, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                centers[j] = x[mask].mean(axis=0)
    return labels, centers


def main():
    mag_rows = read_csv(ROOT / 'egauge_9-Mains_Power.csv')
    pha_rows = read_csv(ROOT / 'egauge_9-S3.csv')

    mag_map = {}
    for r in mag_rows:
        dt = parse_ts(r['t'])
        mag_map[nearest_second_key(dt)] = float(r['v'])

    merged = []
    for r in pha_rows:
        dt = parse_ts(r['t'])
        sec = nearest_second_key(dt)
        if sec not in mag_map:
            continue
        merged.append({
            't': dt,
            'v_mag': mag_map[sec],
            'frequency': float(r['frequency']),
            'rms': float(r['rms']),
            'mag0': float(r['magnitude_harmonic_0']),
            'mag1': float(r['magnitude_harmonic_1']),
            'mag2': float(r['magnitude_harmonic_2']),
        })

    feats = np.array([[m['v_mag'], m['frequency'], m['rms'], m['mag0'], m['mag1'], m['mag2']] for m in merged], dtype=np.float64)
    mu = feats.mean(axis=0)
    sigma = feats.std(axis=0)
    sigma[sigma == 0] = 1.0
    z = (feats - mu) / sigma
    labels, centers = kmeans(z, k=4, iters=40, seed=42)

    cluster_stats = defaultdict(list)
    for row, lab in zip(merged, labels):
        cluster_stats[int(lab)].append(row)

    lines = ['SoCal measurement regime library']
    lines.append(f'merged_rows={len(merged)}')
    lines.append(f'feature_dim={feats.shape[1]}')
    lines.append(f'merge_start={merged[0]["t"].isoformat(timespec="microseconds")}')
    lines.append(f'merge_end={merged[-1]["t"].isoformat(timespec="microseconds")}')
    lines.append(f'merge_duration_hours={(merged[-1]["t"] - merged[0]["t"]).total_seconds()/3600.0:.3f}')
    lines.append('features=v_mag,frequency,rms,mag0,mag1,mag2')

    for cid in sorted(cluster_stats):
        rows = cluster_stats[cid]
        times = [r['t'] for r in rows]
        arr = np.array([[r['v_mag'], r['frequency'], r['rms'], r['mag0'], r['mag1'], r['mag2']] for r in rows], dtype=np.float64)
        lines.append(f'cluster_{cid}.count={len(rows)}')
        lines.append(f'cluster_{cid}.first={times[0].isoformat(timespec="microseconds")}')
        lines.append(f'cluster_{cid}.last={times[-1].isoformat(timespec="microseconds")}')
        lines.append(f'cluster_{cid}.v_mag_mean={arr[:,0].mean():.3f}')
        lines.append(f'cluster_{cid}.frequency_mean={arr[:,1].mean():.6f}')
        lines.append(f'cluster_{cid}.rms_mean={arr[:,2].mean():.6f}')
        lines.append(f'cluster_{cid}.mag0_mean={arr[:,3].mean():.6f}')
        lines.append(f'cluster_{cid}.mag1_mean={arr[:,4].mean():.6f}')
        lines.append(f'cluster_{cid}.mag2_mean={arr[:,5].mean():.6f}')

    change_count = int(np.sum(labels[1:] != labels[:-1])) if len(labels) > 1 else 0
    lines.append(f'regime_transition_steps={change_count}')
    lines.append('boundary=This regime library is measurement-only over the local public preview overlap between magnitude and phasor streams; it is not synchronized with the public topology-event timeline.')
    OUT.write_text('\n'.join(lines), encoding='utf-8')
    print(OUT)

if __name__ == '__main__':
    main()
