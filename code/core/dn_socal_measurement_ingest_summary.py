import csv
from datetime import datetime
from pathlib import Path
from statistics import mean

ROOT = Path.home() / 'Desktop' / '配电网实验_临时' / 'real_data_preview_samples'
OUT = Path.home() / 'Desktop' / '配电网实验_临时' / 'socal_measurement_ingest_summary.txt'
FILES = [
    ('egauge_9-Mains_Power.csv', 'magnitude'),
    ('egauge_9-S3.csv', 'phasor_channel'),
    ('t_phasor.csv', 'phasor_timebase'),
]


def read_rows(path):
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def parse_ts(ts):
    return datetime.fromisoformat(ts)


def summarize_times(rows, field='t'):
    ts = [parse_ts(r[field]) for r in rows if r.get(field)]
    if not ts:
        return None
    deltas = [(b - a).total_seconds() for a, b in zip(ts, ts[1:])]
    uniq = sorted(set(deltas))
    return {
        'rows': len(ts),
        'first': ts[0].isoformat(timespec='microseconds'),
        'last': ts[-1].isoformat(timespec='microseconds'),
        'duration_hours': (ts[-1] - ts[0]).total_seconds() / 3600.0,
        'delta_unique': uniq[:10],
        'delta_mean': mean(deltas) if deltas else 0.0,
    }


def float_series(rows, key):
    vals = []
    for r in rows:
        v = r.get(key)
        if v in (None, ''):
            continue
        try:
            vals.append(float(v))
        except Exception:
            pass
    return vals


def main():
    lines = ['SoCal measurement ingest summary']
    for name, kind in FILES:
        path = ROOT / name
        rows = read_rows(path)
        lines.append(f'file={name}')
        lines.append(f'kind={kind}')
        lines.append('columns=' + ','.join(rows[0].keys()))
        tstats = summarize_times(rows)
        if tstats:
            lines.append(f"rows={tstats['rows']}")
            lines.append(f"first={tstats['first']}")
            lines.append(f"last={tstats['last']}")
            lines.append(f"duration_hours={tstats['duration_hours']:.3f}")
            lines.append(f"delta_mean_sec={tstats['delta_mean']:.6f}")
            lines.append('delta_unique_sec=' + ','.join(str(x) for x in tstats['delta_unique']))
        if name == 'egauge_9-Mains_Power.csv':
            vals = float_series(rows, 'v')
            lines.append(f'value_min={min(vals):.3f}')
            lines.append(f'value_max={max(vals):.3f}')
            lines.append(f'value_mean={mean(vals):.3f}')
        elif name == 'egauge_9-S3.csv':
            for key in ['frequency', 'rms', 'magnitude_harmonic_0', 'magnitude_harmonic_1', 'magnitude_harmonic_2']:
                vals = float_series(rows, key)
                if vals:
                    lines.append(f'{key}.min={min(vals):.6f}')
                    lines.append(f'{key}.max={max(vals):.6f}')
                    lines.append(f'{key}.mean={mean(vals):.6f}')
        lines.append('')

    mag_rows = read_rows(ROOT / 'egauge_9-Mains_Power.csv')
    pha_rows = read_rows(ROOT / 't_phasor.csv')
    mag_first = parse_ts(mag_rows[0]['t'])
    mag_last = parse_ts(mag_rows[-1]['t'])
    pha_first = parse_ts(pha_rows[0]['t'])
    pha_last = parse_ts(pha_rows[-1]['t'])
    overlap_start = max(mag_first, pha_first)
    overlap_end = min(mag_last, pha_last)
    overlap_sec = (overlap_end - overlap_start).total_seconds()
    lines.append('cross_track_alignment')
    lines.append(f'magnitude_phasor_overlap_hours={max(0.0, overlap_sec) / 3600.0:.3f}')
    lines.append(f'magnitude_window_inside_phasor_window={str(mag_first >= pha_first and mag_last <= pha_last).lower()}')
    lines.append('boundary=Summary covers only the locally acquired public preview measurement files and does not imply synchronized alignment with the public topology event timeline.')

    OUT.write_text('\n'.join(lines), encoding='utf-8')
    print(OUT)

if __name__ == '__main__':
    main()
