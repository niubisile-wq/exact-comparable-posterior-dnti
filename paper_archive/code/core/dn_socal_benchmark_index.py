import csv
from pathlib import Path
from datetime import datetime

ROOT = Path.home() / 'Desktop' / '配电网实验_临时'
TOPO_ROOT = ROOT / 'digital-twin-dataset' / 'sample_dataset' / 'topology'
STATUS_DIR = TOPO_ROOT / 'parameter_timeseries'
META_DIR = TOPO_ROOT / 'metadata'
PREVIEW_DIR = ROOT / 'real_data_preview_samples'
OUT = ROOT / 'socal_benchmark_index.txt'


def read_csv_rows(path):
    with path.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def time_bounds_from_csv(path, field='t'):
    rows = read_csv_rows(path)
    if not rows:
        return None, None, 0
    vals = [r[field] for r in rows if r.get(field)]
    return vals[0], vals[-1], len(vals)


def main():
    lines = ['SoCal public sample benchmark index']

    netfile_changes = META_DIR / 'netfile_changes.csv'
    parameter_changes = META_DIR / 'parameter_changes.csv'
    if netfile_changes.exists():
        a,b,n = time_bounds_from_csv(netfile_changes)
        lines.append(f'metadata.netfile_changes.rows={n}')
        lines.append(f'metadata.netfile_changes.first={a}')
        lines.append(f'metadata.netfile_changes.last={b}')
    if parameter_changes.exists():
        a,b,n = time_bounds_from_csv(parameter_changes)
        lines.append(f'metadata.parameter_changes.rows={n}')
        lines.append(f'metadata.parameter_changes.first={a}')
        lines.append(f'metadata.parameter_changes.last={b}')

    status_files = sorted(STATUS_DIR.glob('*_status.csv'))
    lines.append(f'topology.status_file_count={len(status_files)}')
    timeline = {}
    total_rows = 0
    for path in status_files:
        rows = read_csv_rows(path)
        total_rows += len(rows)
        for row in rows:
            timeline.setdefault(row['t'], {})[path.stem.replace('_status','')] = row['str']
    ts_sorted = sorted(timeline)
    lines.append(f'topology.status_rows={total_rows}')
    if ts_sorted:
        lines.append(f'topology.first_snapshot={ts_sorted[0]}')
        lines.append(f'topology.last_snapshot={ts_sorted[-1]}')
        lines.append(f'topology.snapshot_count={len(ts_sorted)}')

    preview_files = sorted(PREVIEW_DIR.glob('*.csv'))
    lines.append(f'preview.file_count={len(preview_files)}')
    for path in preview_files:
        field = 't'
        a,b,n = time_bounds_from_csv(path, field=field)
        lines.append(f'preview.{path.name}.rows={n}')
        lines.append(f'preview.{path.name}.first={a}')
        lines.append(f'preview.{path.name}.last={b}')

    topo_last = datetime.fromisoformat(ts_sorted[-1]) if ts_sorted else None
    preview_first_candidates = []
    for path in preview_files:
        a,b,n = time_bounds_from_csv(path)
        if a:
            preview_first_candidates.append(datetime.fromisoformat(a))
    if topo_last and preview_first_candidates:
        earliest_preview = min(preview_first_candidates)
        gap = earliest_preview - topo_last
        lines.append(f'alignment.public_gap_hours={gap.total_seconds()/3600.0:.2f}')
        lines.append(f'alignment.overlap_available={str(gap.total_seconds() <= 0).lower()}')

    lines.append('benchmark_tracks')
    lines.append('- track_1: topology_state_graph from metadata + switch-status sample')
    lines.append('- track_2: preview_measurement_ingest from magnitude/phasor CSV samples')
    lines.append('- track_3: synchronized_posterior_benchmark blocked by public time-window gap')
    lines.append('boundary=Current local SoCal sample supports topology-state indexing and measurement-ingest indexing, but not synchronized posterior benchmarking.')
    OUT.write_text('\n'.join(lines), encoding='utf-8')
    print(OUT)

if __name__ == '__main__':
    main()
