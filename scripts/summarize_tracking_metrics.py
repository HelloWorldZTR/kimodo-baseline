"""Recompute tracking metrics from saved per-sample measurements (no inference).

Reach dimensions are evaluated separately and averaged; the overall row is
the unweighted mean of task metrics. Negative tracking R² values are retained.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def summarize(reports, output):
    protocol = json.loads((reports / 'protocol.json').read_text())
    original = json.loads((reports / 'results.json').read_text())
    with (reports / 'samples.csv').open(newline='') as stream:
        samples = list(csv.DictReader(stream))
    assert original['complete']
    assert len(samples) == original['scored_samples'] == original['expected_samples']
    assert len({row['sample_id'] for row in samples}) == len(samples)
    task_rows, dimension_rows = [], []
    metrics = ('nmae', 'r2_track', 'one_minus_r2', 'slope_a', 'abs_normalized_intercept')
    for task, previous in zip(protocol['tasks'], original['tasks']):
        name = task['name']
        assert name == previous['task']
        rows = [row for row in samples if row['task'] == name]
        assert len(rows) == previous['samples']
        dims = []
        normalized_errors = []
        for dimension, (lower, upper) in enumerate(task['definition']['bounds']):
            x = np.array([float(row[f'requested_{dimension}']) for row in rows], dtype=np.float64)
            y = np.array([float(row[f'measured_{dimension}']) for row in rows], dtype=np.float64)
            assert np.isfinite(x).all() and np.isfinite(y).all() and upper > lower
            # Verify every per-sample value against the saved, unaveraged curves.
            curves = {(c['prompt_index'], c['seed']): c for c in previous['curves']}
            positions = {index: pos for pos, index in enumerate(previous['command_indices'])}
            for row, target, measured in zip(rows, x, y):
                pos = positions[int(row['command_index'])]
                curve = curves[(int(row['prompt_index']), int(row['seed']))]
                assert target == previous['requested'][pos][dimension]
                assert measured == curve['measured'][pos][dimension]
            centered = x - x.mean()
            denominator = float(centered @ centered)
            assert denominator > 0
            error = y - x
            ratio = float(error @ error / denominator)
            slope = float(centered @ (y - y.mean()) / denominator)
            intercept = float(y.mean() - slope * x.mean())
            normalized_intercept = (intercept + (slope - 1.0) * lower) / (upper - lower)
            normalized_errors.append(np.abs(error) / (upper - lower))
            # Independent least-squares cross-check of the fitted parameters.
            fit = np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]), y, rcond=None)[0]
            np.testing.assert_allclose(fit, [slope, intercept], rtol=1e-10, atol=1e-10)
            xn, yn = (x - lower) / (upper - lower), (y - lower) / (upper - lower)
            normalized_fit = np.linalg.lstsq(np.column_stack([xn, np.ones_like(xn)]), yn, rcond=None)[0]
            np.testing.assert_allclose(normalized_fit, [slope, normalized_intercept], rtol=1e-10, atol=1e-10)
            result = dict(task=name, dimension=('XYZ'[dimension] if len(task['definition']['bounds']) == 3 else 'scalar'),
                          samples=len(rows), lower=lower, upper=upper,
                          nmae=float(np.abs(error).mean() / (upper - lower)),
                          r2_track=1.0-ratio, one_minus_r2=ratio,
                          slope_a=slope, intercept_b=intercept,
                          normalized_intercept_bn=normalized_intercept,
                          abs_normalized_intercept=abs(normalized_intercept))
            dims.append(result)
            dimension_rows.append(result)
        summary = dict(task=name, samples=len(rows), **{key: float(np.mean([d[key] for d in dims])) for key in metrics})
        per_sample_errors = np.column_stack(normalized_errors)
        for percent in (5, 10):
            epsilon = percent / 100.0
            success = np.max(per_sample_errors, axis=1) <= epsilon
            assert np.array_equal(success, np.all(per_sample_errors <= epsilon, axis=1))
            summary[f'success_at_{percent}pct'] = float(success.mean())
            summary[f'success_count_{percent}pct'] = int(success.sum())
        assert summary['success_count_10pct'] >= summary['success_count_5pct']
        np.testing.assert_allclose(summary['nmae'], previous['normalized_mae'], rtol=1e-5, atol=1e-8)
        task_rows.append(summary)
    assert sum(row['samples'] for row in task_rows) == len(samples)
    task_rows.append(dict(task='macro_average', samples=len(samples),
                          **{key: float(np.mean([row[key] for row in task_rows])) for key in (*metrics, 'success_at_5pct', 'success_at_10pct')},
                          **{f'success_count_{percent}pct': sum(row[f'success_count_{percent}pct'] for row in task_rows)
                             for percent in (5, 10)}))
    output.mkdir(parents=True, exist_ok=True)
    complete_rows = []
    units = {task['task']: task['unit'] for task in original['tasks']}
    for row in task_rows:
        components = [dim for dim in dimension_rows if dim['task'] == row['task']]
        # Keep dimensional intercepts separate; mixing m and m/s has no meaningful macro.
        complete_rows.append(dict(
            row, unit=units.get(row['task'], ''),
            intercept_b=json.dumps([dim['intercept_b'] for dim in components]) if components else '',
            intercept_components=','.join(dim['dimension'] for dim in components)))
    for filename, rows in [('task_metrics.csv', task_rows), ('dimension_metrics.csv', dimension_rows),
                           ('complete_metrics.csv', complete_rows)]:
        with (output / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result = dict(protocol_id=original['protocol_id'], run_id=original['run_id'],
                  source_sha256={name: hashlib.sha256((reports / name).read_bytes()).hexdigest()
                                 for name in ('protocol.json', 'results.json', 'samples.csv')},
                  method='All individual samples, float64; normalize by protocol bounds. '
                         'Reach: arithmetic mean of separate XYZ component metrics. '
                         'Normalized intercept bn=(b+(a-1)*L)/(U-L); average abs(bn) within each task. '
                         'Success@5%/10%: all valid components must satisfy abs(y-x)/(U-L)<=0.05/0.10 per sample. '
                         'Overall metrics: unweighted task mean (success count is a sum, not macro weighting). '
                         'No calibration, clipping, command-level averaging, or motion regeneration.',
                  tasks=task_rows, dimensions=dimension_rows)
    (output / 'tracking_metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(task_rows, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.reports, args.output)
