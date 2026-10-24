"""Plot a saved nine-task response panel, optionally paired with a prior run.

Reads reports only. Never generates motion or recomputes task measurements.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_results(folder):
    return json.loads((folder/'results.json').read_text())


def verify_pair(before_folder, after_folder, before, after):
    before_protocol = json.loads((before_folder/'protocol.json').read_text())
    after_protocol = json.loads((after_folder/'protocol.json').read_text())
    for field in ('tasks', 'fps', 'seeds', 'settings', 'scoring', 'sources'):
        if before_protocol[field] != after_protocol[field]:
            raise ValueError('Comparison changed benchmark field: ' + field)
    if [task['task'] for task in before['tasks']] != [task['task'] for task in after['tasks']]:
        raise ValueError('Different task panels')
    for old, new in zip(before['tasks'], after['tasks']):
        for field in ('command_indices', 'requested', 'samples', 'frames_per_command'):
            if old[field] != new[field]:
                raise ValueError(f"Different {field} for {new['task']}")
        old_keys = [(curve['prompt_index'], curve['seed'], curve['prompt']) for curve in old['curves']]
        new_keys = [(curve['prompt_index'], curve['seed'], curve['prompt']) for curve in new['curves']]
        if old_keys != new_keys:
            raise ValueError('Different prompt/seed pairing')


def plot_responses(report_folder, output, before_folder=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    after = load_results(report_folder)
    before = load_results(before_folder) if before_folder else None
    if before:
        verify_pair(before_folder, report_folder, before, after)
    previous = {task['task']: task for task in before['tasks']} if before else {}
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})

    def panel(axis, task):
        requested = np.asarray(task['requested'])
        measured = np.asarray(task['measured'])
        deviation = np.asarray(task['measured_std'])
        curves = np.asarray([curve['measured'] for curve in task['curves']])
        old = previous.get(task['task'])
        values = [requested.ravel(), curves.ravel(), (measured-deviation).ravel(), (measured+deviation).ravel()]
        if old:
            old_measured, old_std = np.asarray(old['measured']), np.asarray(old['measured_std'])
            values += [(old_measured-old_std).ravel(), (old_measured+old_std).ravel()]
        combined = np.concatenate(values)
        low, high = min(0, combined.min()), combined.max()
        margin = max(.02, (high-low)*.07)
        axis.set_xlim(low-margin, high+margin)
        axis.set_ylim(low-margin, high+margin)
        axis.plot([low-margin, high+margin], [low-margin, high+margin], '--', color='#384654', lw=1.3, zorder=1)
        if requested.shape[1] == 3:
            for component, (color, label) in enumerate(zip(['#2166ac','#23946c','#d67b16'], ['X (forward)','Y (left)','Z (up)'])):
                for curve in curves:
                    axis.scatter(requested[:,component], curve[:,component], color=color, s=12, alpha=.25)
                if old:
                    axis.scatter(requested[:,component], old_measured[:,component], color='#ce7131', marker='x', s=55, zorder=3)
                axis.errorbar(requested[:,component], measured[:,component], yerr=deviation[:,component],
                              fmt='o', color=color, capsize=4, ms=5, label=label, zorder=4)
            axis.legend(loc='upper left', frameon=False, fontsize=8)
            axis.set_title('reach | XYZ projections (same targets)', fontsize=12, pad=10)
        else:
            order = np.argsort(requested[:,0])
            targets = requested[order,0]
            for curve in curves:
                axis.plot(targets, curve[order,0], '-o', color='#9ba8b6', alpha=.35, ms=3, lw=.7, zorder=2)
            if old:
                axis.errorbar(targets, old_measured[order,0], yerr=old_std[order,0], fmt='s--',
                              color='#ce7131', capsize=3, ms=5, lw=1.4, alpha=.85, zorder=3)
            axis.errorbar(targets, measured[order,0], yerr=deviation[order,0], fmt='o-',
                          color='#1668b0', capsize=4, ms=6, lw=2, zorder=4)
            mae = f"{old['mae'][0]:.3f} → {task['mae'][0]:.3f}" if old else f"{task['mae'][0]:.3f}"
            axis.set_title(f"{task['task']} | MAE {mae} {task['unit']}", fontsize=12, pad=10)
        axis.set_xlabel(f"Requested command ({task['unit']})")
        axis.set_ylabel(f"Measured motion ({task['unit']})")
        axis.set_aspect('equal', adjustable='box')
        axis.xaxis.set_major_locator(MaxNLocator(5))
        axis.yaxis.set_major_locator(MaxNLocator(5))
        axis.grid(alpha=.2)

    layouts = [('command_response_overview', after['tasks'], 3, 3, (13, 12))]
    affected = [task for task in after['tasks'] if task['task'] in ('jump','squat','strike','kick')]
    if before and len(affected) == 4:
        layouts.append(('affected_response_comparison', affected, 2, 2, (10, 10)))
    for filename, tasks, rows, columns, size in layouts:
        figure, axes = plt.subplots(rows, columns, figsize=size, squeeze=False)
        for axis, task in zip(axes.flat, tasks):
            panel(axis, task)
        for axis in axes.flat[len(tasks):]:
            axis.set_visible(False)
        figure.suptitle('Kimodo: paired command response' if before else 'Kimodo: command response', fontsize=18, y=.99)
        figure.text(.5,.956, f"{after['scored_samples']} samples per run · same commands / prompts / seeds · raw output", ha='center', color='#53616d')
        legend = [Line2D([0],[0],ls='--',color='#384654',label='Ideal response'),
                  Line2D([0],[0],marker='o',color='#1668b0',lw=2,label='Anchored v3: mean ± SD' if before else 'Mean ± SD')]
        if before:
            legend.append(Line2D([0],[0],marker='s',ls='--',color='#ce7131',label='Single-event v2: mean ± SD'))
        figure.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5,.938), ncol=3, frameon=False, fontsize=9)
        figure.tight_layout(rect=[.005,.03,.995,.907], h_pad=2., w_pad=1.8)
        command_counts = sorted({len(task['requested']) for task in after['tasks']})
        coverage = ('3 commands per task' if command_counts == [3]
                    else ' / '.join(str(count) for count in command_counts) + ' command points per task (see task reports)')
        footer = coverage + '; gray curves = individual prompt/seed results.'
        if before:
            footer += ' Added anchors change conditioning.'
        figure.text(.5,.012,footer,
                    ha='center', fontsize=8, color='#687580')
        figure.savefig(output/(filename+'.png'), dpi=160, facecolor='white')
        figure.savefig(output/(filename+'.pdf'), facecolor='white')
        plt.close(figure)
    if before:
        with (output/'comparison.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['task','unit','before_mae','after_mae','before_nmae','after_nmae'])
            for task in after['tasks']:
                old = previous[task['task']]
                writer.writerow([task['task'],task['unit'],json.dumps(old['mae']),json.dumps(task['mae']),old['normalized_mae'],task['normalized_mae']])
    return after


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, required=True)
    parser.add_argument('--before', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plot_responses(args.reports, args.output, args.before)
    print(args.output)


if __name__ == '__main__':
    main()
