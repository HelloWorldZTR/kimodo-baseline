"""CPU scoring against the verified composite task snapshot."""

import argparse
import csv
import io
import importlib
import importlib.machinery
import importlib.metadata
import json
from pathlib import Path
import sys
import platform
import types

import numpy as np

from .conditions import condition_spec, primary_error
from .generate import output_lock, validate_condition_archive
from .protocol import (ROOT, SETTINGS, archive_path, atomic_bytes, canonicalize,
                       checked_identity, file_sha256, load_protocol, read_json,
                       read_sample, samples, task_by_name, verify_sources,
                       write_json, write_npz)


def connect_scoring_source(root, protocol):
    root = Path(root).resolve()
    verify_sources(root, protocol["scoring"]["files"])
    for name, module in tuple(sys.modules.items()):
        if name.split(".")[0] in ("shared_motion", "data_processing", "src"):
            locations = list(getattr(module, "__path__", []))
            if getattr(module, "__file__", None):
                locations.append(module.__file__)
            if any(not Path(p).resolve().is_relative_to(root) for p in locations):
                raise RuntimeError("Another task implementation is already imported; score in a fresh process")
    sys.path.insert(0, str(root))
    # The upstream umbrella registry imports four out-of-scope tasks and their
    # unfrozen configs. Load only the nine pinned task modules, unmodified.
    if "shared_motion.tasks" not in sys.modules:
        package = types.ModuleType("shared_motion.tasks")
        package.__path__ = [str(root / "shared_motion/tasks")]
        package.__spec__ = importlib.machinery.ModuleSpec(package.__name__, None, is_package=True)
        sys.modules[package.__name__] = package
    for task in protocol["tasks"]:
        actual = json.loads(json.dumps(get_definition(task["name"]).definition()))
        if actual != task["definition"]:
            raise ValueError("Frozen scoring definition mismatch: " + task["name"])


def get_definition(name):
    return importlib.import_module("shared_motion.tasks." + name).TASK


def measure_sample(task, positions_f):
    import torch

    if positions_f.shape != (task["definition"]["evaluation_frames"], 22, 3):
        raise ValueError("Measurement must use exactly the scored prefix")
    tensor = torch.from_numpy(canonicalize(positions_f))[None]
    definition = get_definition(task["name"])
    with torch.no_grad():
        measured = definition.measure_positions(tensor).reshape(-1).cpu().numpy()
        quality = ({} if definition.diagnostics is None else
                   {name: float(value[0]) for name, value in definition.diagnostics(tensor).items()})
    if measured.shape != (len(task["definition"]["bounds"]),) or not np.isfinite(measured).all():
        raise ValueError("Invalid task measurement")
    if not all(np.isfinite(v) for v in quality.values()):
        raise ValueError("Invalid quality diagnostic")
    return measured, quality


def skating_ratio(positions_f):
    """OmniControl's toe skating statistic at 20 Hz, in F coordinates.

    Either toe: consecutive heights < 5 cm, planar speed > 0.5 m/s,
    and five-frame constant-padded mean speed > 0.5 m/s.
    """
    from scipy.ndimage import uniform_filter1d
    feet = positions_f[:, [10, 11]]
    speed = np.linalg.norm(np.diff(feet[..., :2], axis=0), axis=-1) * 20
    mean_speed = uniform_filter1d(speed, size=5, axis=0, mode="constant", origin=0)
    contact = (feet[:-1, :, 2] < .05) & (feet[1:, :, 2] < .05)
    return float((contact & (speed > .5) & (mean_speed > .5)).any(axis=1).mean())


def summarize(task, entries, seeds):
    """Aggregate a balanced command × prompt × seed panel, never average first."""
    command_indices = sorted({entry["command_index"] for entry in entries})
    curve_keys = [(prompt_index, seed) for prompt_index in range(3) for seed in seeds]
    by_identity = {
        (entry["prompt_index"], entry["seed"], entry["command_index"]): entry
        for entry in entries
    }
    if len(by_identity) != len(entries):
        raise ValueError("Duplicate measurements")
    if len(entries) != len(command_indices) * len(curve_keys):
        raise ValueError(
            "Incomplete prompt/seed panel for {}; finish generation before scoring".format(
                task["name"]
            )
        )
    ordered = [
        [by_identity[(prompt_index, seed, index)] for index in command_indices]
        for prompt_index, seed in curve_keys
    ]
    dimensions = len(task["definition"]["bounds"])
    requested = np.asarray(task["commands"], dtype=np.float32)[
        command_indices, :dimensions
    ]
    measured = np.asarray(
        [[entry["measured"] for entry in curve] for curve in ordered], dtype=np.float32
    )
    spans = np.asarray(
        [upper - lower for lower, upper in task["definition"]["bounds"]],
        dtype=np.float32,
    )
    errors = np.abs(measured - requested[None])
    result = dict(
        task=task["name"],
        unit=task["definition"]["unit"],
        dimensions=dimensions,
        command_indices=command_indices,
        commands=len(command_indices),
        samples=len(entries),
        frames_per_command=[task["lengths"][index] for index in command_indices],
        requested=requested.tolist(),
        measured=measured.mean(0).tolist(),
        measured_std=measured.std(0, ddof=0).tolist(),
        mae=errors.mean((0, 1)).tolist(),
        normalized_mae=float((errors / spans).mean()),
        per_command_mae=errors.mean(0).tolist(),
        curves=[],
        native_primary_error_m=float(
            np.mean([entry["native_primary_error_m"] for entry in entries])
        ),
        skating_ratio=float(np.mean([entry["skating_ratio"] for entry in entries])),
    )
    for curve_index, (prompt_index, seed) in enumerate(curve_keys):
        result["curves"].append(
            dict(
                prompt_index=prompt_index,
                prompt=task["prompts"][prompt_index],
                seed=seed,
                measured=measured[curve_index].tolist(),
                mae=errors[curve_index].mean(0).tolist(),
                normalized_mae=float((errors[curve_index] / spans).mean()),
            )
        )
    quality_names = sorted(entries[0]["quality"])
    if any(sorted(e["quality"]) != quality_names for e in entries):
        raise ValueError("Inconsistent task diagnostics")
    if quality_names:
        result["quality"] = {
            name: float(np.mean([entry["quality"][name] for entry in entries]))
            for name in quality_names
        }
        for curve, curve_entries in zip(result["curves"], ordered):
            curve["quality"] = {
                name: float(
                    np.mean([entry["quality"][name] for entry in curve_entries])
                )
                for name in quality_names
            }
    metadata = task["metadata"]
    if metadata.get("position_tolerances_m"):
        difference = measured - requested[None]
        distances = np.linalg.norm(difference, axis=-1)
        quality = np.asarray(
            [[entry["quality"]["valid"] for entry in curve] for curve in ordered],
            dtype=bool,
        )
        groups = np.asarray(metadata["evaluation_groups"])[command_indices]
        nearest = np.asarray(metadata["nearest_real_train_m"])[command_indices]
        subsets = {"all": np.ones(len(command_indices), dtype=bool)}
        subsets.update({name: groups == name for name in sorted(set(groups))})
        subsets.update(real_supported=nearest <= 0.05, real_gap=nearest > 0.05)
        result["position_metrics"] = {}
        for name, selected in subsets.items():
            if not selected.any():
                continue
            residual = difference[:, selected]
            metrics = dict(
                commands=int(selected.sum()),
                samples=int(distances[:, selected].size),
                rmse_m=float(np.sqrt(np.mean(distances[:, selected] ** 2))),
                bias_rms_m=float(
                    np.sqrt(np.mean(np.sum(residual.mean(0) ** 2, axis=-1)))
                ),
                std_rms_m=float(
                    np.sqrt(np.mean(np.sum(residual.var(0, ddof=0), axis=-1)))
                ),
                quality_pass_rate=float(quality[:, selected].mean()),
            )
            for tolerance in metadata["position_tolerances_m"]:
                success = distances[:, selected] <= tolerance
                metrics["success_{:g}m".format(tolerance)] = float(success.mean())
                metrics["joint_success_{:g}m".format(tolerance)] = float(
                    (success & quality[:, selected]).mean()
                )
            result["position_metrics"][name] = metrics
        result["evaluation_groups"] = groups.tolist()
        result["nearest_real_train_m"] = nearest.tolist()
    return result, measured

def score_run(protocol, run, generated_root, allow_partial=False):
    records = list(samples(protocol))
    present = [r for r in records if archive_path(generated_root, r).is_file()]
    if not allow_partial and len(present) != len(records):
        raise ValueError(f"Missing {len(records) - len(present)} samples; partial scoring requires --allow-partial")
    if not present:
        raise ValueError("No generated samples found")
    # Fail incomplete panels before doing measurements.
    for task in protocol["tasks"]:
        chosen = [r for r in present if r["task"] == task["name"]]
        if len(chosen) != 9 * len({r["command_index"] for r in chosen}):
            raise ValueError("Incomplete prompt/seed panel: " + task["name"])
    entries = []
    geometry = run["geometry"]
    for index, record in enumerate(present):
        task = task_by_name(protocol, record["task"])
        arrays = read_sample(archive_path(generated_root, record), record, run["run_id"], geometry["names77"])
        spec = condition_spec(task, record["command"], record["frames"], geometry, protocol["adapter"])
        validate_condition_archive(arrays, spec, geometry)
        measured, quality = measure_sample(task, arrays["positions_f"])
        entries.append(dict(record, measured=measured.tolist(), quality=quality,
                            native_primary_error_m=primary_error(spec, arrays["native_positions_k"], arrays["smooth_root_pos"], geometry),
                            skating_ratio=skating_ratio(arrays["positions_f"])))
        if (index + 1) % 500 == 0:
            print(f"Scored {index + 1}/{len(present)}", flush=True)
    results, panels = [], {}
    for task in protocol["tasks"]:
        selected = [entry for entry in entries if entry["task"] == task["name"]]
        if selected:
            result, panels[task["name"]] = summarize(task, selected, protocol["seeds"])
            results.append(result)
    complete = len(present) == len(records)
    result = dict(protocol_id=protocol["protocol_id"], run_id=run["run_id"],
                  complete=complete, expected_samples=len(records), scored_samples=len(entries),
                  macro_normalized_mae=float(np.mean([t["normalized_mae"] for t in results])) if complete else None,
                  status="full_benchmark" if complete else "partial_smoke_only_not_comparable", tasks=results)
    return result, entries, panels


def write_csv(path, fieldnames, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, stream.getvalue().encode())

def save_reports(output, protocol, run, result, entries, panels, generated_root):
    from .reports import response_plot, motion_gallery

    if (output / "run.json").exists():
        if read_json(output / "run.json") != run:
            raise ValueError("Report directory belongs to another run")
    elif output.exists() and {p.name for p in output.iterdir()} - {".writer.lock"}:
        raise ValueError("Report directory is nonempty and has no run identity")
    result["scoring_files"] = {
        "kimodo_benchmark/" + name + ".py": file_sha256(ROOT / "kimodo_benchmark" / (name + ".py"))
        for name in ("score", "reports", "protocol", "conditions", "anchored_conditions", "generate")
    }
    result["scoring_runtime"] = {
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ("torch", "numpy", "scipy", "omegaconf", "matplotlib")},
    }
    write_json(output / "protocol.json", protocol)
    write_json(output / "run.json", run)
    write_json(output / "results.json", result)
    write_csv(
        output / "summary.csv",
        [
            "task",
            "samples",
            "unit",
            "mae",
            "normalized_mae",
            "native_primary_error_m",
            "skating_ratio",
        ],
        [
            {
                name: task[name]
                for name in (
                    "task",
                    "samples",
                    "unit",
                    "mae",
                    "normalized_mae",
                    "native_primary_error_m",
                    "skating_ratio",
                )
            }
            for task in result["tasks"]
        ],
    )
    rows = []
    for entry in entries:
        row = {
            name: entry[name]
            for name in (
                "sample_id",
                "task",
                "command_index",
                "prompt_index",
                "seed",
                "frames",
                "native_primary_error_m",
                "skating_ratio",
            )
        }
        for component in range(3):
            row["requested_" + str(component)] = entry["command"][component]
            row["measured_" + str(component)] = (
                entry["measured"][component]
                if component < len(entry["measured"])
                else ""
            )
        row["quality"] = json.dumps(entry["quality"], sort_keys=True)
        rows.append(row)
    write_csv(output / "samples.csv", list(rows[0]), rows)
    galleries = []
    for task_result in result["tasks"]:
        task = task_by_name(protocol, task_result["task"])
        name = task["name"]
        write_json(output / (name + ".json"), task_result)
        write_npz(
            output / (name + ".npz"),
            measured=panels[name],
            requested=np.asarray(task_result["requested"], dtype=np.float32),
            command_indices=np.asarray(task_result["command_indices"]),
            prompt_indices=np.repeat(np.arange(3), len(protocol["seeds"])),
            seeds=np.tile(protocol["seeds"], 3),
        )
        response_plot(output / (name + "_response.png"), task, task_result)
        motion_gallery(
            output / (name + "_motions.html"),
            protocol,
            run,
            task,
            task_result,
            generated_root,
        )
        galleries.append(
            "- [{} motions]({}_motions.html) · [response]({}_response.png)".format(
                name, name, name
            )
        )
    text = (
        "# Kimodo raw command benchmark\n\nStatus: **{}**. Samples: {}/{}.\n\n".format(
            result["status"], result["scored_samples"], result["expected_samples"]
        )
    )
    if result["complete"]:
        text += "Task-macro normalized MAE: {:.6f}.\n\n".format(
            result["macro_normalized_mae"]
        )
    text += "Command error, native primary-target error and skating are separate metrics. "
    text += "Scalar metrics alone do not certify action semantics. Reach certificates apply to reference endpoints only.\n\n"
    text += "Native DDIM: 100 steps; separated CFG [2, 2]; raw output, no postprocessing. Native 30 Hz; scoring at exact 20 Hz timestamps; seeds: 1928/1938/1948. "
    text += "Gallery curves use fixed low/median/high command choices (reach: shoulder radius); no best-seed selection.\n\n"
    text += "\n".join(galleries) + "\n\nProtocol: `{}`\n\nRun: `{}`\n".format(
        protocol["protocol_id"], run["run_id"]
    )
    atomic_bytes(output / "README.md", text.encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.generated.resolve() == args.output.resolve():
        parser.error("Reports must be separate from generated samples")
    protocol = load_protocol(args.generated / "protocol.json")
    run = read_json(args.generated / "run.json")
    checked_identity(run, "run_id")
    if run["protocol_id"] != protocol["protocol_id"] or run["settings"] != SETTINGS:
        raise ValueError("Run provenance mismatch")
    verify_sources(ROOT, run["files"])
    connect_scoring_source(args.generated / "scoring_source", protocol)
    # Exclusive lock prevents generation from changing the panel while scoring.
    with output_lock(args.generated), output_lock(args.output):
        result, entries, panels = score_run(protocol, run, args.generated, args.allow_partial)
        save_reports(args.output, protocol, run, result, entries, panels, args.generated)
    print(f"{result['status']}: {result['scored_samples']} samples -> {args.output}")


if __name__ == "__main__":
    main()
