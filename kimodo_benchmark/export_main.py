"""Freeze current main's task plans and real-train references without loading Kimodo.

Input is a git-archive snapshot accompanied by SOURCE.json (commit and SHA256s).
The original composite exporter remains available for historical comparisons.
"""

import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

from .protocol import (MAIN_ADAPTER, SEEDS, SETTINGS, TASKS, canonicalize,
                       file_sha256, identity, safe_child, read_json,
                       validate_protocol, verify_sources, write_json)


def reference_pose(row, skeleton, amass_roots):
    import torch
    from data_processing.pipeline.sources import read_raw, resolve_amass_source
    from data_processing.pipeline.transforms import TransformContext, apply_transforms

    roots = {"amass": str(amass_roots[0]), "amass_roots": [str(p) for p in amass_roots[1:]]}
    raw_path = resolve_amass_source(roots, row["source"]["path"])
    if file_sha256(raw_path) != row["source"]["sha256"]:
        raise ValueError("Raw reference source changed: " + str(raw_path))
    sample = apply_transforms(read_raw(raw_path), row["transforms"], TransformContext(skeleton))
    motion = np.asarray(sample["motion"][:1], dtype=np.float32).copy()
    with torch.no_grad():
        pose = skeleton(torch.from_numpy(motion)[None])[0, :, :22].cpu().numpy()
    return canonicalize(pose)[0].tolist()


def choose_reference(name, rows, workspace):
    if name == "reach":
        return next(row for row in rows if row["id"] == workspace["reference"]["episode_id"])
    median = np.median([float(np.asarray(row["expected_quantity"]).reshape(-1)[0]) for row in rows])
    return min(rows, key=lambda row: (abs(float(np.asarray(row["expected_quantity"]).reshape(-1)[0]) - median), row["id"]))


def prepare_source(source, destination):
    provenance = read_json(source / "SOURCE.json")
    if provenance["ref"] != "origin/main" or len(provenance["revision"]) != 40:
        raise ValueError("A pinned main commit is required")
    verify_sources(source, provenance["files"])
    for relative in provenance["files"]:
        target = safe_child(destination, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(safe_child(source, relative), target)
    # Upstream resolves config from shared_motion, rather than the repo root.
    # Supply byte-identical aliases; keep every main source file unmodified.
    aliases = list((destination / "config/task/parameters").glob("*.yaml"))
    aliases.append(destination / "config/data_processing/task_rules/reach.yaml")
    alias_origins = {}
    for path in aliases:
        relative = path.relative_to(destination)
        target = destination / "shared_motion" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        alias_origins[str(target.relative_to(destination))] = str(relative)
    return provenance, alias_origins


def export_main(source, skeleton_path, amass_roots, output):
    import torch
    from omegaconf import OmegaConf

    if not amass_roots:
        raise ValueError("Raw AMASS roots are required for current reference poses")
    if output.exists():
        raise FileExistsError("Use a new frozen output directory")
    if any(name.split('.')[0] in ("shared_motion", "data_processing", "src") for name in sys.modules):
        raise RuntimeError("Export in a fresh interpreter to avoid mixed upstream versions")
    torch.set_num_threads(2)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.main-export-', dir=output.parent)).resolve()
    try:
        root = temporary / "scoring_source"
        provenance, aliases = prepare_source(source, root)
        sys.path.insert(0, str(root))
        from shared_motion.tasks import get_task
        from shared_motion.tasks.evaluation import references_from_manifest
        from shared_motion.tasks.reach_evaluation import load_workspace
        from shared_motion.utils.geometry import Skeleton

        data_config = OmegaConf.load(root / "config/data/real.yaml")
        episode_root = safe_child(root, data_config.episodes)
        release_root = episode_root.parent
        if file_sha256(release_root / "manifest.json") != data_config.manifest_sha256:
            raise ValueError("main's real data manifest hash differs")
        verify_sources(release_root, read_json(release_root / "manifest.json")["files"])
        settings = OmegaConf.load(root / "config/task/parameters/reach.yaml")
        settings.workspace = str(safe_child(root, settings.workspace))
        workspace = load_workspace(settings.workspace, settings.workspace_sha256)
        if file_sha256(skeleton_path) != workspace["skeleton_sha256"]:
            raise ValueError("Reference skeleton differs from reach certificate")
        skeleton = Skeleton(skeleton_path).cpu()
        grid = OmegaConf.load(root / "config/benchmark/dense.yaml")
        evaluate = OmegaConf.load(root / "config/evaluate.yaml")
        if tuple(evaluate.seeds) != SEEDS or (grid.bins, grid.points_per_bin) != (10, 3):
            raise ValueError("Sampling panel changed; review adapter support")
        protocol = dict(schema_version=2, fps=20, seeds=list(evaluate.seeds),
                        adapter=copy.deepcopy(MAIN_ADAPTER), settings=copy.deepcopy(SETTINGS), tasks=[],
                        sources=dict(main_revision=provenance["revision"], repository=provenance["repository"],
                                     release=release_root.name, workspace_sha256=settings.workspace_sha256,
                                     skeleton_sha256=file_sha256(skeleton_path),
                                     snapshot_manifest_sha256=file_sha256(source / "SOURCE.json"),
                                     source_files=provenance["files"], config_aliases=aliases,
                                     reference_selection="current real-train median (ID tie break); reach workspace reference; first frame FK",
                                     exporter_files={"export_main.py": file_sha256(Path(__file__)),
                                                     "protocol.py": file_sha256(Path(__file__).with_name('protocol.py'))},
                                     export_packages={name: importlib.metadata.version(name) for name in ('torch','numpy','omegaconf')}))
        for name in TASKS:
            task = get_task(name)
            references = references_from_manifest(episode_root, name)
            plan = task.evaluation_plans("benchmark", evaluate.points, settings if name == 'reach' else None,
                                         references, command_grid=grid)[0]
            rows = [json.loads(line) for line in (episode_root / (name + '.jsonl')).read_text().splitlines() if line]
            real_train = [row for row in rows if row['split'] == 'train' and row['source']['dataset'] == 'amass']
            selected = choose_reference(name, real_train, workspace)
            reference = dict(episode_id=selected['id'], source=selected['source'], transforms=selected['transforms'],
                             source_split='train', expected_quantity=selected['expected_quantity'], pose_frame_in_crop=0,
                             positions_f=reference_pose(selected, skeleton, amass_roots))
            protocol['tasks'].append(dict(name=name, definition=task.definition(), prompts=list(task.prompts),
                                          commands=plan.commands.float().tolist(), lengths=plan.lengths,
                                          metadata=plan.metadata, reference=reference))
            print(f"Exported {name}: {len(plan.commands)} commands; bounds {task.bounds}", flush=True)
        protocol['expected_samples'] = sum(len(task['commands']) * 9 for task in protocol['tasks'])
        protocol['sources']['task_command_counts'] = {task['name']: len(task['commands']) for task in protocol['tasks']}
        hashes = {str(path.relative_to(root)): file_sha256(path) for path in sorted(root.rglob('*'))
                  if path.is_file() and '__pycache__' not in path.parts}
        protocol['scoring'] = dict(files=hashes, policy='all nine definitions, command grids and reach diagnostics from pinned current main')
        protocol = json.loads(json.dumps(protocol))
        protocol['protocol_id'] = identity(protocol)
        validate_protocol(protocol)
        shutil.copyfile(source / 'SOURCE.json', temporary / 'SOURCE.json')
        write_json(temporary / 'protocol.json', protocol)
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-source', type=Path, required=True)
    parser.add_argument('--skeleton', type=Path, required=True)
    parser.add_argument('--amass-root', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = export_main(args.main_source, args.skeleton, args.amass_root, args.output)
    print(f"Frozen main {protocol['sources']['main_revision']}: {protocol['expected_samples']} samples, no model loaded")


if __name__ == '__main__':
    main()
