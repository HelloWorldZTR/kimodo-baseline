"""Freeze the agreed composite protocol; never generate or score motion."""

import argparse
import copy
from pathlib import Path
import shutil
import tempfile

from .protocol import (ADAPTER, BASE_ID, EVALUATION_ID, LIMB_ID, SETTINGS, TASKS,
                       checked_identity, file_sha256, identity, read_json,
                       safe_child, validate_protocol, verify_sources, write_json)


def build_protocol(base, limb, evaluation):
    for doc, field, expected in [(base, "protocol_id", BASE_ID), (limb, "protocol_id", LIMB_ID),
                                 (evaluation, "evaluation_id", EVALUATION_ID)]:
        if checked_identity(doc, field) != expected:
            raise ValueError(f"Not the approved frozen {field}")
    if not base["references_ready"] or not limb["references_ready"]:
        raise ValueError("Reference poses are required")
    if limb["revision"]["parent_protocol_id"] != base["protocol_id"] or evaluation["generation_protocol_id"] != limb["protocol_id"]:
        raise ValueError("Composite protocol ancestry differs")
    if tuple(t["name"] for t in base["tasks"]) != TASKS or [t["name"] for t in limb["tasks"]] != ["strike", "kick"]:
        raise ValueError("Wrong task sources")
    if base["seeds"] != limb["seeds"] or base["fps"] != limb["fps"]:
        raise ValueError("Source sampling settings differ")
    tasks = copy.deepcopy(base["tasks"])
    for t in tasks:
        if t["name"] not in ("strike", "kick"):
            continue
        revised = next(x for x in limb["tasks"] if x["name"] == t["name"])
        for key in ("commands", "prompts", "lengths", "reference", "metadata"):
            if t[key] != revised[key]:
                raise ValueError(f"Limb revision changes {key}")
        t["sampling_definition"] = copy.deepcopy(t["definition"])
        t["definition"] = copy.deepcopy(evaluation["definitions"][t["name"]])
    return {
        "schema_version": 1, "fps": 20, "seeds": base["seeds"], "expected_samples": 23400,
        "adapter": copy.deepcopy(ADAPTER), "settings": copy.deepcopy(SETTINGS), "tasks": tasks,
        "sources": {"base_protocol_id": BASE_ID, "limb_protocol_id": LIMB_ID, "limb_evaluation_id": EVALUATION_ID},
    }


def export_bundle(base_path, limb_path, evaluation_path, base_source, limb_source, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new output directory for a frozen bundle")
    base, limb, ev = (read_json(p) for p in (base_path, limb_path, evaluation_path))
    p = build_protocol(base, limb, ev)
    verify_sources(base_source, base["source"]["files"])
    verify_sources(limb_source, ev["source_files"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".kimodo-export-", dir=output.parent))
    try:
        destination = temporary / "scoring_source"
        origins = {}
        for rel in base["source"]["files"]:
            target = safe_child(destination, rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(safe_child(base_source, rel), target)
            origins[rel] = "base"
        # Only limb definitions and their helper are updated. In particular, do
        # NOT copy the newer reach task, workspace or quality rules.
        revised = ["shared_motion/tasks/strike.py", "shared_motion/tasks/kick.py",
                   "shared_motion/utils/command_frame.py", "shared_motion/utils/file.py",
                   "shared_motion/utils/__init__.py", "shared_motion/__init__.py"]
        for rel in revised:
            if rel not in ev["source_files"]:
                raise ValueError(f"Unpinned scoring dependency: {rel}")
            target = safe_child(destination, rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(safe_child(limb_source, rel), target)
            origins[rel] = "limb"
        # The frozen reach loader resolves config relative to shared_motion.
        # Supply the identical pinned YAML at that location, without editing it.
        rel = "config/data_processing/task_rules/reach.yaml"
        alias = "shared_motion/" + rel
        target = destination / alias
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(destination / rel, target)
        origins[alias] = "base_config_alias"
        hashes = {str(f.relative_to(destination)): file_sha256(f) for f in sorted(destination.rglob("*")) if f.is_file()}
        p["scoring"] = {"files": hashes, "origins": origins,
                        "policy": "original seven tasks and reach quality; initial-root-frame limb metrics"}
        for label, path in [("base_protocol", base_path), ("limb_protocol", limb_path), ("limb_evaluation", evaluation_path)]:
            p["sources"][label + "_sha256"] = file_sha256(path)
            shutil.copyfile(path, temporary / (label + ".json"))
        p["protocol_id"] = identity(p)
        validate_protocol(p)
        write_json(temporary / "protocol.json", p)
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("base-protocol", "limb-protocol", "limb-evaluation", "base-source", "limb-source", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    p = export_bundle(args.base_protocol, args.limb_protocol, args.limb_evaluation, args.base_source, args.limb_source, args.output)
    print(f"Exported {p['expected_samples']} identities to {args.output}; no generation or scoring performed.")


if __name__ == "__main__":
    main()
