"""Explicit offline generation entrypoint. --preflight never imports torch/Kimodo."""

import argparse
from contextlib import contextmanager
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil

import numpy as np

from .conditions import condition_spec, describe_skeleton, expected_mask, native_constraints
from .protocol import (MODEL, ROOT, SETTINGS, TASKS, archive_path,
                       file_sha256, identity, load_protocol, read_json,
                       read_sample, sample_digest, samples, scoring_positions, task_by_name,
                       verify_sources, write_json, write_npz)

BASE = "text_encoders/meta-llama/Meta-Llama-3-8B-Instruct"
MNTP = "text_encoders/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
SUPERVISED = "text_encoders/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"


def asset_paths(root):
    root = Path(root)
    paths = [f"{MODEL}/config.yaml", f"{MODEL}/model.safetensors", f"{BASE}/config.json", f"{BASE}/model.safetensors.index.json"]
    paths += [f"{MODEL}/stats/motion/{group}/{stat}.npy" for group in ("body", "global_root", "local_root") for stat in ("mean", "std")]
    paths += [f"{BASE}/model-{i:05d}-of-00004.safetensors" for i in range(1, 5)]
    paths += [f"{MNTP}/{f}" for f in ("adapter_model.safetensors", "adapter_config.json", "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")]
    paths += [f"{SUPERVISED}/{f}" for f in ("adapter_model.safetensors", "adapter_config.json")]
    for folder in (MNTP, SUPERVISED):
        if (root / folder / "llm2vec_config.json").exists():
            paths.append(folder + "/llm2vec_config.json")
    return sorted(paths)


def preflight(protocol_path, assets, verify_assets=False):
    p = load_protocol(protocol_path)
    paths = asset_paths(assets)
    missing = [rel for rel in paths if not (assets / rel).is_file()]
    if missing:
        raise FileNotFoundError("Missing assets: " + ", ".join(missing))
    adapter = read_json(assets / MNTP / "adapter_config.json")
    if Path(adapter["base_model_name_or_path"]).resolve() != (assets / BASE).resolve():
        raise ValueError("MNTP must point to the local Llama base model")
    if read_json(assets / MNTP / "config.json").get("_name_or_path") != "meta-llama/Meta-Llama-3-8B-Instruct":
        raise ValueError("LLM2Vec prompt-format identity changed")
    index = read_json(assets / BASE / "model.safetensors.index.json")
    if set(index["weight_map"].values()) != {f"model-{i:05d}-of-00004.safetensors" for i in range(1, 5)}:
        raise ValueError("Unexpected Llama shard layout")
    report = {"protocol_id": p["protocol_id"], "expected_samples": p["expected_samples"],
              "assets_present": len(paths), "model_loaded": False, "gpu_used": False,
              "note": "File readiness only; generation has not been tested."}
    if verify_assets:
        report["asset_sha256"] = {rel: file_sha256(assets / rel) for rel in paths}
        manifest = assets / "asset_manifest.json"
        if manifest.exists():
            for name, digest in read_json(manifest)["official_weight_sha256"].items():
                if report["asset_sha256"][f"{BASE}/{name}"] != digest:
                    raise ValueError("Llama checkpoint checksum mismatch")
    return p, report


def code_fingerprint():
    paths = list((ROOT / "kimodo").rglob("*.py"))
    paths += [ROOT / "kimodo_benchmark" / f"{name}.py" for name in ("protocol", "conditions", "anchored_conditions", "generate")]
    for folder in ("somaskel30", "somaskel77"):
        paths += [p for p in (ROOT / "kimodo/assets/skeletons" / folder).rglob("*") if p.is_file()]
    return {str(p.relative_to(ROOT)): file_sha256(p) for p in sorted(paths)}


def runtime_identity(device, text_device):
    packages = {}
    for name in ("torch", "numpy", "transformers", "peft", "huggingface-hub", "safetensors", "hydra-core", "einops"):
        packages[name] = importlib.metadata.version(name)
    return {"python": platform.python_version(), "packages": packages, "device": device,
            "text_encoder_device": text_device, "platform": platform.platform()}


class CachedEncoder:
    """Cache exactly the native embeddings, including their dtype and lengths."""

    def __init__(self, encoder):
        self.encoder = encoder
        self.cache = {}

    def __call__(self, texts):
        import torch
        if isinstance(texts, str):
            texts = [texts]
        for text in texts:
            if text not in self.cache:
                tensor, lengths = self.encoder([text])
                self.cache[text] = (tensor.detach().clone(), list(lengths))
        return torch.cat([self.cache[t][0] for t in texts]), [self.cache[t][1][0] for t in texts]


class NativeEngine:
    def __init__(self, assets, device, text_device):
        # Set all switches before transformers/HF can read environment constants.
        os.environ.update(CHECKPOINT_DIR=str(assets), TEXT_ENCODERS_DIR=str(assets / "text_encoders"),
                          TEXT_ENCODER_MODE="local", TEXT_ENCODER_DEVICE=text_device,
                          TEXT_ENCODER="llm2vec",
                          LOCAL_CACHE="True", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        import torch
        from kimodo import load_model
        from kimodo.sanitize import sanitize_text
        self.device, self.sanitize = device, sanitize_text
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        self.model = load_model(MODEL, device=device)
        if self.model.fps != 30 or self.model.motion_rep.motion_rep_dim != 369:
            raise ValueError("Unsupported model FPS/representation")
        self.geometry = describe_skeleton(self.model.skeleton, self.model.output_skeleton)
        self.model.text_encoder = CachedEncoder(self.model.text_encoder)
        self.runtime = runtime_identity(device, text_device)
        self.runtime["torch_num_threads"] = torch.get_num_threads()
        self.runtime["cuda_version"] = torch.version.cuda
        if str(device).startswith("cuda"):
            self.runtime["gpu_name"] = torch.cuda.get_device_name(device)

    def infer(self, record, spec):
        import torch
        constraints = native_constraints(spec, self.model.skeleton, self.device)
        observed, mask = self.model.motion_rep.create_conditions_from_constraints(
            constraints, spec["native_frames"], to_normalize=True, device=self.device)
        if not np.array_equal(mask.cpu().numpy(), expected_mask(spec, self.geometry)):
            raise ValueError("Native condition mask differs from adapter contract")
        # Warm the embedding BEFORE resetting the sampling RNG. Cache hits and
        # misses must not change the initial diffusion noise.
        normalized_prompt = self.sanitize(record["prompt"])
        self.model.text_encoder([normalized_prompt])
        random.seed(record["seed"])
        np.random.seed(record["seed"])
        torch.manual_seed(record["seed"])
        if str(self.device).startswith("cuda"):
            torch.cuda.manual_seed_all(record["seed"])
        result = self.model(record["prompt"], spec["native_frames"], num_denoising_steps=100,
                            multi_prompt=False, constraint_lst=constraints, cfg_type="separated",
                            cfg_weight=[2.0, 2.0], first_heading_angle=0.0,
                            post_processing=False, return_numpy=True, progress_bar=lambda x: x)
        return {"native_positions_k": np.asarray(result["posed_joints"], dtype=np.float32),
                "local_rot_mats": np.asarray(result["local_rot_mats"], dtype=np.float32),
                "root_positions": np.asarray(result["root_positions"], dtype=np.float32),
                "smooth_root_pos": np.asarray(result["smooth_root_pos"], dtype=np.float32),
                "observed_motion": observed.detach().cpu().numpy(), "motion_mask": mask.cpu().numpy(),
                "actual_prompt": np.asarray(normalized_prompt)}


@contextmanager
def output_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".writer.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another writer owns this output directory") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def prepare_run(output, protocol_path, protocol, unsigned):
    run = dict(unsigned, protocol_id=protocol["protocol_id"])
    run["run_id"] = identity(run)
    run_path = output / "run.json"
    if run_path.exists():
        if read_json(run_path) != run:
            raise ValueError("Run identity changed; use a new output directory")
        if load_protocol(output / "protocol.json")["protocol_id"] != protocol["protocol_id"]:
            raise ValueError("Run protocol changed")
    else:
        allowed = {".writer.lock", "protocol.json", "scoring_source"}
        if {p.name for p in output.iterdir()} - allowed:
            raise ValueError("Refusing to adopt an unowned nonempty output directory")
        if (output / "protocol.json").exists() and read_json(output / "protocol.json") != protocol:
            raise ValueError("Interrupted export has another protocol")
        shutil.copytree(Path(protocol_path).parent / "scoring_source", output / "scoring_source", dirs_exist_ok=True)
        verify_sources(output / "scoring_source", protocol["scoring"]["files"])
        write_json(output / "protocol.json", protocol)
        write_json(run_path, run)
    return run


def validate_condition_archive(arrays, spec, geometry):
    if str(arrays["condition_json"]) != json.dumps(spec, sort_keys=True, allow_nan=False):
        raise ValueError("Stored targets differ from frozen condition adapter")
    if not np.array_equal(arrays["motion_mask"], expected_mask(spec, geometry)):
        raise ValueError("Stored native condition mask differs")


def generate_samples(protocol, output, engine, run, selected_tasks=None, maximum_commands=None):
    completed = reused = 0
    for record in samples(protocol, selected_tasks, maximum_commands):
        task = task_by_name(protocol, record["task"])
        spec = condition_spec(task, record["command"], record["frames"], engine.geometry, protocol["adapter"])
        path = archive_path(output, record)
        if path.exists():
            arrays = read_sample(path, record, run["run_id"], engine.geometry["names77"])
            validate_condition_archive(arrays, spec, engine.geometry)
            reused += 1
            continue
        arrays = engine.infer(record, spec)
        arrays.update(
            positions_f=scoring_positions(arrays["native_positions_k"], engine.geometry["names77"], record["frames"]),
            condition_json=np.asarray(json.dumps(spec, sort_keys=True, allow_nan=False)),
            run_id=np.asarray(run["run_id"]), sample_id=np.asarray(record["sample_id"]),
            length=np.asarray(record["frames"]), fps=np.asarray(20), native_fps=np.asarray(30),
            seed=np.asarray(record["seed"]), prompt_index=np.asarray(record["prompt_index"]),
            command_index=np.asarray(record["command_index"]), prompt=np.asarray(record["prompt"]),
            command=np.asarray(record["command"], dtype=np.float32),
        )
        validate_condition_archive(arrays, spec, engine.geometry)
        if any(not np.isfinite(a).all() for a in arrays.values() if np.asarray(a).dtype.kind in "fci"):
            raise ValueError(f"Nonfinite generation: {record['sample_id']}")
        arrays["sample_sha256"] = np.asarray(sample_digest(arrays))
        write_npz(path, **arrays)
        read_sample(path, record, run["run_id"], engine.geometry["names77"])
        completed += 1
        if completed % 25 == 0:
            print(f"Generated {completed}, reused {reused}", flush=True)
    return {"generated": completed, "reused": reused}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=Path.home() / "checkpoints/kimodo")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--verify-assets", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--text-encoder-device", default="cuda:0")
    parser.add_argument("--tasks", choices=TASKS, nargs="+")
    parser.add_argument("--maximum-commands", type=int)
    args = parser.parse_args()
    if not args.preflight and args.output is None:
        parser.error("Generation requires --output; use --preflight for file-only checks")
    if args.maximum_commands is not None and args.maximum_commands < 1:
        parser.error("--maximum-commands must be positive")
    args.assets = args.assets.expanduser().resolve()
    p, report = preflight(args.protocol, args.assets, args.verify_assets or not args.preflight)
    if args.preflight:
        print(json.dumps(report, indent=2))
        return
    with output_lock(args.output):
        engine = NativeEngine(args.assets, args.device, args.text_encoder_device)
        unsigned = {"settings": SETTINGS, "files": code_fingerprint(), "assets": report["asset_sha256"],
                    "geometry": engine.geometry, "runtime": engine.runtime}
        run = prepare_run(args.output, args.protocol, p, unsigned)
        result = generate_samples(p, args.output, engine, run, args.tasks, args.maximum_commands)
        write_json(args.output / "generation_status.json", dict(result, status="requested_subset_finished", run_id=run["run_id"]))
        print(json.dumps(result))


if __name__ == "__main__":
    main()
