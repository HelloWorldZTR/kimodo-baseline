"""CPU-only protocol, coordinate conversion, identities and atomic artifacts."""

import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("walk", "back_walk", "sidestep", "jog", "reach", "jump", "squat", "strike", "kick")
LENGTHS = dict(zip(TASKS, (60, 60, 30, 50, 49, 66, 82, 40, 60)))
KEYFRAMES = {"strike": 30, "kick": 45, "reach": 66, "jump": 48, "squat": 61}
SEEDS = (1928, 1938, 1948)
FPS = 20
NATIVE_FPS = 30
MODEL = "Kimodo-SOMA-RP-v1.2"
VERSION = "kimodo_command9_single_keyframe_raw_v1"
WORKSPACE = "8df745f08b937ecff8e9abb85ad6e79af3a544c0bd9a5b23613d12d86b0ad17d"
BASE_ID = "cccee66b326263fcb8ca73fe45513e06c769bd18dab8d727a659e75e643b5ea8"
LIMB_ID = "fb2d7e81fca272031b00f41c9df75ff5ef5f81598eab0de87e2c74ee6c76a9b6"
EVALUATION_ID = "d3af3800a0907d81d8411666422c31a31c8637e392c418f9b7d2ad81fd0fce7f"
JOINTS_22 = (
    "Hips", "LeftLeg", "RightLeg", "Spine1", "LeftShin", "RightShin", "Spine2",
    "LeftFoot", "RightFoot", "Chest", "LeftToeBase", "RightToeBase", "Neck1",
    "LeftShoulder", "RightShoulder", "Head", "LeftArm", "RightArm",
    "LeftForeArm", "RightForeArm", "LeftHand", "RightHand",
)
ADAPTER = {
    "version": VERSION, "score_fps": FPS, "native_fps": NATIVE_FPS,
    "keyframes": KEYFRAMES, "postprocessing": False,
    "coordinate_mapping": "K=(F.y,F.z,F.x); F=forward,left,up; metres",
    "root_paths": "dense XZ only; initial heading zero; height and subsequent heading free",
    "event_root": "XZ zero, reference height, heading zero, only at the event frame",
    "rotations": "shortest alignment of native distal bone to reference forearm/foot; Hips identity",
    "kick_target": "reference forefoot + forward command; absolute forefoot height 1m",
    "strike_target": "reference shoulder + forward command",
    "reach_target": "command XYZ plus initial reference pelvis height",
    "sidestep_endpoint": "last two native frames at final distance",
    "unreachable": "retain targets and all samples; no clipping or root relocation",
}
SETTINGS = {
    "model": MODEL, "diffusion_steps": 100, "cfg_type": "separated",
    "cfg_weight": [2.0, 2.0], "post_processing": False, "multi_prompt": False,
    "first_heading_angle": 0.0, "batch_size": 1, "text_encoder_dtype": "bfloat16",
    "noise": "reset seed immediately before each independent native call",
}
MAIN_ADAPTER = dict(ADAPTER, version="kimodo_command9_single_keyframe_raw_main_v2")
ANCHORED_ADAPTER = dict(
    MAIN_ADAPTER,
    version="kimodo_command9_anchored_delta_raw_v3",
    strike_target="reference wrist forward coordinate + command; reference shoulder lateral coordinate and height",
    kick_target="reference ankle forward coordinate + command; fixed lateral coordinate; forefoot height 1m",
    event_root="same reference root height and heading at anchors; jump/squat target changes root height",
    anchors="strike: ready at 0 and 15, retract at 45 and final timestamp support; kick/jump: initial and final timestamp support; squat: initial and final five scoring frames",
    squat_target="constant reference height minus command over five scoring timestamps centered nearest native frame 61",
    unchanged_tasks=["walk", "back_walk", "sidestep", "jog", "reach"],
)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def checked_identity(value, field):
    unsigned = dict(value)
    supplied = unsigned.pop(field, None)
    if supplied != identity(unsigned):
        raise ValueError(f"{field} content hash mismatch")
    return supplied


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def write_npz(path, **arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    atomic_bytes(path, buffer.getvalue())


def sample_digest(arrays):
    h = hashlib.sha256()
    for key in sorted(set(arrays) - {"sample_sha256"}):
        value = np.asarray(arrays[key])
        h.update(json.dumps([key, value.dtype.str, value.shape]).encode())
        h.update(value.tobytes(order="C"))
    return h.hexdigest()


def safe_child(root, relative):
    root = Path(root).resolve()
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Path escapes source root: {relative}")
    return path


def verify_sources(root, hashes):
    for name, expected in hashes.items():
        if file_sha256(safe_child(root, name)) != expected:
            raise ValueError(f"Frozen source changed: {name}")


def native_frames(frames):
    return ((frames - 1) * 3 + 1) // 2 + 1


def f_to_k(positions):
    return np.asarray(positions)[..., [1, 2, 0]]


def k_to_f(positions):
    return np.asarray(positions)[..., [2, 0, 1]]


def canonicalize(positions):
    p = np.asarray(positions, dtype=np.float32).copy()
    lateral = p[0, 1, :2] - p[0, 2, :2]
    if np.linalg.norm(lateral) < 1e-8:
        raise ValueError("Degenerate initial hips")
    angle = np.arctan2(lateral[1], lateral[0]) - np.pi / 2
    rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]], dtype=np.float32)
    p[..., :2] -= p[0, 0, :2].copy()
    p[..., :2] = p[..., :2] @ rotation.T
    return p


def scoring_positions(native, joint_names, frames):
    native = np.asarray(native)
    if native.shape != (native_frames(frames), len(joint_names), 3) or not np.isfinite(native).all():
        raise ValueError("Invalid native positions")
    if len(set(joint_names)) != len(joint_names):
        raise ValueError("Duplicate native joint names")
    indices = [joint_names.index(name) for name in JOINTS_22]
    selected = k_to_f(native[:, indices])
    # Exact 30/20 time ratio; do not stretch endpoints with linspace.
    u = np.arange(frames, dtype=np.float64) * 1.5
    lo = u.astype(int)
    hi = np.ceil(u).astype(int)
    weight = (u - lo)[:, None, None]
    return ((1 - weight) * selected[lo] + weight * selected[hi]).astype(np.float32)


def task_by_name(protocol, name):
    return next(t for t in protocol["tasks"] if t["name"] == name)


def samples(protocol, selected_tasks=None, maximum_commands=None):
    if selected_tasks and not set(selected_tasks) <= set(TASKS):
        raise ValueError("Unknown task selection")
    if maximum_commands is not None and maximum_commands < 1:
        raise ValueError("maximum_commands must be positive")
    for task in protocol["tasks"]:
        name = task["name"]
        if selected_tasks and name not in selected_tasks:
            continue
        indices = np.arange(len(task["commands"]))
        if maximum_commands is not None:
            indices = np.linspace(0, len(indices) - 1, min(len(indices), maximum_commands)).round().astype(int)
        for prompt_index, prompt in enumerate(task["prompts"]):
            for seed in protocol["seeds"]:
                for index in indices.tolist():
                    yield {
                        "sample_id": f"{name}__c{index:04d}__p{prompt_index}__s{seed}",
                        "task": name, "command_index": index, "command": task["commands"][index],
                        "frames": task["lengths"][index], "prompt_index": prompt_index,
                        "prompt": prompt, "seed": seed,
                    }


def validate_protocol(p):
    checked_identity(p, "protocol_id")
    schema = p["schema_version"]
    expected_adapter = {1: ADAPTER, 2: MAIN_ADAPTER, 3: ANCHORED_ADAPTER}.get(schema)
    if schema not in (1, 2, 3) or p["adapter"] != expected_adapter or p["settings"] != SETTINGS:
        raise ValueError("Unsupported Kimodo benchmark configuration")
    count = sum(len(t["commands"]) * len(t["prompts"]) * len(p["seeds"]) for t in p["tasks"])
    if p["fps"] != FPS or tuple(p["seeds"]) != SEEDS or p["expected_samples"] != count:
        raise ValueError("Unexpected FPS, seeds or sample count")
    if schema == 1 and count != 23400:
        raise ValueError("Original panel size changed")
    if tuple(t["name"] for t in p["tasks"]) != TASKS:
        raise ValueError("Nine-task order changed")
    for task in p["tasks"]:
        n = (2360 if schema == 1 else p["sources"]["task_command_counts"]["reach"]) if task["name"] == "reach" else 30
        commands = np.asarray(task["commands"], dtype=np.float32)
        pose = np.asarray(task["reference"]["positions_f"])
        if commands.shape != (n, 3) or not np.isfinite(commands).all():
            raise ValueError("Invalid frozen commands")
        if pose.shape != (22, 3) or not np.isfinite(pose).all() or not np.allclose(pose[0, :2], 0, atol=1e-6):
            raise ValueError("Invalid reference pose")
        if task["lengths"] != [LENGTHS[task["name"]]] * n:
            raise ValueError("Changed task duration")
        if len(task["prompts"]) != 3 or len(set(task["prompts"])) != 3:
            raise ValueError("Invalid prompt panel")
        definition = task["definition"]
        if definition["evaluation_frames"] != LENGTHS[task["name"]] or definition["prompts"] != task["prompts"]:
            raise ValueError("Evaluation definition does not match sample panel")
        if schema >= 2:
            bounds = np.asarray(definition["bounds"])
            values = commands[:, :len(bounds)]
            if not ((values >= bounds[:, 0] - 1e-6) & (values <= bounds[:, 1] + 1e-6)).all():
                raise ValueError("Commands outside current task bounds")
        workspace_hash = WORKSPACE if schema == 1 else p["sources"]["workspace_sha256"]
        if task["name"] == "reach" and task["metadata"]["workspace_sha256"] != workspace_hash:
            raise ValueError("Reach workspace changed")
    return p


def load_protocol(path, verify=True):
    p = validate_protocol(read_json(path))
    if verify:
        verify_sources(Path(path).parent / "scoring_source", p["scoring"]["files"])
    return p


def archive_path(root, record):
    return Path(root) / "samples" / record["task"] / (record["sample_id"] + ".npz")


def read_sample(path, record, run_id, names77):
    with np.load(path, allow_pickle=False) as data:
        a = dict(data)
    if str(a.get("sample_sha256", "")) != sample_digest(a):
        raise ValueError("Sample content hash mismatch")
    if str(a["run_id"]) != run_id or str(a["sample_id"]) != record["sample_id"]:
        raise ValueError("Sample identity mismatch")
    for field, expected in [("length", record["frames"]), ("fps", FPS), ("native_fps", NATIVE_FPS),
                            ("seed", record["seed"]), ("prompt_index", record["prompt_index"]),
                            ("command_index", record["command_index"])]:
        if int(a[field]) != expected:
            raise ValueError(f"Wrong sample {field}")
    if str(a["prompt"]) != record["prompt"] or not np.array_equal(a["command"], np.asarray(record["command"], np.float32)):
        raise ValueError("Wrong sample prompt/command")
    if a["actual_prompt"].shape != () or a["actual_prompt"].dtype.kind not in "US" or not str(a["actual_prompt"]).strip():
        raise ValueError("Missing actual text-encoder prompt")
    n = native_frames(record["frames"])
    shapes = {"native_positions_k": (n, 77, 3), "positions_f": (record["frames"], 22, 3),
              "local_rot_mats": (n, 77, 3, 3), "root_positions": (n, 3),
              "smooth_root_pos": (n, 3), "observed_motion": (n, 369), "motion_mask": (n, 369)}
    for key, shape in shapes.items():
        if a[key].shape != shape or not np.isfinite(a[key]).all():
            raise ValueError(f"Invalid sample array: {key}")
    if a["motion_mask"].dtype != np.bool_:
        raise ValueError("Condition mask must be boolean")
    expected = scoring_positions(a["native_positions_k"], names77, record["frames"])
    if not np.array_equal(a["positions_f"], expected):
        raise ValueError("Scoring coordinates differ from native output")
    return a
