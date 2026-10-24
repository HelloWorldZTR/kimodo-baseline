"""Deterministic native conditions. No trajectory fitting, IK or target clipping."""

import numpy as np

from .protocol import KEYFRAMES, LENGTHS, f_to_k, native_frames


def shortest_rotation(source, destination):
    """Proper shortest rotation, with a deterministic axis for antiparallel inputs."""
    a, b = np.asarray(source, float), np.asarray(destination, float)
    if min(np.linalg.norm(a), np.linalg.norm(b)) < 1e-8:
        raise ValueError("Cannot orient a degenerate reference limb")
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    c = np.clip(a @ b, -1, 1)
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        basis = np.eye(3)[np.argmin(np.abs(a))]
        axis = np.cross(a, basis)
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    v = np.cross(a, b)
    cross = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + cross + cross @ cross / (1 + c)


def describe_skeleton(skeleton, output_skeleton):
    if skeleton.name != "somaskel30" or output_skeleton.name != "somaskel77":
        raise ValueError("This adapter requires SOMA 30 -> 77")
    return {
        "names30": list(skeleton.bone_order_names), "names77": list(output_skeleton.bone_order_names),
        "neutral30": skeleton.neutral_joints.detach().cpu().numpy().tolist(),
    }


def condition_spec(task, command, frames, geometry, adapter=None):
    if adapter is not None and adapter["version"] == "kimodo_command9_anchored_delta_raw_v3":
        from .anchored_conditions import anchored_condition_spec
        return anchored_condition_spec(task, command, frames, geometry)
    name = task["name"]
    if frames != LENGTHS[name]:
        raise ValueError("Unexpected task length")
    pose = np.asarray(task["reference"]["positions_f"], dtype=float)
    command = np.asarray(command, dtype=np.float32).astype(float)
    if command.shape != (3,) or not np.isfinite(command).all():
        raise ValueError("Invalid command")
    n = native_frames(frames)
    spec = {"task": name, "native_frames": n, "native_fps": 30}
    if name in ("walk", "back_walk", "jog", "sidestep"):
        times = np.arange(n, dtype=float) / 30
        path = np.zeros((n, 2))
        if name == "sidestep":
            u = np.clip(times / ((frames - 1) / 20), 0, 1)
            path[:, 0] = -command[0] * u * u * (3 - 2 * u)
            path[-2:, 0] = -command[0]
        else:
            path[:, 1] = command[0] * times * (-1 if name == "back_walk" else 1)
        spec.update(kind="root2d", frame_indices=list(range(n)), smooth_root_2d=path.tolist())
        return spec

    names = geometry["names30"]
    neutral = np.asarray(geometry["neutral30"], dtype=float)
    if neutral.shape != (30, 3) or names[0] != "Hips" or not np.isfinite(neutral).all():
        raise ValueError("Invalid SOMA geometry")
    h = pose[0, 2]
    root = np.array([0, h, 0.0])
    joints = neutral - neutral[0] + root
    rotations = np.repeat(np.eye(3)[None], 30, axis=0)
    anchor, distal = "Hips", None
    kind = "end-effector"
    if name in ("jump", "squat"):
        root[1] += command[0] * (1 if name == "jump" else -1)
        joints = neutral - neutral[0] + root
        primary = root.copy()
    elif name in ("reach", "strike"):
        kind, anchor, distal = "right-hand", "RightHand", "RightHandMiddleEnd"
        target_f = command + [0, 0, h] if name == "reach" else pose[17] + [command[0], 0, 0]
        primary = f_to_k(target_f)
        ai, di = names.index(anchor), names.index(distal)
        offset = neutral[di] - neutral[ai]
        rotation = shortest_rotation(offset, f_to_k(pose[21] - pose[19]))
        joints[ai] = primary
        joints[di] = primary + rotation @ offset
        rotations[ai] = rotation
    elif name == "kick":
        kind, anchor, distal = "right-foot", "RightFoot", "RightToeBase"
        primary = f_to_k([pose[11, 0] + command[0], pose[11, 1], 1.0])
        ai, di = names.index(anchor), names.index(distal)
        offset = neutral[di] - neutral[ai]
        rotation = shortest_rotation(offset, f_to_k(pose[11] - pose[8]))
        joints[di] = primary
        joints[ai] = primary - rotation @ offset
        rotations[ai] = rotation
    else:
        raise ValueError("Unsupported event task")
    spec.update(
        kind=kind, frame_indices=[KEYFRAMES[name]], joint_names=[anchor],
        global_joints_positions=joints[None].tolist(), global_joints_rots=rotations[None].tolist(),
        smooth_root_2d=[[0.0, 0.0]], global_root_heading=[[1.0, 0.0]],
        primary_joint=distal if name == "kick" else anchor, primary_target_k=primary.tolist(),
    )
    return spec


def native_constraints(spec, skeleton, device="cpu"):
    """Lazy imports keep export/preflight usable without PyTorch or CUDA."""
    import torch
    from kimodo.constraints import EndEffectorConstraintSet, RightFootConstraintSet, RightHandConstraintSet, Root2DConstraintSet

    if spec["kind"] == "sequence":
        return [constraint for part in spec["constraints"]
                for constraint in native_constraints(part, skeleton, device)]

    frames = torch.tensor(spec["frame_indices"], dtype=torch.long)
    root = torch.tensor(spec["smooth_root_2d"], dtype=torch.float32)
    if spec["kind"] == "root2d":
        constraint = Root2DConstraintSet(skeleton, frames, root)
    else:
        cls = {"right-hand": RightHandConstraintSet, "right-foot": RightFootConstraintSet,
               "end-effector": EndEffectorConstraintSet}[spec["kind"]]
        extra = {"joint_names": spec["joint_names"]} if spec["kind"] == "end-effector" else {}
        constraint = cls(skeleton, frames,
                         torch.tensor(spec["global_joints_positions"], dtype=torch.float32),
                         torch.tensor(spec["global_joints_rots"], dtype=torch.float32), root, **extra)
        # This is the agreed heading, independent of rounding in neutral rig hips.
        constraint.global_root_heading = torch.tensor(spec["global_root_heading"], dtype=torch.float32)
    # Native .to(dtype=...) also casts index tensors. All floating inputs were
    # already created as float32; move only the device to preserve integer indices.
    return [constraint.to(device=device)]


def expected_mask(spec, geometry):
    """Independent audit of the pinned 369-channel SOMA representation."""
    mask = np.zeros((spec["native_frames"], 369), dtype=bool)
    if spec["kind"] == "sequence":
        for part in spec["constraints"]:
            part_mask = expected_mask(part, geometry)
            if (mask & part_mask).any():
                raise ValueError("Overlapping constraint writes")
            mask |= part_mask
        return mask
    frames = np.asarray(spec["frame_indices"])
    if spec["kind"] == "root2d":
        mask[np.ix_(frames, [0, 2])] = True
        return mask
    mask[frames, :5] = True
    names = geometry["names30"]
    anchor = spec["joint_names"][0]
    positions = {"Hips": ["Hips"], "RightHand": ["RightHand", "RightHandMiddleEnd"],
                 "RightFoot": ["RightFoot", "RightToeBase"]}[anchor]
    for name in positions:
        j = names.index(name)
        mask[frames, 5 + j * 3:5 + j * 3 + 3] = True
    j = names.index(anchor)
    mask[frames, 95 + j * 6:95 + j * 6 + 6] = True
    return mask


def primary_error(spec, positions77, smooth_root, geometry):
    if spec["kind"] == "sequence":
        return primary_error(spec["constraints"][spec["primary_condition"]], positions77, smooth_root, geometry)
    frames = spec["frame_indices"]
    if spec["kind"] == "root2d":
        residual = smooth_root[frames][:, [0, 2]] - np.asarray(spec["smooth_root_2d"])
    else:
        index = geometry["names77"].index(spec["primary_joint"])
        residual = positions77[frames, index] - np.asarray(spec["primary_target_k"])
    return float(np.linalg.norm(residual, axis=-1).mean())
