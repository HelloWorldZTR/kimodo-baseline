"""Match relative command quantities with sparse native start/target/return anchors.

No generated outputs determine these targets. No IK, target clipping, root
relocation, additional text, or postprocessing is used. Unchanged tasks retain
byte-identical condition specifications from v2.
"""

import copy
import math

import numpy as np

from .conditions import condition_spec
from .protocol import KEYFRAMES, f_to_k


def scoring_window(start, stop):
    """Inclusive native frame support of half-open 20 Hz scoring frames."""
    return list(range(math.floor(start * 1.5), math.ceil((stop - 1) * 1.5) + 1))


def stationary_constraint(template, frame_indices, primary_joint, target, geometry):
    """Translate only the selected EE group and retain its fixed orientation."""
    result = copy.deepcopy(template)
    names = geometry['names30']
    primary_index = names.index(primary_joint)
    positions = np.asarray(template['global_joints_positions'])[0].copy()
    offset = np.asarray(target) - positions[primary_index]
    anchor = template['joint_names'][0]
    group = {'RightHand': ['RightHand', 'RightHandMiddleEnd'],
             'RightFoot': ['RightFoot', 'RightToeBase'], 'Hips': ['Hips']}[anchor]
    for name in group:
        positions[names.index(name)] += offset
    count = len(frame_indices)
    result.update(frame_indices=frame_indices,
                  global_joints_positions=np.repeat(positions[None], count, axis=0).tolist(),
                  global_joints_rots=[copy.deepcopy(template['global_joints_rots'][0]) for _ in frame_indices],
                  smooth_root_2d=[[0., 0.] for _ in frame_indices],
                  global_root_heading=[[1., 0.] for _ in frame_indices],
                  primary_joint=primary_joint, primary_target_k=np.asarray(target).tolist())
    return result


def anchored_condition_spec(task, command, frames, geometry):
    template = condition_spec(task, command, frames, geometry)
    name = task['name']
    if name not in ('strike', 'kick', 'jump', 'squat'):
        return template
    pose = np.asarray(task['reference']['positions_f'], dtype=float)
    distance = float(np.asarray(command, np.float32)[0])
    first = [0]
    last = scoring_window(frames - 1, frames)
    target_frames = [KEYFRAMES[name]]
    if name == 'strike':
        # A punch is an event, not a whole-clip reach: preserve the ready pose
        # at 0.5 s and return at 1.5 s around the fixed 1.0 s hit keyframe.
        first = [0, 15]
        last = [45] + last
        primary = 'RightHand'
        start = f_to_k(pose[21])
        target = f_to_k([pose[21, 0] + distance, pose[17, 1], pose[17, 2]])
    elif name == 'kick':
        primary = 'RightFoot'  # The task metric is ankle excursion, not toe excursion.
        start = f_to_k(pose[8])
        names = geometry['names30']
        positions = np.asarray(template['global_joints_positions'])[0]
        toe_offset = positions[names.index('RightToeBase')] - positions[names.index('RightFoot')]
        target = f_to_k([pose[8, 0] + distance, pose[8, 1], 1.0 - toe_offset[1]])
    else:
        primary = 'Hips'
        start = f_to_k(pose[0])
        target = start + [0., distance * (1 if name == 'jump' else -1), 0.]
        if name == 'squat':
            first = scoring_window(0, 5)
            last = scoring_window(frames - 5, frames)
            center = round(KEYFRAMES[name] / 1.5)
            target_frames = scoring_window(center - 2, center + 3)
    parts = [stationary_constraint(template, first, primary, start, geometry),
             stationary_constraint(template, target_frames, primary, target, geometry),
             stationary_constraint(template, last, primary, start, geometry)]
    return dict(kind='sequence', task=name, native_frames=template['native_frames'], native_fps=30,
                constraints=parts, primary_condition=1, primary_joint=primary,
                primary_target_k=np.asarray(target).tolist(),
                frame_indices=sorted(frame for part in parts for frame in part['frame_indices']),
                initial_primary_target_k=np.asarray(start).tolist(),
                command_semantics='fixed initial joint forward displacement' if name in ('strike','kick')
                                  else 'fixed initial pelvis height difference')
