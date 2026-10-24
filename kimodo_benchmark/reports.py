"""Comparable response plots and offline native-frame motion galleries."""

import json
import numpy as np

from .protocol import JOINTS_22, archive_path, atomic_bytes, read_sample, samples


def response_plot(path, task, result):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    requested = np.asarray(result["requested"])
    measured = np.asarray(result["measured"])
    deviation = np.asarray(result["measured_std"])
    dimensions = requested.shape[1]
    figure, axes = plt.subplots(
        1, dimensions, figsize=(5 * dimensions, 4), squeeze=False
    )
    for component, axis in enumerate(axes[0]):
        order = np.argsort(requested[:, component], kind="stable")
        target = requested[order, component]
        if dimensions == 1:
            for curve in result["curves"]:
                axis.plot(
                    target,
                    np.asarray(curve["measured"])[order, component],
                    alpha=0.25,
                    linewidth=0.8,
                )
            axis.errorbar(
                target,
                measured[order, component],
                yerr=deviation[order, component],
                color="#245db5",
                linewidth=1.5,
                label="mean ± population std",
            )
        else:
            # These are projections of the same workspace, not invented XYZ slices.
            axis.scatter(
                target,
                measured[order, component],
                s=6,
                alpha=0.45,
                label="workspace targets: mean",
            )
        axis.plot(
            [target.min(), target.max()],
            [target.min(), target.max()],
            "k--",
            label="identity",
        )
        label = "xyz"[component] if dimensions == 3 else task["name"]
        axis.set(
            xlabel="Requested {} ({})".format(label, task["definition"]["unit"]),
            ylabel="Measured ({})".format(task["definition"]["unit"]),
        )
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle(
        "{} · nMAE {:.4f} · {} samples".format(
            task["name"], result["normalized_mae"], result["samples"]
        )
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)

def gallery_indices(task, result):
    indices = np.asarray(result["command_indices"])
    commands = np.asarray(task["commands"])[indices]
    if task["name"] == "reach":
        values = np.linalg.norm(commands - task["metadata"]["workspace_center"], axis=1)
    else:
        values = commands[:, 0]
    order = np.argsort(values, kind="stable")
    return set(indices[order[[0, len(order) // 2, len(order) - 1]]].tolist())

def motion_gallery(path, protocol, run, task, result, generated_root):
    selected = gallery_indices(task, result)
    examples = []
    names = run["geometry"]["names77"]
    indices = [names.index(name) for name in JOINTS_22]
    for record in samples(protocol, [task["name"]]):
        if record["command_index"] not in selected:
            continue
        arrays = read_sample(archive_path(generated_root, record), record, run["run_id"], names)
        spec = json.loads(str(arrays["condition_json"]))
        positions = arrays["native_positions_k"][:, indices]
        hint = np.zeros((len(positions), 3))
        mask = np.zeros(len(positions), dtype=bool)
        targets = []
        if spec["kind"] == "root2d":
            hint[:, [0, 2]] = spec["smooth_root_2d"]
            mask[spec["frame_indices"]] = True
        else:
            parts = spec["constraints"] if spec["kind"] == "sequence" else [spec]
            for part in parts:
                frames = part["frame_indices"]
                hint[frames] = part["primary_target_k"]
                mask[frames] = True
                matching = next((target for target in targets if target["position"] == part["primary_target_k"]), None)
                if matching is None:
                    targets.append(dict(position=part["primary_target_k"], frames=list(frames)))
                else:
                    matching["frames"].extend(frames)
            for target in targets:
                frame_list = sorted(set(target["frames"]))
                ranges = []
                first = last = frame_list[0]
                for frame in frame_list[1:] + [None]:
                    if frame is not None and frame == last + 1:
                        last = frame
                        continue
                    ranges.append(str(first) if first == last else f"{first}–{last}")
                    first = last = frame
                target["label"] = "frames " + ", ".join(ranges)
        examples.append(dict(
            label=f"c{record['command_index']} · {record['command']} · prompt {record['prompt_index']} · seed {record['seed']}",
            prompt=record["prompt"], positions=np.round(positions, 5).tolist(),
            hint=np.round(hint, 5).tolist(), hint_mask=mask.tolist(), targets=targets, root_path=spec["kind"] == "root2d"))
    # Text uses textContent, and escaped JSON cannot close the script element.
    data = json.dumps(examples, separators=(",", ":"), allow_nan=False).replace(
        "<", "\\u003c"
    )
    template = """<!doctype html>
<html lang="en"><meta charset="utf-8"><title>Command benchmark motions</title>
<style>body{font:16px system-ui;max-width:1000px;margin:30px auto;color:#15263a}canvas{width:100%;border:1px solid #ccd5df;background:#f8fafc}select{max-width:100%;padding:8px}button{padding:8px}label{margin-right:20px}.muted{color:#50647c}</style>
<h1>Fixed command examples</h1><p class="muted">All prompts and seeds at low / median / high commands. Reach uses distance to the workspace shoulder. Native 30 Hz motion; 22 anatomical joints shown. Blue: generated skeleton. Red: requested targets, always visible; filled at constrained frames, hollow otherwise. Target labels give native frame indices. Gray: actual pelvis path. Every target conditions every diffusion step; marker visibility does not change model input.</p>
<select id="sample"></select><p id="prompt"></p><canvas id="view" width="1000" height="600"></canvas>
<p><button id="play">Pause</button> <label>Frame <input id="frame" type="range" min="0" value="0"><span id="number"></span></label><label>Camera <input id="angle" type="range" min="0" max="360" value="35"></label></p>
<script>
const examples=__DATA__;
const selector=document.getElementById('sample'), canvas=document.getElementById('view'), context=canvas.getContext('2d'), slider=document.getElementById('frame'), angle=document.getElementById('angle');
const chains=[[0,2,5,8,11],[0,1,4,7,10],[0,3,6,9,12,15],[9,14,17,19,21],[9,13,16,18,20]];
let current=0,playing=true,previous=0;
examples.forEach((example,index)=>{let option=document.createElement('option');option.value=index;option.textContent=example.label;selector.append(option);});
function select(){current=Number(selector.value);slider.max=examples[current].positions.length-1;slider.value=0;document.getElementById('prompt').textContent=examples[current].prompt;draw();}
function draw(){const example=examples[current],frame=Number(slider.value),radians=Number(angle.value)*Math.PI/180;
 const activeHint=example.hint.filter((point,index)=>!example.hint_mask||example.hint_mask[index]);
 const points=example.positions.flat().concat(activeHint),minimum=[0,1,2].map(axis=>Math.min(...points.map(point=>point[axis]))),maximum=[0,1,2].map(axis=>Math.max(...points.map(point=>point[axis])));
 const center=minimum.map((value,axis)=>(value+maximum[axis])/2),scale=430/Math.max(2,Math.hypot(maximum[0]-minimum[0],maximum[2]-minimum[2]),maximum[1]-minimum[1]);
 function project(point){const lateral=point[0]-center[0],depth=point[2]-center[2],height=point[1]-center[1];return [500+scale*(Math.cos(radians)*lateral+Math.sin(radians)*depth),300-scale*(0.90*height-0.22*(-Math.sin(radians)*lateral+Math.cos(radians)*depth))];}
 function line(points,color,width){context.beginPath();points.forEach((point,index)=>{const pixel=project(point);if(index)context.lineTo(...pixel);else context.moveTo(...pixel);});context.strokeStyle=color;context.lineWidth=width;context.stroke();}
 context.clearRect(0,0,1000,600);line(example.positions.map(pose=>pose[0]),'#bcc9db',1);if(example.root_path)line(activeHint,'#db7777',1);
 chains.forEach(chain=>line(chain.map(joint=>example.positions[frame][joint]),'#245db5',4));
 for(const goal of example.targets){const pixel=project(goal.position);context.beginPath();context.arc(...pixel,7,0,2*Math.PI);context.strokeStyle='#c92d32';context.lineWidth=2;context.stroke();if(goal.frames.includes(frame)){context.fillStyle='#c92d32';context.fill();}context.font='12px system-ui';context.fillStyle='#8b2024';context.fillText(goal.label,pixel[0]+10,pixel[1]-8);}
 if(example.root_path){const target=project(example.hint[frame]);context.beginPath();context.arc(...target,6,0,2*Math.PI);context.fillStyle='#c92d32';context.fill();}document.getElementById('number').textContent=frame+'/'+slider.max;
}
selector.onchange=select;slider.oninput=draw;angle.oninput=draw;document.getElementById('play').onclick=()=>{playing=!playing;document.getElementById('play').textContent=playing?'Pause':'Play';};
function tick(time){if(playing&&time-previous>=1000/30){slider.value=(Number(slider.value)+1)%examples[current].positions.length;draw();previous=time;}requestAnimationFrame(tick);}select();requestAnimationFrame(tick);
</script></html>"""
    atomic_bytes(path, template.replace("__DATA__", data).encode())
