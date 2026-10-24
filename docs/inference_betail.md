# Betail Kimodo inference setup

Repository: `/home/psirobot/projects/kimodo-baseline` (commit `e1c219c`).
Environment: `/home/psirobot/miniconda3/envs/kimodo`.
Assets: `/home/psirobot/checkpoints/kimodo`.

Status (2026-10-23): all four asset groups downloaded and weight checksums verified
on betail. Local adapter paths and required config/statistics files were checked.
After the user confirmed the GPU was free, offline inference was verified with
`A person walks forward.`: 4 seconds, 120 frames, SOMA 77 joints, seed 1928,
100 diffusion steps, separated CFG `[2, 2]`, and no postprocessing. Results are in
`outputs/smoke_20261023/` (NPZ, BVH, demo example, preview, and validation JSON).
All output arrays are finite; forward root displacement is 4.716 m. Both text
adapters' 448 tensors were separately checked against their checkpoint files and
matched exactly despite intermediate Transformers loading warnings.

## Download and verification

```bash
conda activate kimodo
cd ~/projects/kimodo-baseline
HF_ENDPOINT=https://hf-mirror.com \
HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=30 \
python scripts/prepare_inference_assets.py
```

This downloads about 17.6 GB of model files, using CPU/network/disk only:

| Asset | Source | Purpose |
|---|---|---|
| Meta-Llama-3-8B-Instruct, BF16, four original safetensors shards | ModelScope `LLM-Research/Meta-Llama-3-8B-Instruct` | Base of the text encoder |
| LLM2Vec-Meta-Llama-3-8B-Instruct-mntp | Hugging Face `McGill-NLP` | First text-encoder adapter and tokenizer |
| LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised | Hugging Face `McGill-NLP` | Second text-encoder adapter |
| Kimodo-SOMA-RP-v1.2 | Hugging Face `nvidia` | Motion generator, config, normalization statistics |

All revisions are pinned in the preparation script. The four Llama shard hashes
are compared with the official Meta repository metadata, then downloaded files
are SHA256-verified. `asset_manifest.json` records sources, revisions, hashes,
and the local adapter path change. The original adapter config is retained as
`adapter_config.upstream.json`. No model weights or inference algorithms are changed.

Direct Hugging Face downloads repeatedly timed out. The command above uses
`hf-mirror.com` for public Kimodo/LLM2Vec files and ModelScope for Llama. Betail also
has an existing HTTP proxy at `127.0.0.1:17897`, available as a fallback with
`HTTPS_PROXY`/`HTTP_PROXY`; no machine-wide network settings were changed.
Downloads are resumable;
successful preparation ends with `READY` and writes `asset_manifest.json`.

The gated HF Llama URL returns 401 without approved access; changing `HF_ENDPOINT`
to `hf-mirror.com` does not fix that. ModelScope supplies the same four weight files
through its public distribution. The accompanying Llama license and use policy
are downloaded with the model.

No training dataset, BONES-SEED, TMR evaluation model, SMPL-X body model, or SOMA-X
body-model download is needed for SOMA skeleton NPZ/BVH inference. Skeleton assets
are already included in this repository. Dataset/evaluator downloads belong to
the specific benchmark setup, not this minimal inference setup.

## Run only after the training GPU is free

Initial setup downloaded files only, at the user's request. The subsequent
GPU smoke test passed once the user confirmed the GPU was free. Package
consistency, offline loading, generation, and NPZ/BVH export have been checked.

```bash
conda activate kimodo
cd ~/projects/kimodo-baseline
bash scripts/infer_offline.sh "A person walks forward." \
  --duration 4 --num_samples 1 --seed 1928 \
  --diffusion_steps 100 --cfg_type separated --cfg_weight 2 2 \
  --no-postprocess --bvh --save_example_dir \
  --output outputs/smoke/walk
```

The wrapper selects the pinned `Kimodo-SOMA-RP-v1.2`, local LLM2Vec assets, and
offline mode. It imports this checkout rather than a potentially different
installed package. Override `KIMODO_ASSET_ROOT` or `KIMODO_PYTHON` if needed.
The default text encoder uses CUDA. `TEXT_ENCODER_DEVICE=cpu` puts only the text
encoder on CPU; the motion generator still uses CUDA. **That setting does not
make inference safe to run alongside GPU training.** The wrapper is manual and
does not automatically launch inference after download.

To use native spatial constraints, append `--constraints /path/to/constraints.json`.
The format is documented in `docs/source/user_guide/constraints.md`.

## Baseline integration with OmniControl / frankenmotion

This prepares an inference installation, not the nine-task benchmark adapter.
For that adapter, preserve the existing frozen physical commands, exact prompts,
seeds, and scorer. Kimodo-SOMA uses 30 internal joints and exports 77 joints,
normally at 30 FPS, in metres with Y up and initial heading along +Z. The existing
command benchmark scores at 20 FPS in X-forward/Y-left/Z-up coordinates. Explicit
joint mapping, time sampling, and coordinate conversion are therefore required.
Do not relabel SOMA joints as HumanML3D's 22 joints or score 30 FPS data as 20 FPS.

Use the Python API for exact benchmark prompt strings: the convenience CLI splits
prompts on periods and adds periods back. The benchmark must explicitly record
CFG, denoising steps, model and adapter revisions, and whether native postprocessing
is enabled. The example above disables postprocessing to preserve raw generation;
report any postprocessed variant separately.

SOMA-RP-v1.2 is selected because upstream identifies it as the corrected-split
benchmark release. RP uses a different and larger training corpus than OmniControl;
this is a comparison of pretrained systems, not a matched-training-data experiment.
The SMPL-X variant offers 22-joint/AMASS convenience, but upstream warns of stronger
retargeting artifacts, so it is not substituted merely to match joint counts.
