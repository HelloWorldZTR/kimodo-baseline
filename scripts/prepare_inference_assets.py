"""Download pinned Kimodo assets, using ModelScope for the original Llama weights.

Run inside the kimodo environment. No HF token is required for these public sources.
Llama weight hashes must match the official Meta HF metadata before use.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from urllib.parse import quote

import httpx
from huggingface_hub import HfApi, snapshot_download

LLAMA = "meta-llama/Meta-Llama-3-8B-Instruct"
LLAMA_REV = "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
MIRROR = "LLM-Research/Meta-Llama-3-8B-Instruct"
MIRROR_REV = "e9f7e7d3fa08b550ea228e38bb5501d35be92c0d"
PUBLIC = {
    "nvidia/Kimodo-SOMA-RP-v1.2": "0e3c89024d896064b6851201373f7834784a386b",
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp": "31474e395ada192e8ed1586db6be79fb3b70c9c0",
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised": "baa8ebf04a1c2500e61288e7dad65e8ae42601a7",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.home() / "checkpoints/kimodo")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    official = HfApi().model_info(LLAMA, revision=LLAMA_REV, files_metadata=True)
    official_weights = {s.rfilename: s.lfs.sha256 for s in official.siblings if s.rfilename.endswith(".safetensors")}
    response = httpx.get(
        f"https://modelscope.cn/api/v1/models/{MIRROR}/repo/files",
        params={"Revision": MIRROR_REV, "Recursive": "true"}, timeout=30,
    )
    response.raise_for_status()
    files = [f for f in response.json()["Data"]["Files"]
             if f["Type"] == "blob" and "/" not in f["Path"] and f["Path"] != ".gitattributes"]
    mirror_weights = {f["Path"]: f["Sha256"] for f in files if f["Path"].endswith(".safetensors")}
    if mirror_weights != official_weights:
        raise RuntimeError("ModelScope weights do not match the pinned official Meta checkpoint")
    base = root / "text_encoders" / LLAMA
    base.mkdir(parents=True, exist_ok=True)

    def download_file(info):
        path = base / info["Path"]
        expected = info["Sha256"]
        if path.exists() and path.stat().st_size == info["Size"] and sha256(path) == expected:
            print("Verified existing", path.name, flush=True)
            return
        if path.exists():
            raise RuntimeError(f"Existing invalid file: {path}; inspect it before retrying")
        partial = path.with_name(path.name + ".partial")
        url = f"https://modelscope.cn/models/{MIRROR}/resolve/{MIRROR_REV}/{quote(info['Path'])}"
        print("Downloading", path.name, info["Size"], flush=True)
        curl = ["curl", "--fail", "--location", "--silent", "--show-error",
                "--retry", "5", "--retry-all-errors", "--retry-delay", "3", "--connect-timeout", "30",
                "--speed-limit", "1024", "--speed-time", "120"]
        if info["Size"] > 512 * 1024 * 1024:
            # Keep an already-downloaded prefix and fetch the remainder in bounded
            # parallel ranges. Never allocate model tensors or touch CUDA.
            start = partial.stat().st_size if partial.exists() else 0
            if start > info["Size"]:
                raise RuntimeError(f"Oversized partial file: {partial}")
            ranges = [(offset, min(offset + 256 * 1024 * 1024, info["Size"]) - 1)
                      for offset in range(start, info["Size"], 256 * 1024 * 1024)]

            def fetch_range(bounds):
                first, last = bounds
                piece = path.with_name(f"{path.name}.range-{first}-{last}")
                if not piece.exists() or piece.stat().st_size != last - first + 1:
                    subprocess.run(curl + ["--range", f"{first}-{last}", "--output", str(piece), url], check=True)
                if piece.stat().st_size != last - first + 1:
                    raise RuntimeError(f"Invalid HTTP range length: {piece}")
                return piece

            with ThreadPoolExecutor(max_workers=4) as ranges_pool:
                pieces = list(ranges_pool.map(fetch_range, ranges))
            # Assemble to a new file so interruption cannot corrupt the prefix.
            assembled = path.with_name(path.name + ".assembling")
            with assembled.open("wb") as output:
                if partial.exists():
                    with partial.open("rb") as source:
                        shutil.copyfileobj(source, output, 8 * 1024 * 1024)
                for piece in pieces:
                    with piece.open("rb") as source:
                        shutil.copyfileobj(source, output, 8 * 1024 * 1024)
            if assembled.stat().st_size != info["Size"] or sha256(assembled) != expected:
                raise RuntimeError(f"Checksum failure: {assembled}")
            assembled.rename(path)
            partial.unlink(missing_ok=True)
            for piece in pieces:
                piece.unlink()
            print("Verified", path.name, flush=True)
            return
        subprocess.run(curl + ["--continue-at", "-", "--output", str(partial), url], check=True)
        if partial.stat().st_size != info["Size"] or sha256(partial) != expected:
            raise RuntimeError(f"Checksum failure: {partial}")
        partial.rename(path)
        print("Verified", path.name, flush=True)

    def download_public(item):
        repo, revision = item
        destination = root / (repo.split("/")[1] if repo.startswith("nvidia/") else "text_encoders/" + repo)
        # Restore our sole local config edit before verifying the upstream copy
        # on a repeat run. Reject unrelated edits rather than overwriting them.
        original = destination / "adapter_config.upstream.json"
        config_path = destination / "adapter_config.json"
        if original.exists() and config_path.exists():
            upstream = json.loads(original.read_text())
            current = json.loads(config_path.read_text())
            expected_local = {**upstream, "base_model_name_or_path": str(base)}
            if current not in (upstream, expected_local):
                raise RuntimeError(f"Unexpected local adapter edits: {config_path}")
            config_path.write_text(original.read_text())
        # A verified snapshot may also be transferred from a machine with a
        # working HF mirror connection. Recheck all files without downloading.
        transferred_metadata = destination / "source_metadata.json"
        if transferred_metadata.exists():
            metadata = json.loads(transferred_metadata.read_text())
            if metadata.get("id") != repo or metadata.get("sha") != revision:
                raise RuntimeError(f"Unexpected snapshot identity: {transferred_metadata}")
            for entry in metadata["siblings"]:
                local = destination / entry["rfilename"]
                if not local.resolve().is_relative_to(destination.resolve()):
                    raise RuntimeError("Invalid snapshot filename")
                if local.stat().st_size != entry["size"]:
                    raise RuntimeError(f"Wrong size: {local}")
                if entry.get("lfs") and sha256(local) != entry["lfs"]["sha256"]:
                    raise RuntimeError(f"Checksum failure: {local}")
                if not entry.get("lfs"):
                    data = local.read_bytes()
                    blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
                    if blob != entry["blobId"]:
                        raise RuntimeError(f"Git blob checksum failure: {local}")
            print("Verified transferred snapshot", repo, flush=True)
            return
        snapshot_download(repo, revision=revision, local_dir=destination, max_workers=3)
        metadata = HfApi().model_info(repo, revision=revision, files_metadata=True)
        for entry in metadata.siblings:
            local = destination / entry.rfilename
            if local.stat().st_size != entry.size:
                raise RuntimeError(f"Wrong size: {local}")
            if entry.lfs and sha256(local) != entry.lfs.sha256:
                raise RuntimeError(f"Checksum failure: {local}")
        print("Downloaded and verified", repo, flush=True)

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(download_file, info) for info in files]
        futures += [pool.submit(download_public, item) for item in PUBLIC.items()]
        for future in futures:
            future.result()

    # Change only the local MNTP adapter's base-model location. Its config.json
    # retains the official _name_or_path used by LLM2Vec's prompt formatting.
    mntp = root / "text_encoders/McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
    config_path = mntp / "adapter_config.json"
    config = json.loads(config_path.read_text())
    original = mntp / "adapter_config.upstream.json"
    if not original.exists():
        original.write_text(config_path.read_text())
    config["base_model_name_or_path"] = str(base)
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    provenance = {
        "llama_official_repo": LLAMA, "llama_official_revision": LLAMA_REV,
        "llama_mirror_repo": MIRROR, "llama_mirror_revision": MIRROR_REV,
        "llama_files": files, "official_weight_sha256": official_weights,
        "huggingface_revisions": PUBLIC,
        "local_change": {"file": str(config_path), "base_model_name_or_path": str(base)},
    }
    (root / "asset_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("READY:", root, flush=True)


if __name__ == "__main__":
    main()
