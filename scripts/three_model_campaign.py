"""CPU-only campaign manifest helpers; never submits Slurm jobs."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "experiments" / "three_model_range_sweep.yaml"

def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    with Path(path).open() as handle: return yaml.safe_load(handle)

def configuration_names(manifest: dict[str, Any]) -> list[str]:
    return [*manifest["modes"], *[f"range_threshold_bf16-t{t:.1f}" for t in manifest["range_thresholds"]]]

def output_path(manifest: dict[str, Any], model: str, stage: str, configuration: str) -> Path:
    return ROOT / manifest["outputs"]["root"] / manifest["outputs"]["result_pattern"].format(model=model, stage=stage, configuration=configuration)

def validate_lock(manifest: dict[str, Any], lock_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(lock_path or ROOT / manifest["experiment"]["revision_lock"])
    if not path.exists(): raise ValueError(f"Immutable revision lock is required: {path}")
    lock = json.loads(path.read_text())
    for slug, spec in manifest["models"].items():
        for field in ("model_revision", "tokenizer_revision"):
            value = lock.get(slug, {}).get(field, spec[field])
            if not isinstance(value, str) or value == "RESOLVE_STAGE_A" or len(value) < 12:
                raise ValueError(f"{slug} has no immutable {field} in {path}")
    return lock

def resolve_revisions(manifest: dict[str, Any], output: str | Path) -> None:
    """Authenticated read-only HF lookup; record immutable repo commits, never main."""
    from huggingface_hub import HfApi
    api = HfApi(); resolved = {}
    for slug, spec in manifest["models"].items():
        model_revision = spec["model_revision"]
        if model_revision == "RESOLVE_STAGE_A": model_revision = api.model_info(spec["model_id"]).sha
        tokenizer_revision = spec["tokenizer_revision"]
        if tokenizer_revision == "RESOLVE_STAGE_A": tokenizer_revision = api.model_info(spec["model_id"]).sha
        resolved[slug] = {"model_id": spec["model_id"], "model_revision": model_revision, "tokenizer_revision": tokenizer_revision}
    Path(output).write_text(json.dumps(resolved, indent=2) + "\n")

def checksum_indices(indices: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, indices)).encode()).hexdigest()

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("command", choices=["resolve-revisions", "list-configurations"]); parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST)); parser.add_argument("--output", default=None); args=parser.parse_args()
    manifest=load_manifest(args.manifest)
    if args.command == "list-configurations": print("\n".join(configuration_names(manifest))); return
    resolve_revisions(manifest, args.output or str(ROOT / manifest["experiment"]["revision_lock"]))
if __name__ == "__main__": main()
