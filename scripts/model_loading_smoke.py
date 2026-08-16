"""GPU loading/discovery smoke; performs no calibration or evaluation."""
from __future__ import annotations
import argparse, json, platform, socket, subprocess, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from lowbit_tp_comm.dtypes import model_load_kwargs, validate_model_dtype
from lowbit_tp_comm.hooks import list_candidate_sync_modules

def _feature_dim(module: torch.nn.Module) -> int:
    """The last output dimension used by calibration partial outputs."""
    if hasattr(module, "out_features"):
        return int(module.out_features)
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim < 2:
        raise ValueError(f"Cannot determine feature_dim for {type(module).__name__}.")
    return int(weight.shape[0])

def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--model_name",required=True); p.add_argument("--model_revision",required=True); p.add_argument("--tokenizer_revision",required=True); p.add_argument("--target_style",choices=["auto","gpt2","llama"],required=True); p.add_argument("--output_path",required=True); p.add_argument("--memory_safety_fraction",type=float,default=.85); a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("Model loading smoke requires an H100 CUDA node.")
    output_path=Path(a.output_path)
    if output_path.exists(): raise FileExistsError(f"Refusing to overwrite loading smoke: {output_path}")
    device=torch.device("cuda"); properties=torch.cuda.get_device_properties(device); total=properties.total_memory
    gpu_name=torch.cuda.get_device_name(device)
    if "H100" not in gpu_name.upper(): raise RuntimeError(f"Loading smoke requires H100 80 GB; received {gpu_name!r}.")
    torch.cuda.reset_peak_memory_stats(device)
    tokenizer=AutoTokenizer.from_pretrained(a.model_name, revision=a.tokenizer_revision)
    model=AutoModelForCausalLM.from_pretrained(a.model_name, revision=a.model_revision, **model_load_kwargs("bfloat16")).eval().to(device)
    validate_model_dtype(model, torch.bfloat16); targets=list_candidate_sync_modules(model,target_style=a.target_style)
    names=[name for name,_ in targets]; o_proj=[name for name in names if name.endswith("self_attn.o_proj")]; down_proj=[name for name in names if name.endswith("mlp.down_proj")]
    peak=torch.cuda.max_memory_allocated(device); reserved=torch.cuda.max_memory_reserved(device); safe=peak <= total*a.memory_safety_fraction
    import transformers
    output={"model_name":a.model_name,"model_revision":a.model_revision,"tokenizer_revision":a.tokenizer_revision,"target_style":a.target_style,"dtype":"bfloat16","device":str(device),"hostname":socket.gethostname(),"slurm_job_id":__import__("os").environ.get("SLURM_JOB_ID"),"gpu_name":gpu_name,"gpu_total_bytes":total,"peak_allocated_bytes":peak,"peak_reserved_bytes":reserved,"memory_safety_fraction":a.memory_safety_fraction,"safe":safe,"targets": [{"name":n,"feature_dim":_feature_dim(m)} for n,m in targets],"target_count":len(targets),"o_proj_count":len(o_proj),"down_proj_count":len(down_proj),"torch_version":torch.__version__,"transformers_version":transformers.__version__,"python_version":platform.python_version(),"git_commit":_git_commit()}
    output_path.parent.mkdir(parents=True,exist_ok=True); output_path.write_text(json.dumps(output,indent=2),encoding="utf-8")
    if not o_proj or not down_proj or not safe: raise RuntimeError("Loading smoke failed expected boundary discovery or memory safety check.")
if __name__ == "__main__": main()
