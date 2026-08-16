"""GPU loading/discovery smoke; performs no calibration or evaluation."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from lowbit_tp_comm.dtypes import model_load_kwargs, validate_model_dtype
from lowbit_tp_comm.hooks import list_candidate_sync_modules

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--model_name",required=True); p.add_argument("--model_revision",required=True); p.add_argument("--tokenizer_revision",required=True); p.add_argument("--output_path",required=True); p.add_argument("--memory_safety_fraction",type=float,default=.85); a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("Model loading smoke requires an H100 CUDA node.")
    device=torch.device("cuda"); total=torch.cuda.get_device_properties(device).total_memory
    tokenizer=AutoTokenizer.from_pretrained(a.model_name, revision=a.tokenizer_revision)
    model=AutoModelForCausalLM.from_pretrained(a.model_name, revision=a.model_revision, **model_load_kwargs("bfloat16")).eval().to(device)
    validate_model_dtype(model, torch.bfloat16); targets=list_candidate_sync_modules(model,target_style="llama")
    names=[name for name,_ in targets]; expected=[name for name in names if name.endswith("self_attn.o_proj") or name.endswith("mlp.down_proj")]
    peak=torch.cuda.max_memory_allocated(device); safe=peak <= total*a.memory_safety_fraction
    output={"model_name":a.model_name,"model_revision":a.model_revision,"tokenizer_revision":a.tokenizer_revision,"dtype":"bfloat16","device":str(device),"gpu_total_bytes":total,"peak_allocated_bytes":peak,"memory_safety_fraction":a.memory_safety_fraction,"safe":safe,"targets": [{"name":n,"feature_dim":int(getattr(m,"out_features",m.weight.shape[1]))} for n,m in targets],"expected_boundary_count":len(expected)}
    Path(a.output_path).parent.mkdir(parents=True,exist_ok=True); Path(a.output_path).write_text(json.dumps(output,indent=2))
    if not expected or not safe: raise RuntimeError("Loading smoke failed target discovery or memory safety check.")
if __name__ == "__main__": main()
