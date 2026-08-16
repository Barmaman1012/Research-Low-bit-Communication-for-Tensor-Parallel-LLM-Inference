"""CPU-only result validator for the three-model range sweep."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
from three_model_campaign import configuration_names, load_manifest, output_path
def main() -> None:
 p=argparse.ArgumentParser(); p.add_argument('--manifest',default='experiments/three_model_range_sweep.yaml'); p.add_argument('--stage',choices=['smoke','full'],default='full'); p.add_argument('--output_dir',required=True); a=p.parse_args(); m=load_manifest(a.manifest); report={'missing':[],'invalid':[],'found':[]}
 for model in m['models']:
  for config in configuration_names(m):
   path=output_path(m,model,a.stage,config)
   if not path.exists(): report['missing'].append(str(path)); continue
   try:
    payload=json.loads(path.read_text()); metadata=next(iter(payload.get('metadata',{}).values()));
    if metadata.get('model_name') != m['models'][model]['model_id']: raise ValueError('model mismatch')
    report['found'].append(str(path))
   except Exception as exc: report['invalid'].append({'path':str(path),'error':str(exc)})
 out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); (out/'missing_invalid_results.json').write_text(json.dumps(report,indent=2)); print(json.dumps({k:len(v) for k,v in report.items()}))
 if report['missing'] or report['invalid']: raise SystemExit(1)
if __name__=='__main__': main()
