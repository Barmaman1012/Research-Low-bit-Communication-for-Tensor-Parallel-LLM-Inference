from pathlib import Path
import importlib.util
ROOT=Path(__file__).resolve().parents[1]
def campaign():
 spec=importlib.util.spec_from_file_location('campaign',ROOT/'scripts'/'three_model_campaign.py'); module=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(module); return module
def test_manifest_has_three_models_and_thirteen_configurations():
 m=campaign().load_manifest(); assert set(m['models']) == {'gemma2_27b','llama2_13b','mistral_nemo_12b'}; assert len(campaign().configuration_names(m)) == 13
def test_output_names_are_distinct_by_model_stage_and_configuration():
 c=campaign(); m=c.load_manifest(); paths={c.output_path(m,'gemma2_27b','smoke',x) for x in c.configuration_names(m)}; assert len(paths)==13; assert 't2.0' in str(c.output_path(m,'gemma2_27b','full','range_threshold_bf16-t2.0'))
