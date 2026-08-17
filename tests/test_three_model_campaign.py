import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def campaign():
    spec = importlib.util.spec_from_file_location("campaign", ROOT / "scripts" / "three_model_campaign.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def manifest_and_lock():
    c = campaign()
    manifest = c.load_manifest()
    return c, manifest, ROOT / "experiments" / "three_model_range_sweep.revisions.json"


def test_manifest_has_three_models_and_thirteen_configurations():
    c, manifest, _ = manifest_and_lock()
    assert set(manifest["models"]) == {"gemma2_27b", "llama2_13b", "mistral_nemo_12b"}
    assert len(c.configuration_names(manifest)) == 13


def test_all_array_indices_and_threshold_flags_are_exact():
    c, manifest, lock = manifest_and_lock()
    assert c.configuration_for_index(manifest, 0) == ("full", None)
    assert c.configuration_for_index(manifest, 3) == ("selected_bf16", None)
    assert c.configuration_for_index(manifest, 12) == ("range_threshold_bf16", 2.3)
    for index in range(13):
        command = c.command_for_stage("f", "mistral_nemo_12b", manifest, lock, array_index=index)
        assert ("--bf16_range_threshold" in command) is (index >= 4)
    with pytest.raises(ValueError, match="SLURM_ARRAY_TASK_ID"):
        c.configuration_for_index(manifest, 13)


def test_lock_rejects_missing_main_short_invalid_and_model_disagreement(tmp_path):
    c, manifest, lock = manifest_and_lock()
    base = json.loads(lock.read_text())
    for mutate in (
        lambda x: x.pop("gemma2_27b"),
        lambda x: x["gemma2_27b"].update(model_revision="main"),
        lambda x: x["gemma2_27b"].update(tokenizer_revision="abcd"),
        lambda x: x["gemma2_27b"].update(model_id="wrong/model"),
    ):
        candidate = json.loads(json.dumps(base)); mutate(candidate)
        path = tmp_path / "lock.json"; path.write_text(json.dumps(candidate))
        with pytest.raises(ValueError): c.validate_lock(manifest, path)


def test_lookup_command_and_calibration_command_are_locked():
    c, manifest, lock = manifest_and_lock()
    spec = c.model_lookup("llama2_13b", manifest, lock)
    assert spec["model_revision"] == "5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1"
    command = c.command_for_stage("c", "llama2_13b", manifest, lock)
    joined = " ".join(command)
    for flag in ("--simulate_row_parallel_calibration", "--dataset_revision", "--sampling_strategy random_token_chunks", "--dtype bfloat16"):
        assert flag in joined
    assert "--num_partitions 8" in joined and "--target_style llama" in joined


def test_stage_d_passes_exact_ordered_manifest_threshold_grid():
    c, manifest, lock = manifest_and_lock()
    command = c.command_for_stage("d", "mistral_nemo_12b", manifest, lock, job_id="123")
    threshold_arg = command[command.index("--thresholds") + 1]
    assert threshold_arg == ",".join(str(value) for value in manifest["range_thresholds"])


def test_stage_b_and_eval_commands_have_expected_unique_outputs():
    c, manifest, lock = manifest_and_lock()
    b = c.command_for_stage("b", "gemma2_27b", manifest, lock, job_id="123")
    assert "model_loading_smoke.py" in " ".join(b) and "--target_style" in b
    smoke = [c.command_for_stage("f", "gemma2_27b", manifest, lock, array_index=i) for i in range(13)]
    full = [c.command_for_stage("g", "gemma2_27b", manifest, lock, array_index=i) for i in range(13)]
    paths = {command[command.index("--output_path") + 1] for command in smoke + full}
    assert len(paths) == 26
    assert all("--limit" in command for command in smoke)
    assert all("--limit" not in command for command in full)


def test_no_overwrite_protection(tmp_path):
    c, _, _ = manifest_and_lock()
    path = tmp_path / "already-there.json"; path.write_text("x")
    with pytest.raises(FileExistsError): c._refuse_overwrite(path)


def test_dependency_graph_and_safe_no_argument_submission_behavior():
    c, _, _ = manifest_and_lock()
    assert c.dependency_graph("g") == [("b", None), ("c", "b"), ("d", "c"), ("f", "d"), ("g", "f")]
    result = subprocess.run(["bash", "experiments/slurm/submit_three_model_range_sweep.sh"], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 2 and "Usage:" in result.stderr


def test_dry_run_has_no_sbatch_call_or_token_value():
    token = "test-token-must-never-appear"
    env = dict(os.environ, HF_TOKEN=token, PYTHON_BIN=sys.executable)
    result = subprocess.run(
        ["bash", "experiments/slurm/submit_three_model_range_sweep.sh", "--dry-run", "--stage-only", "b"],
        cwd=ROOT, env=env, text=True, capture_output=True,
    )
    assert result.returncode == 0
    script = (ROOT / "experiments/slurm/submit_three_model_range_sweep.sh").read_text()
    assert "--export=ALL" in script and token not in script
    assert "sbatch" in script and token not in result.stdout + result.stderr
    assert "DRY-RUN" in result.stdout
    stage_b_lines = [line for line in result.stdout.splitlines() if "stage=b" in line]
    assert len(stage_b_lines) == 3
    for model, memory in (("gemma2_27b", "192G"), ("llama2_13b", "128G"), ("mistral_nemo_12b", "128G")):
        line = next(line for line in stage_b_lines if model in line)
        assert f"--mem={memory}" in line and "--export=ALL" in line
        assert str(ROOT / "experiments" / "three_model_range_sweep.revisions.json") in line


def test_submitter_finds_repository_when_started_outside_repository(tmp_path):
    token = "token-never-in-dry-run"
    result = subprocess.run(
        ["bash", str(ROOT / "experiments/slurm/submit_three_model_range_sweep.sh"), "--dry-run", "--stage-only", "b"],
        cwd=tmp_path, env=dict(os.environ, HF_TOKEN=token, PYTHON_BIN=sys.executable),
        text=True, capture_output=True,
    )
    assert result.returncode == 0
    assert str(ROOT / "experiments" / "three_model_range_sweep.revisions.json") in result.stdout
    assert token not in result.stdout + result.stderr


def test_sbatch_scripts_are_real_programs_not_echo_placeholders():
    for path in sorted((ROOT / "experiments" / "slurm").glob("stage_*.sbatch")):
        text = path.read_text()
        assert "run-stage" in text
        assert "echo \"Use scripts" not in text
        assert "set -euo pipefail" in text


def test_sbatch_stages_use_submit_dir_not_spooled_bash_source():
    for path in sorted((ROOT / "experiments" / "slurm").glob("stage_*.sbatch")):
        text = path.read_text()
        assert "BASH_SOURCE" not in text
        assert "ROOT=${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}" in text
        assert "Invalid campaign repository ROOT=$ROOT" in text


def test_spooled_stage_uses_submit_dir_and_reports_missing_venv(tmp_path):
    spool = tmp_path / "cm/local/apps/slurm/var/spool/job1"; spool.mkdir(parents=True)
    copied = spool / "slurm_script"
    copied.write_text((ROOT / "experiments/slurm/stage_b_load_smoke.sbatch").read_text())
    fake_repo = tmp_path / "fake-repository"
    for relative in (
        "experiments/three_model_range_sweep.yaml",
        "experiments/three_model_range_sweep.revisions.json",
        "scripts/three_model_campaign.py",
    ):
        marker = fake_repo / relative
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("marker")
    real_outputs = ROOT / "outputs" / "three_model_range_sweep" / "gemma2_27b"
    before = sorted(real_outputs.rglob("*")) if real_outputs.exists() else []
    result = subprocess.run(
        ["bash", str(copied), "gemma2_27b", str(fake_repo / "experiments/three_model_range_sweep.revisions.json")],
        env=dict(os.environ, SLURM_SUBMIT_DIR=str(fake_repo), SLURM_JOB_ID="1", HF_TOKEN="fake"),
        text=True, capture_output=True,
    )
    assert result.returncode == 2
    assert f"ROOT={fake_repo}" in result.stderr
    assert str(spool) not in result.stderr
    assert str(fake_repo / ".venv-gpu310/bin/activate") in result.stderr
    assert "model_loading_smoke.py" not in result.stdout + result.stderr
    after = sorted(real_outputs.rglob("*")) if real_outputs.exists() else []
    assert after == before


def test_stage_rejects_missing_lock_before_activation(tmp_path):
    repo = tmp_path / "repo"
    for relative in ("experiments/three_model_range_sweep.yaml", "experiments/three_model_range_sweep.revisions.json", "scripts/three_model_campaign.py", ".venv-gpu310/bin/activate"):
        path = repo / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("marker")
    result = subprocess.run(
        ["bash", str(ROOT / "experiments/slurm/stage_b_load_smoke.sbatch"), "gemma2_27b", str(repo / "missing-lock.json")],
        env=dict(os.environ, SLURM_SUBMIT_DIR=str(repo), SLURM_JOB_ID="1", HF_TOKEN="fake"),
        text=True, capture_output=True,
    )
    assert result.returncode == 2
    assert f"ROOT={repo}" in result.stderr and "revision lock" in result.stderr.lower()
