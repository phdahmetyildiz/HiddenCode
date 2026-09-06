"""
Cluster job bundle + merge (Phase 11).

Author: Cursor Grok 4.6 High Fast
"""

import json
from pathlib import Path

from src.sweep import (
    ParameterSweep,
    SingleRunResult,
    collect_job_result_files,
    load_job_from_bundle,
    merge_job_results,
    run_single_job,
)
from tests.test_sweep import _fast_base, _fast_settings


def test_export_and_run_job_index(tmp_path: Path):
    sweep = ParameterSweep(_fast_settings(parallel_workers=1), base_config=_fast_base())
    paths = sweep.export_job_bundle(tmp_path / "bundle")
    assert paths["jobs"].exists()
    lines = paths["jobs"].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 8
    job = load_job_from_bundle(tmp_path / "bundle", 0)
    assert "base_config_dict" in job
    assert job["run_index"] == 0
    result = run_single_job(job)
    assert result.seed == job["seed"]
    out = tmp_path / "job0.json"
    out.write_text(json.dumps(result.to_dict()), encoding="utf-8")
    loaded = SingleRunResult.from_dict(json.loads(out.read_text(encoding="utf-8")))
    assert loaded.final_alive_count == result.final_alive_count


def test_merge_job_results(tmp_path: Path):
    settings = _fast_settings(parallel_workers=1)
    sweep = ParameterSweep(settings, base_config=_fast_base())
    jobs = sweep._build_jobs()[:4]
    files = []
    for i, job in enumerate(jobs):
        r = run_single_job(job)
        p = tmp_path / f"r{i}.json"
        p.write_text(json.dumps(r.to_dict()), encoding="utf-8")
        files.append(p)
    merged_dir = tmp_path / "merged"
    result = merge_job_results(files, settings, merged_dir)
    assert result.total_runs == 4
    assert (merged_dir / "summary.csv").exists()
    assert (merged_dir / "stability_report.json").exists()


def test_collect_job_result_files_ignores_metadata(tmp_path: Path):
    (tmp_path / "job_000000.json").write_text(
        json.dumps({
            "combination_id": 0,
            "combination_params": {},
            "seed": 1,
            "run_index": 0,
            "final_alive_count": 3,
        }),
        encoding="utf-8",
    )
    (tmp_path / "manifest.json").write_text(json.dumps({"n_jobs": 1}), encoding="utf-8")
    files = collect_job_result_files(tmp_path)
    assert len(files) == 1
    assert files[0].name == "job_000000.json"
