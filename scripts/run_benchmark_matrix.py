#!/usr/bin/env python3
"""Fan-out benchmarking matrix across datasets/backbones/systems."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

load_dotenv()


def _expand_env(value: Optional[str]) -> Optional[str]:
    if value is None or not isinstance(value, str):
        return value
    return os.path.expandvars(value)


def _normalise_dataset(entry: Any) -> Dict[str, Any]:
    if isinstance(entry, str):
        entry = {"name": entry}
    if not isinstance(entry, dict):
        raise TypeError(f"Unsupported dataset entry: {entry}")

    entry.setdefault("label", entry.get("name"))
    entry.setdefault("dataset_arg", entry.get("name"))
    entry.setdefault("split", "dev")
    entry.setdefault("variant", "")
    entry.setdefault("enabled", True)
    entry.setdefault("env", {})
    entry.setdefault("extra_args", "")
    entry.setdefault("only_systems", [])
    entry.setdefault("skip_systems", [])
    entry.setdefault("only_backbones", [])
    entry.setdefault("skip_backbones", [])
    entry.setdefault("sample_limit", None)
    return entry


def _normalise_backbone(entry: Dict[str, Any]) -> Dict[str, Any]:
    required = {"name", "model"}
    missing = required - entry.keys()
    if missing:
        raise ValueError(f"Backbone entry missing keys: {missing}")

    entry.setdefault("label", entry["name"])
    entry.setdefault("enabled", True)
    entry.setdefault("env", {})
    entry.setdefault("decoding", {})
    entry.setdefault("extra_args", "")
    entry["model"] = _expand_env(entry.get("model"))
    return entry


def _normalise_system(entry: Dict[str, Any]) -> Dict[str, Any]:
    required = {"name", "command"}
    missing = required - entry.keys()
    if missing:
        raise ValueError(f"System entry missing keys: {missing}")

    entry.setdefault("label", entry["name"])
    entry.setdefault("enabled", True)
    entry.setdefault("env", {})
    entry.setdefault("role", "")
    entry.setdefault("extra_args", "")
    return entry


def load_matrix(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    datasets = [_normalise_dataset(item) for item in config.get("datasets", []) if item]
    backbones = [_normalise_backbone(item) for item in config.get("backbones", []) if item]
    systems = [_normalise_system(item) for item in config.get("systems", []) if item]
    baseline = config.get("baseline_system", "cot")
    return {"datasets": datasets, "backbones": backbones, "systems": systems, "baseline_system": baseline}


@dataclass
class BenchmarkJob:
    dataset: Dict[str, Any]
    backbone: Dict[str, Any]
    system: Dict[str, Any]
    command: str
    log_path: Path
    metrics_path: Path
    sample_limit: Optional[int]


def _is_enabled(entry: Dict[str, Any]) -> bool:
    return entry.get("enabled", True)


def _should_skip(dataset: Dict[str, Any], backbone: Dict[str, Any], system: Dict[str, Any]) -> bool:
    if dataset.get("only_backbones") and backbone["name"] not in dataset["only_backbones"]:
        return True
    if dataset.get("skip_backbones") and backbone["name"] in dataset["skip_backbones"]:
        return True
    if dataset.get("only_systems") and system["name"] not in dataset["only_systems"]:
        return True
    if dataset.get("skip_systems") and system["name"] in dataset["skip_systems"]:
        return True
    return False


def build_jobs(config: Dict[str, Any], logs_dir: Path, metrics_dir: Path) -> List[BenchmarkJob]:
    jobs: List[BenchmarkJob] = []
    for dataset in config["datasets"]:
        if not _is_enabled(dataset):
            continue
        for backbone in config["backbones"]:
            if not _is_enabled(backbone):
                continue
            for system in config["systems"]:
                if not _is_enabled(system) or _should_skip(dataset, backbone, system):
                    continue

                limit = dataset.get("sample_limit")
                if isinstance(limit, str) and limit.strip():
                    try:
                        limit = int(limit)
                    except ValueError:
                        limit = None
                env_limit = os.getenv("BENCHMARK_SAMPLE_LIMIT")
                if limit in (None, "", 0) and env_limit:
                    try:
                        limit = int(env_limit)
                    except ValueError:
                        limit = None

                mapping = {
                    "dataset": dataset["name"],
                    "dataset_arg": dataset.get("dataset_arg", dataset["name"]),
                    "dataset_label": dataset.get("label", dataset["name"]),
                    "variant": dataset.get("variant", ""),
                    "split": dataset.get("split", "dev"),
                    "dataset_extra_args": dataset.get("extra_args", ""),
                    "retriever_tag": dataset.get("retriever_tag", ""),
                    "dataset_limit": limit if limit not in (None, 0) else "",
                    "dataset_limit_arg": f"--limit {limit}" if limit not in (None, 0) else "",
                    "backbone": backbone["name"],
                    "backbone_label": backbone.get("label", backbone["name"]),
                    "model": backbone.get("model", ""),
                    "decoding_json": json.dumps(backbone.get("decoding", {})),
                    "summarizer_max_tokens": (backbone.get("decoding", {}) or {}).get("max_new_tokens", 256),
                    "backbone_extra_args": backbone.get("extra_args", ""),
                    "system": system["name"],
                    "system_label": system.get("label", system["name"]),
                    "system_extra_args": system.get("extra_args", ""),
                    "visual_caption_model": os.getenv("VISUAL_CAPTION_MODEL", ""),
                    "visual_caption_model_path": os.getenv("VISUAL_CAPTION_MODEL_PATH", ""),
                    "visual_ocr_engine": os.getenv("VISUAL_OCR_ENGINE", ""),
                    "visual_ocr_model_dir": os.getenv("VISUAL_OCR_MODEL_DIR", ""),
                }
                metrics_name = f"{dataset['name']}__{backbone['name']}__{system['name']}.json"
                metrics_path = metrics_dir / metrics_name
                mapping["metrics_path"] = str(metrics_path)
                mapping["metrics_arg"] = shlex.quote(str(metrics_path))
                mapping["decoding_arg"] = shlex.quote(mapping["decoding_json"])
                command = system["command"].format(**mapping)
                log_name = f"{dataset['name']}__{backbone['name']}__{system['name']}.log"
                jobs.append(
                    BenchmarkJob(
                        dataset=dataset,
                        backbone=backbone,
                        system=system,
                        command=command,
                        log_path=logs_dir / log_name,
                        metrics_path=metrics_path,
                        sample_limit=limit if limit not in (None, 0) else None,
                    )
                )
    return jobs


def _load_metrics(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # pragma: no cover - logging only
        return {"error": f"Failed to parse metrics: {exc}"}


def run_job(job: BenchmarkJob, *, dry_run: bool = False, cwd: Optional[Path] = None) -> Dict[str, Any]:
    env = os.environ.copy()
    env.update(job.dataset.get("env") or {})
    env.update(job.backbone.get("env") or {})
    env.update(job.system.get("env") or {})
    env["BENCHMARK_METRICS_FILE"] = str(job.metrics_path)

    record = {
        "dataset": job.dataset["name"],
        "dataset_label": job.dataset.get("label", job.dataset["name"]),
        "variant": job.dataset.get("variant", ""),
        "split": job.dataset.get("split", "dev"),
        "backbone": job.backbone["name"],
        "backbone_label": job.backbone.get("label", job.backbone["name"]),
        "model": job.backbone.get("model"),
        "system": job.system["name"],
        "system_label": job.system.get("label", job.system["name"]),
        "command": job.command,
        "log_file": str(job.log_path),
        "metrics_file": str(job.metrics_path),
        "limit": job.sample_limit,
        "status": "skipped" if dry_run else "pending",
    }

    if dry_run:
        return record

    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    job.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    start = dt.datetime.utcnow()
    try:
        result = subprocess.run(
            job.command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=str(cwd) if cwd else None,
            env=env,
        )
        duration = (dt.datetime.utcnow() - start).total_seconds()
        record["status"] = "ok" if result.returncode == 0 else "error"
        record["returncode"] = result.returncode
        record["duration_sec"] = duration
        record["metrics"] = _load_metrics(job.metrics_path)
        job.log_path.write_text(
            f"COMMAND: {job.command}\n"
            f"DATASET: {job.dataset['name']} ({job.dataset.get('split', 'dev')})\n"
            f"BACKBONE: {job.backbone.get('label', job.backbone['name'])}\n"
            f"SYSTEM: {job.system.get('label', job.system['name'])}\n"
            f"STATUS: {record['status']} (rc={result.returncode})\n"
            "----- STDOUT -----\n"
            f"{result.stdout}\n"
            "----- STDERR -----\n"
            f"{result.stderr}\n",
            encoding="utf-8",
        )
    except Exception as exc:  # pragma: no cover - best effort logging
        record["status"] = "exception"
        record["error"] = str(exc)
        job.log_path.write_text(
            f"COMMAND: {job.command}\n"
            f"EXCEPTION: {exc}\n",
            encoding="utf-8",
        )
    return record


def run_matrix(
    config_path: Path,
    *,
    max_workers: int,
    dry_run: bool,
    output_dir: Path,
    cwd: Optional[Path],
) -> Path:
    config = load_matrix(config_path)
    timestamp = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    results_dir = output_dir / f"benchmark_{timestamp}"
    logs_dir = results_dir / "logs"
    metrics_dir = results_dir / "metrics"
    results_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(config, logs_dir, metrics_dir)
    if not jobs:
        raise RuntimeError("No benchmark jobs to run. Check your config filters.")

    records: List[Dict[str, Any]] = []
    if dry_run:
        for job in jobs:
            records.append(run_job(job, dry_run=True))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_job = {executor.submit(run_job, job, dry_run=False, cwd=cwd): job for job in jobs}
            for future in as_completed(future_to_job):
                records.append(future.result())

    results_path = results_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    summary_path = results_dir / "summary.json"
    summary = {
        "generated_at": timestamp,
        "config": str(config_path),
        "num_jobs": len(jobs),
        "dry_run": dry_run,
        "results_file": str(results_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return results_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run planner baselines across datasets/backbones.")
    parser.add_argument("--config", type=Path, default=Path("configs/planner_benchmarks.yaml"))
    parser.add_argument("--max-workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--dry-run", action="store_true", help="Only print the resolved commands.")
    parser.add_argument("--cwd", type=Path, default=None, help="Working directory for launched commands.")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = run_matrix(
        args.config,
        max_workers=max(1, args.max_workers),
        dry_run=args.dry_run,
        output_dir=output_dir,
        cwd=args.cwd,
    )
    print(f"Benchmark matrix complete. Results saved to {results_path}")


if __name__ == "__main__":
    main()
