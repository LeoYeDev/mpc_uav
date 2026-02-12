"""
Sensitivity Experiment for AR-MPC (online GP components).

This script is separated from ablation_experiment.py and focuses on:
1) OAT (one-at-a-time) sensitivity under a fixed baseline.
2) Joint-parameter sweep (Sobol if available) for interaction effects.
3) Performance-latency trade-off figures for publication.

Typical usage:
1) Minimal smoke check:
   python src/experiments/sensitivity_experiment.py --mode smoke --max-runs 2 --no-plots
2) Quick sweep:
   python src/experiments/sensitivity_experiment.py --mode quick --studies novelty_decay,levels,distance,kernel
3) Full sweep for paper:
   python src/experiments/sensitivity_experiment.py --mode full --studies novelty_decay,levels,distance,kernel,sobol --seeds 5 --wind-profile regime_shift
"""

import argparse
import csv
import json
import os
from datetime import datetime
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None

from config.gp_config import OnlineGPConfig
from src.experiments.ablation_experiment import run_single_ablation
from src.visualization.style import set_publication_style, SCI_COLORS


def _allocate_level_capacities(total_size: int) -> List[int]:
    """Allocate 3-level capacities summing to total_size."""
    total_size = int(max(3, total_size))
    c0 = max(2, int(round(total_size * 0.55)))
    c1 = max(1, int(round(total_size * 0.30)))
    c2 = max(1, total_size - c0 - c1)

    while c0 + c1 + c2 > total_size:
        if c0 >= c1 and c0 >= c2 and c0 > 1:
            c0 -= 1
        elif c1 >= c2 and c1 > 1:
            c1 -= 1
        elif c2 > 1:
            c2 -= 1
        else:
            break
    while c0 + c1 + c2 < total_size:
        c0 += 1
    return [c0, c1, c2]


def _scale_capacities(template_caps: List[int], total_size: int) -> List[int]:
    """Scale template capacities to a new total size while keeping proportions."""
    total_size = int(max(3, total_size))
    if not template_caps:
        return _allocate_level_capacities(total_size)
    ratios = np.asarray(template_caps, dtype=float)
    ratios = ratios / np.sum(ratios)
    caps = np.maximum(1, np.round(ratios * total_size).astype(int))
    while int(np.sum(caps)) > total_size:
        idx = int(np.argmax(caps))
        if caps[idx] > 1:
            caps[idx] -= 1
        else:
            break
    while int(np.sum(caps)) < total_size:
        caps[0] += 1
    return caps.tolist()


def _study_values(mode: str) -> Dict[str, Dict]:
    if mode == "full":
        novelty_grid = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
        decay_grid = [0.03, 0.05, 0.07, 0.09, 0.12, 0.15, 0.19, 0.24, 0.30]
        levels_profiles = [
            {"name": "recent_heavy", "caps": [10, 3, 1], "sparsity": [1, 3, 6]},
            {"name": "balanced", "caps": [8, 4, 2], "sparsity": [1, 3, 6]},
            {"name": "long_memory", "caps": [6, 4, 4], "sparsity": [1, 2, 4]},
            {"name": "very_sparse_long", "caps": [7, 4, 3], "sparsity": [1, 4, 8]},
            {"name": "dense_three_level", "caps": [8, 4, 2], "sparsity": [1, 2, 3]},
            {"name": "coarse_three_level", "caps": [8, 4, 2], "sparsity": [1, 5, 10]},
            {"name": "mid_heavy", "caps": [7, 5, 2], "sparsity": [1, 2, 6]},
            {"name": "tail_heavy", "caps": [5, 4, 5], "sparsity": [1, 2, 5]},
        ]
        distance_vals = [0.003, 0.005, 0.007, 0.010, 0.013, 0.017, 0.022, 0.028, 0.036, 0.046]
        merge_ratios = [1.0, 1.15, 1.30, 1.45, 1.60, 1.80, 2.00]
        kernel_specs = [
            {"kernel": "rbf", "nu": 2.5},
            {"kernel": "matern12", "nu": 0.5},
            {"kernel": "matern32", "nu": 1.5},
            {"kernel": "matern52", "nu": 2.5},
            {"kernel": "matern_nu", "nu": 1.8},
        ]
        return {
            "novelty_decay": {"novelty": novelty_grid, "decay": decay_grid},
            "levels": {"profiles": levels_profiles},
            "distance": {"min_distance": distance_vals, "merge_ratio": merge_ratios},
            "kernel": {"kernels": kernel_specs},
            "sobol": {"n_points": 96},
        }

    if mode == "quick":
        return {
            "novelty_decay": {
                "novelty": [0.10, 0.25, 0.40, 0.55, 0.70],
                "decay": [0.05, 0.09, 0.13, 0.18, 0.25],
            },
            "levels": {
                "profiles": [
                    {"name": "recent_heavy", "caps": [9, 3, 2], "sparsity": [1, 3, 6]},
                    {"name": "balanced", "caps": [8, 4, 2], "sparsity": [1, 3, 6]},
                    {"name": "long_memory", "caps": [6, 4, 4], "sparsity": [1, 2, 4]},
                    {"name": "very_sparse_long", "caps": [7, 4, 3], "sparsity": [1, 4, 8]},
                ]
            },
            "distance": {"min_distance": [0.005, 0.010, 0.015, 0.022, 0.032], "merge_ratio": [1.0, 1.3, 1.6]},
            "kernel": {
                "kernels": [
                    {"kernel": "rbf", "nu": 2.5},
                    {"kernel": "matern32", "nu": 1.5},
                    {"kernel": "matern52", "nu": 2.5},
                    {"kernel": "matern_nu", "nu": 1.8},
                ]
            },
            "sobol": {"n_points": 24},
        }

    # smoke
    return {
        "novelty_decay": {"novelty": [0.25, 0.55], "decay": [0.08, 0.16]},
        "levels": {
            "profiles": [
                {"name": "balanced", "caps": [8, 4, 2], "sparsity": [1, 3, 6]},
                {"name": "long_memory", "caps": [6, 4, 4], "sparsity": [1, 2, 4]},
            ]
        },
        "distance": {"min_distance": [0.01, 0.03], "merge_ratio": [1.0, 1.5]},
        "kernel": {
            "kernels": [
                {"kernel": "rbf", "nu": 2.5},
                {"kernel": "matern32", "nu": 1.5},
            ]
        },
        "sobol": {"n_points": 8},
    }


def _make_run_budget(max_runs: int) -> Dict[str, Optional[int]]:
    if max_runs is None or int(max_runs) <= 0:
        return {"remaining": None}
    return {"remaining": int(max_runs)}


def _consume_budget(budget: Dict[str, Optional[int]]) -> bool:
    remaining = budget.get("remaining")
    if remaining is None:
        return True
    if remaining <= 0:
        return False
    budget["remaining"] = int(remaining) - 1
    return True


def _resolve_gp_devices(gp_device: str) -> Tuple[str, str]:
    """
    Resolve (main_process_device, worker_device_str).
    """
    device_req = str(gp_device).strip().lower()
    if device_req == "cpu":
        return "cpu", "cpu"

    cuda_ok = bool(torch is not None and torch.cuda.is_available())
    if device_req == "cuda" and not cuda_ok:
        print("[sensitivity] CUDA requested but unavailable; falling back to CPU.")
        return "cpu", "cpu"

    if device_req in ("auto", "cuda") and cuda_ok:
        return "cuda", "cuda"
    return "cpu", "cpu"


def _base_ivs_config(
    buffer_size: int = 14,
    train_mode: str = "sync",
    gp_device: str = "auto",
) -> OnlineGPConfig:
    """
    Shared baseline config for sensitivity runs.
    OAT studies vary one factor around this baseline.
    """
    buffer_size = int(buffer_size)
    min_points = min(buffer_size, max(6, int(round(buffer_size * 0.70))))
    refit_interval = max(4, int(round(buffer_size * 0.40)))
    use_async = train_mode == "async"
    main_dev, worker_dev = _resolve_gp_devices(gp_device)
    return OnlineGPConfig(
        buffer_type='ivs',
        variance_scaling_alpha=0.0,
        async_hp_updates=use_async,
        main_process_device=main_dev,
        worker_device_str=worker_dev,
        ivs_multilevel=True,
        buffer_max_size=buffer_size,
        min_points_for_initial_train=min_points,
        refit_hyperparams_interval=refit_interval,
        worker_train_iters=24,
        novelty_weight=0.35,
        recency_weight=0.65,
        recency_decay_rate=0.12,
        buffer_min_distance=0.02,
        buffer_merge_min_distance=0.025,
        buffer_level_capacities=_allocate_level_capacities(buffer_size),
        buffer_level_sparsity=[1, 3, 6],
        gp_kernel='rbf',
        gp_matern_nu=2.5,
    )


def _run_variant(
    variant_name: str,
    gp_cfg: OnlineGPConfig,
    speed: float,
    trajectory_type: str,
    wind_profile: str,
    seed_base: int,
    n_seeds: int,
    latency_metric: str,
) -> Dict:
    config = {
        "description": variant_name,
        "use_offline_gp": True,
        "use_online_gp": True,
        "gp_config": gp_cfg,
        "solver_options": {"variance_scaling_alpha": 0.0},
    }

    rmse_list, latency_list, max_vel_list = [], [], []
    opt_list, update_list, control_list = [], [], []
    for seed_offset in range(max(1, int(n_seeds))):
        seed = int(seed_base) + seed_offset
        result = run_single_ablation(
            config_name=f"sens_{variant_name}_s{seed}",
            config=config,
            speed=speed,
            trajectory_type=trajectory_type,
            seed=seed,
            wind_profile=wind_profile,
        )
        if result is None:
            continue
        rmse_list.append(float(result["rmse"]))
        max_vel_list.append(float(result["max_vel"]))
        opt_list.append(float(result.get("opt_time", np.nan)))
        update_list.append(float(result.get("gp_update_time", np.nan)))
        control_list.append(float(result.get("control_time", np.nan)))

        if latency_metric == "opt":
            latency_list.append(float(result.get("opt_time", np.nan)))
        elif latency_metric == "update":
            latency_list.append(float(result.get("gp_update_time", np.nan)))
        else:
            latency_list.append(float(result.get("control_time", result.get("opt_time", np.nan))))

    if not rmse_list:
        return {}

    return {
        "rmse_mean": float(np.mean(rmse_list)),
        "rmse_std": float(np.std(rmse_list)),
        "latency_mean": float(np.nanmean(latency_list)),
        "latency_std": float(np.nanstd(latency_list)),
        "opt_time_mean": float(np.nanmean(opt_list)),
        "opt_time_std": float(np.nanstd(opt_list)),
        "gp_update_time_mean": float(np.nanmean(update_list)),
        "gp_update_time_std": float(np.nanstd(update_list)),
        "control_time_mean": float(np.nanmean(control_list)),
        "control_time_std": float(np.nanstd(control_list)),
        "max_vel_mean": float(np.mean(max_vel_list)),
        "n_success": len(rmse_list),
    }


def _make_record(study: str, variant: str, cfg: OnlineGPConfig, metrics: Dict, **extra) -> Dict:
    caps = list(getattr(cfg, "buffer_level_capacities", []) or [])
    sparsity = list(getattr(cfg, "buffer_level_sparsity", []) or [])
    return {
        "study": study,
        "variant": variant,
        "novelty_weight": float(getattr(cfg, "novelty_weight", np.nan)),
        "recency_weight": float(getattr(cfg, "recency_weight", np.nan)),
        "recency_decay_rate": float(getattr(cfg, "recency_decay_rate", np.nan)),
        "buffer_max_size": int(getattr(cfg, "buffer_max_size", 0)),
        "gp_kernel": str(getattr(cfg, "gp_kernel", "rbf")),
        "gp_matern_nu": float(getattr(cfg, "gp_matern_nu", np.nan)),
        "level_profile": str(extra.get("level_profile", "")),
        "level_capacities": "-".join(str(int(c)) for c in caps) if caps else "",
        "level_sparsity": "-".join(str(int(s)) for s in sparsity) if sparsity else "",
        "buffer_min_distance": float(getattr(cfg, "buffer_min_distance", np.nan)),
        "buffer_merge_min_distance": float(getattr(cfg, "buffer_merge_min_distance", np.nan)),
        "train_mode": str(extra.get("train_mode", "sync")),
        "gp_device": str(extra.get("gp_device", "auto")),
        "latency_metric": str(extra.get("latency_metric", "control")),
        "sobol_index": int(extra.get("sobol_index", -1)),
        "sobol_source": str(extra.get("sobol_source", "")),
        "baseline_tag": "fixed_baseline_v1",
        **metrics,
    }


def run_novelty_decay_sensitivity(
    mode: str,
    speed: float,
    trajectory_type: str,
    wind_profile: str,
    seed_base: int,
    n_seeds: int,
    train_mode: str,
    gp_device: str,
    latency_metric: str,
    run_budget: Dict[str, Optional[int]],
) -> List[Dict]:
    vals = _study_values(mode)["novelty_decay"]
    records = []
    for novelty in vals["novelty"]:
        for decay in vals["decay"]:
            if not _consume_budget(run_budget):
                return records
            cfg = _base_ivs_config(buffer_size=14, train_mode=train_mode, gp_device=gp_device)
            cfg = replace(
                cfg,
                novelty_weight=float(novelty),
                recency_weight=float(1.0 - novelty),
                recency_decay_rate=float(decay),
            )
            name = f"nov{novelty:.2f}_dec{decay:.2f}"
            print(f"[novelty_decay] Running {name} ...")
            metrics = _run_variant(
                variant_name=name,
                gp_cfg=cfg,
                speed=speed,
                trajectory_type=trajectory_type,
                wind_profile=wind_profile,
                seed_base=seed_base,
                n_seeds=n_seeds,
                latency_metric=latency_metric,
            )
            if metrics:
                records.append(
                    _make_record(
                        "novelty_decay",
                        name,
                        cfg,
                        metrics,
                        train_mode=train_mode,
                        gp_device=gp_device,
                        latency_metric=latency_metric,
                    )
                )
                print(
                    f"  rmse={metrics['rmse_mean']:.4f}±{metrics['rmse_std']:.4f}, "
                    f"lat={metrics['latency_mean']*1000.0:.2f} ms"
                )
    return records


def run_levels_sensitivity(
    mode: str,
    speed: float,
    trajectory_type: str,
    wind_profile: str,
    seed_base: int,
    n_seeds: int,
    train_mode: str,
    gp_device: str,
    latency_metric: str,
    run_budget: Dict[str, Optional[int]],
) -> List[Dict]:
    profiles = _study_values(mode)["levels"]["profiles"]
    records = []

    for profile in profiles:
        if not _consume_budget(run_budget):
            return records
        caps = [int(v) for v in profile["caps"]]
        sparsity = [int(v) for v in profile["sparsity"]]
        size = int(sum(caps))
        cfg = _base_ivs_config(buffer_size=size, train_mode=train_mode, gp_device=gp_device)
        cfg = replace(
            cfg,
            buffer_type='ivs',
            ivs_multilevel=True,
            buffer_level_capacities=caps,
            buffer_level_sparsity=sparsity,
        )
        name = f"levels_{profile['name']}"
        print(f"[levels] Running {name} ...")
        metrics = _run_variant(
            variant_name=name,
            gp_cfg=cfg,
            speed=speed,
            trajectory_type=trajectory_type,
            wind_profile=wind_profile,
            seed_base=seed_base,
            n_seeds=n_seeds,
            latency_metric=latency_metric,
        )
        if metrics:
            records.append(
                _make_record(
                    "levels",
                    name,
                    cfg,
                    metrics,
                    level_profile=str(profile["name"]),
                    train_mode=train_mode,
                    gp_device=gp_device,
                    latency_metric=latency_metric,
                )
            )
            print(
                f"  rmse={metrics['rmse_mean']:.4f}±{metrics['rmse_std']:.4f}, "
                f"lat={metrics['latency_mean']*1000.0:.2f} ms"
            )
    return records


def run_distance_sensitivity(
    mode: str,
    speed: float,
    trajectory_type: str,
    wind_profile: str,
    seed_base: int,
    n_seeds: int,
    train_mode: str,
    gp_device: str,
    latency_metric: str,
    run_budget: Dict[str, Optional[int]],
) -> List[Dict]:
    vals = _study_values(mode)["distance"]
    min_distances = vals["min_distance"]
    merge_ratios = vals["merge_ratio"]
    records = []

    for min_d in min_distances:
        for ratio in merge_ratios:
            if not _consume_budget(run_budget):
                return records
            merge_d = float(min_d) * float(ratio)
            cfg = _base_ivs_config(buffer_size=14, train_mode=train_mode, gp_device=gp_device)
            cfg = replace(
                cfg,
                buffer_type='ivs',
                ivs_multilevel=True,
                buffer_min_distance=float(min_d),
                buffer_merge_min_distance=float(merge_d),
            )
            name = f"dist{min_d:.3f}_merge{merge_d:.3f}"
            print(f"[distance] Running {name} ...")
            metrics = _run_variant(
                variant_name=name,
                gp_cfg=cfg,
                speed=speed,
                trajectory_type=trajectory_type,
                wind_profile=wind_profile,
                seed_base=seed_base,
                n_seeds=n_seeds,
                latency_metric=latency_metric,
            )
            if metrics:
                records.append(
                    _make_record(
                        "distance",
                        name,
                        cfg,
                        metrics,
                        level_profile=f"merge_ratio_{ratio:.2f}",
                        train_mode=train_mode,
                        gp_device=gp_device,
                        latency_metric=latency_metric,
                    )
                )
                print(
                    f"  rmse={metrics['rmse_mean']:.4f}±{metrics['rmse_std']:.4f}, "
                    f"lat={metrics['latency_mean']*1000.0:.2f} ms"
                )
    return records


def run_kernel_sensitivity(
    mode: str,
    speed: float,
    trajectory_type: str,
    wind_profile: str,
    seed_base: int,
    n_seeds: int,
    train_mode: str,
    gp_device: str,
    latency_metric: str,
    run_budget: Dict[str, Optional[int]],
) -> List[Dict]:
    kernel_specs = _study_values(mode)["kernel"]["kernels"]
    records = []
    for spec in kernel_specs:
        if not _consume_budget(run_budget):
            return records
        kernel = str(spec["kernel"])
        nu = float(spec.get("nu", 2.5))
        cfg = _base_ivs_config(buffer_size=14, train_mode=train_mode, gp_device=gp_device)
        cfg = replace(cfg, gp_kernel=kernel, gp_matern_nu=nu)
        suffix = f"{kernel}_nu{nu:.2f}" if kernel == "matern_nu" else kernel
        name = f"kernel_{suffix}"
        print(f"[kernel] Running {name} ...")
        metrics = _run_variant(
            variant_name=name,
            gp_cfg=cfg,
            speed=speed,
            trajectory_type=trajectory_type,
            wind_profile=wind_profile,
            seed_base=seed_base,
            n_seeds=n_seeds,
            latency_metric=latency_metric,
        )
        if metrics:
            records.append(
                _make_record(
                    "kernel",
                    name,
                    cfg,
                    metrics,
                    train_mode=train_mode,
                    gp_device=gp_device,
                    latency_metric=latency_metric,
                )
            )
            print(
                f"  rmse={metrics['rmse_mean']:.4f}±{metrics['rmse_std']:.4f}, "
                f"lat={metrics['latency_mean']*1000.0:.2f} ms"
            )
    return records


def _unit_sobol_samples(n_points: int, dim: int, seed: int) -> Tuple[np.ndarray, str]:
    """
    Generate quasi-random points in [0,1]^d. Prefer Sobol, fallback to uniform RNG.
    """
    n_points = int(max(1, n_points))
    try:
        from scipy.stats import qmc
        m = int(np.ceil(np.log2(max(2, n_points))))
        sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
        unit = sampler.random_base2(m=m)[:n_points]
        return unit, "sobol"
    except Exception:
        rng = np.random.default_rng(seed)
        return rng.random((n_points, dim)), "uniform_fallback"


def run_sobol_sensitivity(
    mode: str,
    speed: float,
    trajectory_type: str,
    wind_profile: str,
    seed_base: int,
    n_seeds: int,
    train_mode: str,
    gp_device: str,
    latency_metric: str,
    run_budget: Dict[str, Optional[int]],
) -> List[Dict]:
    """
    Joint-parameter sweep to capture interaction effects.
    """
    n_points = int(_study_values(mode)["sobol"]["n_points"])
    level_profiles = _study_values(mode)["levels"]["profiles"]
    kernel_choices = [
        {"kernel": "rbf", "nu": 2.5},
        {"kernel": "matern32", "nu": 1.5},
        {"kernel": "matern52", "nu": 2.5},
    ]

    # dimensions:
    # [novelty, decay, min_distance, merge_ratio, buffer_size, level_profile_idx, kernel_idx]
    unit, source = _unit_sobol_samples(n_points=n_points, dim=7, seed=seed_base + 17)
    records = []
    for i, u in enumerate(unit):
        if not _consume_budget(run_budget):
            return records

        novelty = 0.05 + float(u[0]) * (0.85 - 0.05)
        decay = 0.03 + float(u[1]) * (0.35 - 0.03)
        min_distance = 0.003 + float(u[2]) * (0.046 - 0.003)
        merge_ratio = 1.0 + float(u[3]) * (2.0 - 1.0)
        merge_distance = min_distance * merge_ratio
        buffer_size = int(round(10 + float(u[4]) * (26 - 10)))

        p_idx = min(len(level_profiles) - 1, int(float(u[5]) * len(level_profiles)))
        k_idx = min(len(kernel_choices) - 1, int(float(u[6]) * len(kernel_choices)))
        profile = level_profiles[p_idx]
        kernel_spec = kernel_choices[k_idx]

        caps = _scale_capacities(profile["caps"], buffer_size)
        sparsity = [int(v) for v in profile["sparsity"]]

        cfg = _base_ivs_config(buffer_size=buffer_size, train_mode=train_mode, gp_device=gp_device)
        cfg = replace(
            cfg,
            novelty_weight=float(novelty),
            recency_weight=float(1.0 - novelty),
            recency_decay_rate=float(decay),
            buffer_min_distance=float(min_distance),
            buffer_merge_min_distance=float(merge_distance),
            buffer_level_capacities=caps,
            buffer_level_sparsity=sparsity,
            gp_kernel=str(kernel_spec["kernel"]),
            gp_matern_nu=float(kernel_spec.get("nu", 2.5)),
        )

        name = f"sobol_{i:03d}"
        print(f"[sobol] Running {name} ...")
        metrics = _run_variant(
            variant_name=name,
            gp_cfg=cfg,
            speed=speed,
            trajectory_type=trajectory_type,
            wind_profile=wind_profile,
            seed_base=seed_base,
            n_seeds=n_seeds,
            latency_metric=latency_metric,
        )
        if metrics:
            records.append(
                _make_record(
                    "sobol",
                    name,
                    cfg,
                    metrics,
                    level_profile=str(profile["name"]),
                    train_mode=train_mode,
                    gp_device=gp_device,
                    latency_metric=latency_metric,
                    sobol_index=i,
                    sobol_source=source,
                )
            )
            print(
                f"  rmse={metrics['rmse_mean']:.4f}±{metrics['rmse_std']:.4f}, "
                f"lat={metrics['latency_mean']*1000.0:.2f} ms"
            )
    return records


def save_records_csv(records: List[Dict], out_path: str) -> None:
    if not records:
        return
    fieldnames = [
        "study", "variant", "baseline_tag",
        "novelty_weight", "recency_weight", "recency_decay_rate",
        "buffer_max_size", "buffer_min_distance", "buffer_merge_min_distance",
        "level_profile", "level_capacities", "level_sparsity",
        "gp_kernel", "gp_matern_nu",
        "train_mode", "gp_device", "latency_metric",
        "sobol_index", "sobol_source",
        "rmse_mean", "rmse_std",
        "latency_mean", "latency_std",
        "opt_time_mean", "opt_time_std",
        "gp_update_time_mean", "gp_update_time_std",
        "control_time_mean", "control_time_std",
        "max_vel_mean", "n_success",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)


def save_records_json(records: List[Dict], out_path: str) -> None:
    if not records:
        return
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def _subset(records: List[Dict], study: str) -> List[Dict]:
    return [r for r in records if r.get("study") == study]


def plot_novelty_decay(records: List[Dict], out_dir: str) -> None:
    rows = _subset(records, "novelty_decay")
    if not rows:
        return

    novelty_vals = sorted({float(r["novelty_weight"]) for r in rows})
    decay_vals = sorted({float(r["recency_decay_rate"]) for r in rows})
    rmse_mat = np.full((len(novelty_vals), len(decay_vals)), np.nan, dtype=float)
    lat_mat = np.full((len(novelty_vals), len(decay_vals)), np.nan, dtype=float)

    for r in rows:
        i = novelty_vals.index(float(r["novelty_weight"]))
        j = decay_vals.index(float(r["recency_decay_rate"]))
        rmse_mat[i, j] = float(r["rmse_mean"])
        lat_mat[i, j] = float(r["latency_mean"]) * 1000.0

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    im0 = axes[0].imshow(rmse_mat, cmap="viridis_r", aspect="auto", origin="lower")
    im1 = axes[1].imshow(lat_mat, cmap="magma_r", aspect="auto", origin="lower")

    for ax in axes:
        ax.set_xticks(range(len(decay_vals)))
        ax.set_xticklabels([f"{v:.2f}" for v in decay_vals])
        ax.set_yticks(range(len(novelty_vals)))
        ax.set_yticklabels([f"{v:.2f}" for v in novelty_vals])
        ax.set_xlabel("Recency Decay")
        ax.set_ylabel("Novelty Weight")

    axes[0].set_title("RMSE Heatmap")
    axes[1].set_title("Latency Heatmap")
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar0.set_label("RMSE (m)")
    cbar1.set_label("Latency (ms)")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sensitivity_novelty_decay_heatmap.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_levels_tradeoff(records: List[Dict], out_dir: str) -> None:
    rows = _subset(records, "levels")
    if not rows:
        return

    rows = sorted(rows, key=lambda r: r["level_profile"])
    labels = [str(r["level_profile"]) for r in rows]
    rmse = np.array([float(r["rmse_mean"]) for r in rows], dtype=float)
    lat_ms = np.array([float(r["latency_mean"]) * 1000.0 for r in rows], dtype=float)

    fig, ax1 = plt.subplots(figsize=(7.2, 2.8))
    x = np.arange(len(labels))
    bars = ax1.bar(x, rmse, color=SCI_COLORS[1], alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylabel("RMSE (m)")
    ax1.set_title("Multi-level Layout Sensitivity")
    ax1.grid(axis="y")

    ax2 = ax1.twinx()
    ax2.plot(x, lat_ms, color=SCI_COLORS[3], marker="s", linewidth=1.2)
    ax2.set_ylabel("Latency (ms)")

    ax1.legend([bars, ax2.lines[0]], ["RMSE", "Latency"], loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sensitivity_levels_tradeoff.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_distance_sensitivity(records: List[Dict], out_dir: str) -> None:
    rows = _subset(records, "distance")
    if not rows:
        return

    ratio_to_rows: Dict[str, List[Dict]] = {}
    for r in rows:
        ratio = float(r["buffer_merge_min_distance"]) / max(float(r["buffer_min_distance"]), 1e-9)
        key = f"{ratio:.2f}"
        ratio_to_rows.setdefault(key, []).append(r)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for idx, (ratio, r_rows) in enumerate(sorted(ratio_to_rows.items(), key=lambda kv: float(kv[0]))):
        r_rows = sorted(r_rows, key=lambda r: float(r["buffer_min_distance"]))
        x = [float(r["buffer_min_distance"]) for r in r_rows]
        y_rmse = [float(r["rmse_mean"]) for r in r_rows]
        y_lat = [float(r["latency_mean"]) * 1000.0 for r in r_rows]
        color = SCI_COLORS[idx % len(SCI_COLORS)]
        label = f"merge/min={float(ratio):.2f}"
        axes[0].plot(x, y_rmse, marker="o", linewidth=1.2, color=color, label=label)
        axes[1].plot(x, y_lat, marker="o", linewidth=1.2, color=color, label=label)

    axes[0].set_xlabel("min_distance")
    axes[0].set_ylabel("RMSE (m)")
    axes[0].set_title("Distance vs RMSE")
    axes[0].grid(True)

    axes[1].set_xlabel("min_distance")
    axes[1].set_ylabel("Latency (ms)")
    axes[1].set_title("Distance vs Latency")
    axes[1].grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sensitivity_distance_tradeoff.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_kernel_tradeoff(records: List[Dict], out_dir: str) -> None:
    rows = _subset(records, "kernel")
    if not rows:
        return

    def _label(row: Dict) -> str:
        k = str(row["gp_kernel"]).upper()
        if str(row["gp_kernel"]) == "matern_nu":
            return f"MAT-NU({float(row['gp_matern_nu']):.1f})"
        return k

    rows = sorted(rows, key=lambda r: _label(r))
    labels = [_label(r) for r in rows]
    rmse = np.array([float(r["rmse_mean"]) for r in rows], dtype=float)
    rmse_std = np.array([float(r["rmse_std"]) for r in rows], dtype=float)
    lat_ms = np.array([float(r["latency_mean"]) * 1000.0 for r in rows], dtype=float)

    fig, ax1 = plt.subplots(figsize=(6.2, 2.8))
    x = np.arange(len(labels))
    bars = ax1.bar(x, rmse, yerr=rmse_std, capsize=3, color=SCI_COLORS[0], alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=10, ha="right")
    ax1.set_ylabel("RMSE (m)")
    ax1.set_xlabel("Kernel")
    ax1.set_title("Kernel Sensitivity")
    ax1.grid(axis="y")

    ax2 = ax1.twinx()
    ax2.plot(x, lat_ms, color=SCI_COLORS[3], marker="s", linewidth=1.2)
    ax2.set_ylabel("Latency (ms)")

    ax1.legend([bars, ax2.lines[0]], ["RMSE", "Latency"], loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sensitivity_kernel_tradeoff.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_sobol_scatter(records: List[Dict], out_dir: str) -> None:
    rows = _subset(records, "sobol")
    if not rows:
        return

    x = np.array([float(r["latency_mean"]) * 1000.0 for r in rows], dtype=float)
    y = np.array([float(r["rmse_mean"]) for r in rows], dtype=float)
    c = np.array([float(r["novelty_weight"]) for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    sc = ax.scatter(x, y, c=c, cmap="viridis", s=25, alpha=0.85)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Novelty Weight")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("RMSE (m)")
    ax.set_title("Sobol Sweep: RMSE-Latency")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sensitivity_sobol_scatter.pdf"), bbox_inches="tight")
    plt.close(fig)


def _rank_simple(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks


def _rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or y.size < 3:
        return np.nan
    xr = _rank_simple(x)
    yr = _rank_simple(y)
    if np.std(xr) < 1e-12 or np.std(yr) < 1e-12:
        return np.nan
    return float(np.corrcoef(xr, yr)[0, 1])


def plot_sobol_rankcorr(records: List[Dict], out_dir: str) -> None:
    rows = _subset(records, "sobol")
    if len(rows) < 3:
        return

    params = {
        "novelty": np.array([float(r["novelty_weight"]) for r in rows], dtype=float),
        "decay": np.array([float(r["recency_decay_rate"]) for r in rows], dtype=float),
        "buf_size": np.array([float(r["buffer_max_size"]) for r in rows], dtype=float),
        "min_dist": np.array([float(r["buffer_min_distance"]) for r in rows], dtype=float),
        "merge_dist": np.array([float(r["buffer_merge_min_distance"]) for r in rows], dtype=float),
    }
    rmse = np.array([float(r["rmse_mean"]) for r in rows], dtype=float)
    latency = np.array([float(r["latency_mean"]) for r in rows], dtype=float)

    labels = list(params.keys())
    corr_rmse = [abs(_rank_corr(params[k], rmse)) for k in labels]
    corr_lat = [abs(_rank_corr(params[k], latency)) for k in labels]

    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, corr_rmse, width=w, color=SCI_COLORS[0], alpha=0.85, label="|rank corr| vs RMSE")
    ax.bar(x + w / 2, corr_lat, width=w, color=SCI_COLORS[3], alpha=0.85, label="|rank corr| vs Latency")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Absolute Rank Correlation")
    ax.set_title("Sobol Sweep: Parameter Influence")
    ax.grid(axis="y")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sensitivity_sobol_rankcorr.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_pareto(records: List[Dict], out_dir: str) -> None:
    if not records:
        return
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    studies = sorted({str(r["study"]) for r in records})
    for i, study in enumerate(studies):
        s_rows = _subset(records, study)
        if not s_rows:
            continue
        x = [float(r["latency_mean"]) * 1000.0 for r in s_rows]
        y = [float(r["rmse_mean"]) for r in s_rows]
        ax.scatter(x, y, color=SCI_COLORS[i % len(SCI_COLORS)], alpha=0.85, s=22, label=study)

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("RMSE (m)")
    ax.set_title("Performance-Latency Pareto")
    ax.grid(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sensitivity_pareto.pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_studies(studies_arg: str) -> List[str]:
    allowed = {"novelty_decay", "levels", "distance", "kernel", "sobol"}
    parsed = [s.strip() for s in studies_arg.split(",") if s.strip()]
    parsed = [s for s in parsed if s in allowed]
    return parsed if parsed else ["novelty_decay", "levels", "distance", "kernel", "sobol"]


def main():
    parser = argparse.ArgumentParser(description="Sensitivity analysis for AR-MPC online GP components")
    parser.add_argument("--mode", type=str, default="smoke", choices=["smoke", "quick", "full"])
    parser.add_argument("--studies", type=str, default="novelty_decay,levels,distance,kernel,sobol",
                        help="Comma-separated: novelty_decay,levels,distance,kernel,sobol")
    parser.add_argument("--speed", type=float, default=2.7)
    parser.add_argument("--trajectory", type=str, default="random", choices=["random", "loop", "lemniscate"])
    parser.add_argument("--wind-profile", type=str, default="default", choices=["default", "regime_shift"])
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=303)
    parser.add_argument("--train-mode", type=str, default="async", choices=["sync", "async"],
                        help="sync: no worker process. async: asynchronous worker updates.")
    parser.add_argument("--gp-device", type=str, default="cpu", choices=["auto", "cpu", "cuda"],
                        help="GPU for online GP training/inference (if available).")
    parser.add_argument("--latency-metric", type=str, default="control", choices=["opt", "update", "control"],
                        help="Latency metric used in trade-off plots.")
    parser.add_argument("--max-runs", type=int, default=0,
                        help="Optional cap on total variants across selected studies (0 = no cap).")
    parser.add_argument("--out-dir", type=str, default="outputs/figures")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    selected = parse_studies(args.studies)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    run_budget = _make_run_budget(args.max_runs)

    set_publication_style(base_size=9)

    print("=" * 76)
    print("Sensitivity Experiment (OAT + Sobol)")
    print("=" * 76)
    print(
        f"Mode={args.mode}, Studies={selected}, Seeds={args.seeds}, "
        f"Wind={args.wind_profile}, Speed={args.speed}, "
        f"TrainMode={args.train_mode}, GPDevice={args.gp_device}, "
        f"Latency={args.latency_metric}, "
        f"MaxRuns={args.max_runs if args.max_runs > 0 else 'all'}"
    )

    records: List[Dict] = []
    if "novelty_decay" in selected:
        records.extend(
            run_novelty_decay_sensitivity(
                mode=args.mode,
                speed=args.speed,
                trajectory_type=args.trajectory,
                wind_profile=args.wind_profile,
                seed_base=args.seed_base,
                n_seeds=args.seeds,
                train_mode=args.train_mode,
                gp_device=args.gp_device,
                latency_metric=args.latency_metric,
                run_budget=run_budget,
            )
        )
    if "levels" in selected:
        records.extend(
            run_levels_sensitivity(
                mode=args.mode,
                speed=args.speed,
                trajectory_type=args.trajectory,
                wind_profile=args.wind_profile,
                seed_base=args.seed_base,
                n_seeds=args.seeds,
                train_mode=args.train_mode,
                gp_device=args.gp_device,
                latency_metric=args.latency_metric,
                run_budget=run_budget,
            )
        )
    if "distance" in selected:
        records.extend(
            run_distance_sensitivity(
                mode=args.mode,
                speed=args.speed,
                trajectory_type=args.trajectory,
                wind_profile=args.wind_profile,
                seed_base=args.seed_base,
                n_seeds=args.seeds,
                train_mode=args.train_mode,
                gp_device=args.gp_device,
                latency_metric=args.latency_metric,
                run_budget=run_budget,
            )
        )
    if "kernel" in selected:
        records.extend(
            run_kernel_sensitivity(
                mode=args.mode,
                speed=args.speed,
                trajectory_type=args.trajectory,
                wind_profile=args.wind_profile,
                seed_base=args.seed_base,
                n_seeds=args.seeds,
                train_mode=args.train_mode,
                gp_device=args.gp_device,
                latency_metric=args.latency_metric,
                run_budget=run_budget,
            )
        )
    if "sobol" in selected:
        records.extend(
            run_sobol_sensitivity(
                mode=args.mode,
                speed=args.speed,
                trajectory_type=args.trajectory,
                wind_profile=args.wind_profile,
                seed_base=args.seed_base,
                n_seeds=args.seeds,
                train_mode=args.train_mode,
                gp_device=args.gp_device,
                latency_metric=args.latency_metric,
                run_budget=run_budget,
            )
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(out_dir, f"sensitivity_results_{timestamp}.csv")
    json_path = os.path.join(out_dir, f"sensitivity_results_{timestamp}.json")
    save_records_csv(records, csv_path)
    save_records_json(records, json_path)
    print(f"\nSaved table: {csv_path}")
    print(f"Saved json:  {json_path}")

    if not args.no_plots:
        plot_novelty_decay(records, out_dir)
        plot_levels_tradeoff(records, out_dir)
        plot_distance_sensitivity(records, out_dir)
        plot_kernel_tradeoff(records, out_dir)
        plot_sobol_scatter(records, out_dir)
        plot_sobol_rankcorr(records, out_dir)
        plot_pareto(records, out_dir)
        print(f"Saved figures in: {out_dir}")

    if records:
        print("\nSummary by study:")
        for study in ["novelty_decay", "levels", "distance", "kernel", "sobol"]:
            s_rows = _subset(records, study)
            if not s_rows:
                continue
            best = min(s_rows, key=lambda r: r["rmse_mean"])
            print(
                f"  {study:<12} best={best['variant']:<18} "
                f"rmse={best['rmse_mean']:.4f}, lat={best['latency_mean']*1000.0:.2f} ms"
            )
    else:
        print("\nNo successful runs were recorded.")


if __name__ == "__main__":
    main()
