"""
Sensitivity Experiment for AR-MPC (online GP buffer simplification version).

This script is separated from ablation_experiment.py and focuses on:
1) OAT sensitivity around simplified IVS buffer parameters.
2) Joint Sobol/random sweep for interaction effects.
3) Publication-ready performance-latency trade-off figures.

Current sensitivity dimensions:
- buffer_insert_min_delta_v
- buffer_prune_old_delta_v
- buffer_flip_prune_limit
- buffer_max_size
- gp_kernel
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
except Exception:  # pragma: no cover
    torch = None

from config.gp_config import OnlineGPConfig, build_online_gp_config
from src.experiments.ablation_experiment import run_single_ablation
from src.visualization.style import set_publication_style, SCI_COLORS


def _study_values(mode: str) -> Dict[str, Dict]:
    """Return study grids by run mode."""
    if mode == "full":
        return {
            "thresholds": {
                "insert": [0.08, 0.10, 0.12, 0.14, 0.15, 0.17, 0.20, 0.23],
                "prune": [0.08, 0.10, 0.12, 0.14, 0.15, 0.17, 0.20, 0.23],
            },
            "flip_limit": {"values": [0, 1, 2, 3, 4, 5]},
            "buffer_size": {"values": [8, 10, 12, 13, 16, 20, 24]},
            "kernel": {
                "kernels": [
                    {"kernel": "rbf", "nu": 2.5},
                    {"kernel": "matern12", "nu": 0.5},
                    {"kernel": "matern32", "nu": 1.5},
                    {"kernel": "matern52", "nu": 2.5},
                    {"kernel": "matern_nu", "nu": 1.8},
                ]
            },
            "sobol": {"n_points": 96},
        }

    if mode == "quick":
        return {
            "thresholds": {
                "insert": [0.10, 0.13, 0.15, 0.18, 0.22],
                "prune": [0.10, 0.13, 0.15, 0.18, 0.22],
            },
            "flip_limit": {"values": [1, 2, 3, 4]},
            "buffer_size": {"values": [10, 13, 16, 20]},
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
        "thresholds": {
            "insert": [0.12, 0.18],
            "prune": [0.12, 0.18],
        },
        "flip_limit": {"values": [1, 3]},
        "buffer_size": {"values": [10, 13]},
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
    """Resolve (main_process_device, worker_device_str)."""
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
    buffer_size: int = 13,
    train_mode: str = "async",
    gp_device: str = "cpu",
) -> OnlineGPConfig:
    """Shared baseline config for sensitivity runs."""
    buffer_size = int(buffer_size)
    min_points = min(buffer_size, max(6, int(round(buffer_size * 0.70))))
    refit_interval = max(4, int(round(buffer_size * 0.45)))
    use_async = str(train_mode).lower() == "async"
    main_dev, worker_dev = _resolve_gp_devices(gp_device)
    return build_online_gp_config(
        buffer_type="ivs",
        variance_scaling_alpha=1.0,
        async_hp_updates=use_async,
        main_process_device=main_dev,
        worker_device_str=worker_dev,
        buffer_max_size=buffer_size,
        min_points_for_initial_train=min_points,
        refit_hyperparams_interval=refit_interval,
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
        "solver_options": {"variance_scaling_alpha": float(gp_cfg.variance_scaling_alpha)},
    }

    rmse_list, latency_list, max_vel_list = [], [], []
    opt_list, update_list, control_list = [], [], []
    predict_list, buffer_list, queue_list = [], [], []
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
        predict_list.append(float(result.get("gp_predict_time", np.nan)))
        buffer_list.append(float(result.get("buffer_update_time", np.nan)))
        queue_list.append(float(result.get("queue_overhead_time", np.nan)))

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
        "gp_predict_time_mean": float(np.nanmean(predict_list)),
        "gp_predict_time_std": float(np.nanstd(predict_list)),
        "buffer_update_time_mean": float(np.nanmean(buffer_list)),
        "buffer_update_time_std": float(np.nanstd(buffer_list)),
        "queue_overhead_time_mean": float(np.nanmean(queue_list)),
        "queue_overhead_time_std": float(np.nanstd(queue_list)),
        "max_vel_mean": float(np.mean(max_vel_list)),
        "n_success": len(rmse_list),
    }


def _make_record(study: str, variant: str, cfg: OnlineGPConfig, metrics: Dict, **extra) -> Dict:
    return {
        "study": study,
        "variant": variant,
        "buffer_insert_min_delta_v": float(getattr(cfg, "buffer_insert_min_delta_v", np.nan)),
        "buffer_prune_old_delta_v": float(getattr(cfg, "buffer_prune_old_delta_v", np.nan)),
        "buffer_flip_prune_limit": int(getattr(cfg, "buffer_flip_prune_limit", 0)),
        "buffer_max_size": int(getattr(cfg, "buffer_max_size", 0)),
        "gp_kernel": str(getattr(cfg, "gp_kernel", "rbf")),
        "gp_matern_nu": float(getattr(cfg, "gp_matern_nu", np.nan)),
        "kernel_index": int(extra.get("kernel_index", -1)),
        "train_mode": str(extra.get("train_mode", "async")),
        "gp_device": str(extra.get("gp_device", "cpu")),
        "latency_metric": str(extra.get("latency_metric", "control")),
        "sobol_index": int(extra.get("sobol_index", -1)),
        "sobol_source": str(extra.get("sobol_source", "")),
        "baseline_tag": "simplified_ivs_v1",
        **metrics,
    }


def run_threshold_sensitivity(
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
    vals = _study_values(mode)["thresholds"]
    records = []
    for insert_delta in vals["insert"]:
        for prune_delta in vals["prune"]:
            if not _consume_budget(run_budget):
                return records
            cfg = _base_ivs_config(buffer_size=13, train_mode=train_mode, gp_device=gp_device)
            cfg = replace(
                cfg,
                buffer_insert_min_delta_v=float(insert_delta),
                buffer_prune_old_delta_v=float(prune_delta),
            )
            name = f"thr_ins{insert_delta:.2f}_prn{prune_delta:.2f}"
            print(f"[thresholds] Running {name} ...")
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
                        "thresholds",
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


def run_flip_limit_sensitivity(
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
    values = _study_values(mode)["flip_limit"]["values"]
    records = []
    for flip_limit in values:
        if not _consume_budget(run_budget):
            return records
        cfg = _base_ivs_config(buffer_size=13, train_mode=train_mode, gp_device=gp_device)
        cfg = replace(cfg, buffer_flip_prune_limit=int(flip_limit))
        name = f"flip_{int(flip_limit)}"
        print(f"[flip_limit] Running {name} ...")
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
                    "flip_limit",
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


def run_buffer_size_sensitivity(
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
    values = _study_values(mode)["buffer_size"]["values"]
    records = []
    for size in values:
        if not _consume_budget(run_budget):
            return records
        cfg = _base_ivs_config(buffer_size=int(size), train_mode=train_mode, gp_device=gp_device)
        name = f"buf_{int(size)}"
        print(f"[buffer_size] Running {name} ...")
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
                    "buffer_size",
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
    for idx, spec in enumerate(kernel_specs):
        if not _consume_budget(run_budget):
            return records
        kernel = str(spec["kernel"])
        nu = float(spec.get("nu", 2.5))
        cfg = _base_ivs_config(buffer_size=13, train_mode=train_mode, gp_device=gp_device)
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
                    kernel_index=idx,
                )
            )
            print(
                f"  rmse={metrics['rmse_mean']:.4f}±{metrics['rmse_std']:.4f}, "
                f"lat={metrics['latency_mean']*1000.0:.2f} ms"
            )
    return records


def _unit_sobol_samples(n_points: int, dim: int, seed: int) -> Tuple[np.ndarray, str]:
    """Generate quasi-random points in [0,1]^d; Sobol preferred."""
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
    """Joint sweep to capture interactions under simplified parameter set."""
    n_points = int(_study_values(mode)["sobol"]["n_points"])
    kernel_choices = [
        {"kernel": "rbf", "nu": 2.5},
        {"kernel": "matern32", "nu": 1.5},
        {"kernel": "matern52", "nu": 2.5},
        {"kernel": "matern_nu", "nu": 1.8},
    ]

    # dimensions: [insert, prune, flip_limit, buffer_size, kernel_idx]
    unit, source = _unit_sobol_samples(n_points=n_points, dim=5, seed=seed_base + 17)
    records = []
    for i, u in enumerate(unit):
        if not _consume_budget(run_budget):
            return records

        insert_delta = 0.08 + float(u[0]) * (0.25 - 0.08)
        prune_delta = 0.08 + float(u[1]) * (0.25 - 0.08)
        flip_limit = int(np.clip(np.round(float(u[2]) * 5.0), 0, 5))
        buffer_size = int(np.clip(np.round(8 + float(u[3]) * (24 - 8)), 8, 24))
        kernel_idx = int(min(len(kernel_choices) - 1, np.floor(float(u[4]) * len(kernel_choices))))
        kernel_spec = kernel_choices[kernel_idx]

        cfg = _base_ivs_config(buffer_size=buffer_size, train_mode=train_mode, gp_device=gp_device)
        cfg = replace(
            cfg,
            buffer_insert_min_delta_v=float(insert_delta),
            buffer_prune_old_delta_v=float(prune_delta),
            buffer_flip_prune_limit=int(flip_limit),
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
                    train_mode=train_mode,
                    gp_device=gp_device,
                    latency_metric=latency_metric,
                    sobol_index=i,
                    sobol_source=source,
                    kernel_index=int(kernel_idx),
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
        "buffer_insert_min_delta_v", "buffer_prune_old_delta_v", "buffer_flip_prune_limit",
        "buffer_max_size", "gp_kernel", "gp_matern_nu", "kernel_index",
        "train_mode", "gp_device", "latency_metric",
        "sobol_index", "sobol_source",
        "rmse_mean", "rmse_std",
        "latency_mean", "latency_std",
        "opt_time_mean", "opt_time_std",
        "gp_predict_time_mean", "gp_predict_time_std",
        "gp_update_time_mean", "gp_update_time_std",
        "buffer_update_time_mean", "buffer_update_time_std",
        "queue_overhead_time_mean", "queue_overhead_time_std",
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


def plot_threshold_heatmap(records: List[Dict], out_dir: str) -> None:
    rows = _subset(records, "thresholds")
    if not rows:
        return

    insert_vals = sorted({float(r["buffer_insert_min_delta_v"]) for r in rows})
    prune_vals = sorted({float(r["buffer_prune_old_delta_v"]) for r in rows})
    rmse_mat = np.full((len(insert_vals), len(prune_vals)), np.nan, dtype=float)
    lat_mat = np.full((len(insert_vals), len(prune_vals)), np.nan, dtype=float)

    for r in rows:
        i = insert_vals.index(float(r["buffer_insert_min_delta_v"]))
        j = prune_vals.index(float(r["buffer_prune_old_delta_v"]))
        rmse_mat[i, j] = float(r["rmse_mean"])
        lat_mat[i, j] = float(r["latency_mean"]) * 1000.0

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    im0 = axes[0].imshow(rmse_mat, cmap="viridis_r", aspect="auto", origin="lower")
    im1 = axes[1].imshow(lat_mat, cmap="magma_r", aspect="auto", origin="lower")

    for ax in axes:
        ax.set_xticks(range(len(prune_vals)))
        ax.set_xticklabels([f"{v:.2f}" for v in prune_vals])
        ax.set_yticks(range(len(insert_vals)))
        ax.set_yticklabels([f"{v:.2f}" for v in insert_vals])
        ax.set_xlabel("prune_old_delta_v")
        ax.set_ylabel("insert_min_delta_v")

    axes[0].set_title("RMSE Heatmap")
    axes[1].set_title("Latency Heatmap")
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar0.set_label("RMSE (m)")
    cbar1.set_label("Latency (ms)")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sensitivity_thresholds_heatmap.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_flip_limit_tradeoff(records: List[Dict], out_dir: str) -> None:
    rows = _subset(records, "flip_limit")
    if not rows:
        return

    rows = sorted(rows, key=lambda r: int(r["buffer_flip_prune_limit"]))
    x = np.array([int(r["buffer_flip_prune_limit"]) for r in rows], dtype=int)
    rmse = np.array([float(r["rmse_mean"]) for r in rows], dtype=float)
    lat_ms = np.array([float(r["latency_mean"]) * 1000.0 for r in rows], dtype=float)

    fig, ax1 = plt.subplots(figsize=(6.6, 2.8))
    ax1.plot(x, rmse, color=SCI_COLORS[0], marker="o", linewidth=1.3)
    ax1.set_xlabel("buffer_flip_prune_limit")
    ax1.set_ylabel("RMSE (m)")
    ax1.set_title("Flip-Prune Limit Sensitivity")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(x, lat_ms, color=SCI_COLORS[3], marker="s", linewidth=1.2)
    ax2.set_ylabel("Latency (ms)")

    ax1.legend([ax1.lines[0], ax2.lines[0]], ["RMSE", "Latency"], loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sensitivity_flip_limit_tradeoff.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_buffer_size_tradeoff(records: List[Dict], out_dir: str) -> None:
    rows = _subset(records, "buffer_size")
    if not rows:
        return

    rows = sorted(rows, key=lambda r: int(r["buffer_max_size"]))
    x = np.array([int(r["buffer_max_size"]) for r in rows], dtype=int)
    rmse = np.array([float(r["rmse_mean"]) for r in rows], dtype=float)
    lat_ms = np.array([float(r["latency_mean"]) * 1000.0 for r in rows], dtype=float)

    fig, ax1 = plt.subplots(figsize=(6.8, 2.8))
    ax1.plot(x, rmse, color=SCI_COLORS[1], marker="o", linewidth=1.3)
    ax1.set_xlabel("buffer_max_size")
    ax1.set_ylabel("RMSE (m)")
    ax1.set_title("Buffer Size Sensitivity")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(x, lat_ms, color=SCI_COLORS[3], marker="s", linewidth=1.2)
    ax2.set_ylabel("Latency (ms)")

    ax1.legend([ax1.lines[0], ax2.lines[0]], ["RMSE", "Latency"], loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sensitivity_buffer_size_tradeoff.pdf"), bbox_inches="tight")
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
    c = np.array([float(r["buffer_max_size"]) for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    sc = ax.scatter(x, y, c=c, cmap="viridis", s=25, alpha=0.85)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("buffer_max_size")
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
        "insert_dv": np.array([float(r["buffer_insert_min_delta_v"]) for r in rows], dtype=float),
        "prune_dv": np.array([float(r["buffer_prune_old_delta_v"]) for r in rows], dtype=float),
        "flip_lim": np.array([float(r["buffer_flip_prune_limit"]) for r in rows], dtype=float),
        "buf_size": np.array([float(r["buffer_max_size"]) for r in rows], dtype=float),
        "kernel_i": np.array([float(r.get("kernel_index", -1)) for r in rows], dtype=float),
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
    """
    Parse study list with backward-compatible aliases:
    - novelty_decay / distance -> thresholds
    - levels -> buffer_size
    """
    aliases = {
        "novelty_decay": "thresholds",
        "distance": "thresholds",
        "levels": "buffer_size",
    }
    allowed = {"thresholds", "flip_limit", "buffer_size", "kernel", "sobol"}
    parsed_raw = [s.strip() for s in studies_arg.split(",") if s.strip()]
    parsed = []
    for s in parsed_raw:
        normalized = aliases.get(s, s)
        if normalized in allowed and normalized not in parsed:
            parsed.append(normalized)
    return parsed if parsed else ["thresholds", "flip_limit", "buffer_size", "kernel", "sobol"]


def main():
    parser = argparse.ArgumentParser(description="Sensitivity analysis for AR-MPC online GP simplified IVS buffer")
    parser.add_argument("--mode", type=str, default="smoke", choices=["smoke", "quick", "full"])
    parser.add_argument(
        "--studies",
        type=str,
        default="thresholds,flip_limit,buffer_size,kernel,sobol",
        help="Comma-separated: thresholds,flip_limit,buffer_size,kernel,sobol",
    )
    parser.add_argument("--speed", type=float, default=3.0)
    parser.add_argument("--trajectory", type=str, default="random", choices=["random", "loop", "lemniscate"])
    parser.add_argument("--wind-profile", type=str, default="default", choices=["default", "regime_shift"])
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=303)
    parser.add_argument(
        "--train-mode",
        type=str,
        default="async",
        choices=["sync", "async"],
        help="sync: no worker process. async: asynchronous worker updates.",
    )
    parser.add_argument(
        "--gp-device",
        type=str,
        default="cpu",
        choices=["auto", "cpu", "cuda"],
        help="GPU for online GP training/inference (if available).",
    )
    parser.add_argument(
        "--latency-metric",
        type=str,
        default="control",
        choices=["opt", "update", "control"],
        help="Latency metric used in trade-off plots.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Optional cap on total variants across selected studies (0 = no cap).",
    )
    parser.add_argument("--out-dir", type=str, default="outputs/figures")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    selected = parse_studies(args.studies)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    run_budget = _make_run_budget(args.max_runs)

    set_publication_style(base_size=9)

    print("=" * 76)
    print("Sensitivity Experiment (simplified IVS)")
    print("=" * 76)
    print(
        f"Mode={args.mode}, Studies={selected}, Seeds={args.seeds}, "
        f"Wind={args.wind_profile}, Speed={args.speed}, "
        f"TrainMode={args.train_mode}, GPDevice={args.gp_device}, "
        f"Latency={args.latency_metric}, "
        f"MaxRuns={args.max_runs if args.max_runs > 0 else 'all'}"
    )

    records: List[Dict] = []
    if "thresholds" in selected:
        records.extend(
            run_threshold_sensitivity(
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
    if "flip_limit" in selected:
        records.extend(
            run_flip_limit_sensitivity(
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
    if "buffer_size" in selected:
        records.extend(
            run_buffer_size_sensitivity(
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
        plot_threshold_heatmap(records, out_dir)
        plot_flip_limit_tradeoff(records, out_dir)
        plot_buffer_size_tradeoff(records, out_dir)
        plot_kernel_tradeoff(records, out_dir)
        plot_sobol_scatter(records, out_dir)
        plot_sobol_rankcorr(records, out_dir)
        plot_pareto(records, out_dir)
        print(f"Saved figures in: {out_dir}")

    if records:
        print("\nSummary by study:")
        for study in ["thresholds", "flip_limit", "buffer_size", "kernel", "sobol"]:
            s_rows = _subset(records, study)
            if not s_rows:
                continue
            best = min(s_rows, key=lambda r: r["rmse_mean"])
            print(
                f"  {study:<12} best={best['variant']:<22} "
                f"rmse={best['rmse_mean']:.4f}, lat={best['latency_mean']*1000.0:.2f} ms"
            )
    else:
        print("\nNo successful runs were recorded.")


if __name__ == "__main__":
    main()
