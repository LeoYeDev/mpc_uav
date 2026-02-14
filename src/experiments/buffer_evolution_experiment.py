"""
Visualize buffer point evolution during a full trajectory tracking run.

This script is intentionally separated from ablation/sensitivity scripts.
It records online-GP buffer snapshots at each control step and renders:
1) Per-method animation (FIFO / IVS)
2) Side-by-side comparison animation
3) Reusable trace files for later re-plotting without rerunning simulation

Example:
  python src/experiments/buffer_evolution_experiment.py \
    --methods fifo,ivs \
    --trajectory random \
    --speed 3.0 \
    --max-steps 250 \
    --frame-stride 2 \
    --format gif
"""

import argparse
import csv
import os
import pickle
import random
import time
from dataclasses import replace
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from config.configuration_parameters import SimpleSimConfig
from config.gp_config import OnlineGPConfig, build_online_gp_config
from config.paths import DEFAULT_MODEL_VERSION, DEFAULT_MODEL_NAME
from src.experiments.comparative_experiment import prepare_quadrotor_mpc
from src.gp.online import IncrementalGPManager
from src.gp.utils import world_to_body_velocity_mapping
from src.utils.quad_3d_opt_utils import get_reference_chunk
from src.utils.utils import interpol_mse, separate_variables
from src.utils.wind_model import RealisticWindModel
from src.utils.trajectories import random_trajectory, loop_trajectory, lemniscate_trajectory


DIM_LABELS = ["x", "y", "z"]
LEVEL_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]
MERGED_COLOR = "#D62728"


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


def _build_gp_config(method: str, args: argparse.Namespace) -> OnlineGPConfig:
    """
    构建在线 GP 配置（与核心 Online GP 模块保持严格同步）。

    这里不再在脚本内定义或覆盖 buffer 超参数，统一复用：
    `config/gp_config.py` 里的 `build_online_gp_config()`（其内部基于默认配置派生）。
    """
    method = method.lower().strip()
    base = build_online_gp_config(async_hp_updates=bool(args.async_updates))
    if method == "fifo":
        return replace(base, buffer_type="fifo")
    if method == "ivs":
        return replace(base, buffer_type="ivs")
    raise ValueError(f"Unsupported method: {method}")


def _generate_reference(
    quad,
    reference_type: str,
    speed: float,
    mpc_period: float,
    seed: int,
):
    if reference_type == "loop":
        return loop_trajectory(
            quad=quad,
            discretization_dt=mpc_period,
            radius=5,
            z=1,
            lin_acc=speed * 0.25,
            clockwise=True,
            yawing=False,
            v_max=speed,
            map_name=None,
            plot=False,
        )
    if reference_type == "lemniscate":
        return lemniscate_trajectory(
            quad=quad,
            discretization_dt=mpc_period,
            radius=5,
            z=1,
            lin_acc=speed * 0.25,
            clockwise=True,
            yawing=False,
            v_max=speed,
            map_name=None,
            plot=False,
        )
    return random_trajectory(
        quad=quad,
        discretization_dt=mpc_period,
        seed=seed,
        speed=speed,
        plot=False,
    )


def _capture_buffer_snapshot(manager: IncrementalGPManager, step: int, sim_time: float) -> Dict:
    dims = []
    for gp in manager.gps:
        # refresh merged cache for multilevel buffer
        _ = gp.buffer.get_training_set()
        merged = [(float(v), float(y), float(t)) for (v, y, t) in getattr(gp.buffer, "data", [])]
        diagnostics = {}
        if hasattr(gp.buffer, "get_diagnostics"):
            try:
                diagnostics = gp.buffer.get_diagnostics() or {}
            except Exception:
                diagnostics = {}
        levels = []
        if hasattr(gp.buffer, "levels"):
            for level in gp.buffer.levels:
                levels.append([(float(v), float(y), float(t)) for (v, y, t) in level.data])
        dims.append(
            {
                "merged": merged,
                "levels": levels,
                "trained": bool(gp.is_trained_once),
                "training_in_progress": bool(gp.is_training_in_progress),
                "selected_size": int(diagnostics.get("selected_size", len(merged))),
                "unique_ratio": float(diagnostics.get("unique_ratio", np.nan)),
                "duplicate_ratio": float(diagnostics.get("duplicate_ratio", np.nan)),
                "insert_accept_ratio": float(diagnostics.get("insert_accept_ratio", np.nan)),
                "insert_skip_ratio": float(diagnostics.get("insert_skip_ratio", np.nan)),
                "prune_old_count": float(diagnostics.get("prune_old_count", np.nan)),
                "flip_delete_count": float(diagnostics.get("flip_delete_count", np.nan)),
                "prune_old_count_last": float(diagnostics.get("prune_old_count_last", np.nan)),
                "flip_delete_count_last": float(diagnostics.get("flip_delete_count_last", np.nan)),
            }
        )
    return {"step": int(step), "time": float(sim_time), "dims": dims}


def run_buffer_trace(method: str, args: argparse.Namespace) -> Dict:
    _set_seed(int(args.seed))
    method_key = method.lower().strip()

    simulation_options = SimpleSimConfig.simulation_disturbances
    quad_name = f"my_quad_bufvis_{method_key}_{int(time.time())}"
    quad_mpc = prepare_quadrotor_mpc(
        simulation_options,
        version=DEFAULT_MODEL_VERSION,
        name=DEFAULT_MODEL_NAME,
        reg_type="gp",
        quad_name=quad_name,
        use_online_gp=True,
        solver_options={"variance_scaling_alpha": 0.0},
    )
    manager = IncrementalGPManager(config=_build_gp_config(method_key, args).to_dict())

    my_quad = quad_mpc.quad
    n_mpc_nodes = quad_mpc.n_nodes
    simulation_dt = quad_mpc.simulation_dt
    t_horizon = quad_mpc.t_horizon
    reference_over_sampling = 5
    mpc_period = t_horizon / (n_mpc_nodes * reference_over_sampling)
    wind_model = RealisticWindModel(profile=args.wind_profile)

    reference_traj, reference_timestamps, reference_u = _generate_reference(
        quad=my_quad,
        reference_type=args.trajectory,
        speed=float(args.speed),
        mpc_period=float(mpc_period),
        seed=int(args.seed),
    )
    max_steps = min(int(args.max_steps), int(reference_traj.shape[0] - 1))
    if max_steps <= 0:
        raise RuntimeError("max_steps must be >= 1 for visualization.")

    my_quad.set_state(reference_traj[0, :].tolist())
    ref_u = reference_u[0, :]
    current_idx = 0
    total_sim_time = 0.0
    x_pred = None
    snapshots: List[Dict] = []
    # 记录执行轨迹，用于计算与 comparative_experiment 一致口径的 RMSE。
    exec_traj = np.zeros((max_steps + 1, reference_traj.shape[1]), dtype=float)
    exec_traj[0, :] = my_quad.get_state(quaternion=True, stacked=True)

    try:
        while current_idx < max_steps:
            quad_current_state = my_quad.get_state(quaternion=True, stacked=True)
            ref_traj_chunk, ref_u_chunk = get_reference_chunk(
                reference_traj, reference_u, current_idx, n_mpc_nodes, reference_over_sampling
            )
            model_ind = quad_mpc.set_reference(
                x_reference=separate_variables(ref_traj_chunk),
                u_reference=ref_u_chunk,
            )

            online_predictions = None
            online_variances = None
            if x_pred is not None and any(gp.is_trained_once for gp in manager.gps):
                planned_states_body = world_to_body_velocity_mapping(x_pred)
                planned_velocities_body = planned_states_body[:, 7:10]
                online_predictions, online_variances = manager.predict(planned_velocities_body)

            w_opt, x_pred = quad_mpc.optimize(
                use_model=model_ind,
                return_x=True,
                online_gp_predictions=online_predictions,
                online_gp_variances=online_variances,
            )
            ref_u = np.squeeze(np.array(w_opt[:4]))

            s_before_sim = quad_mpc.get_state()
            v_body_in = world_to_body_velocity_mapping(s_before_sim.T)
            v_body_in = np.squeeze(v_body_in[:, 7:10])

            ext_v_k = wind_model.get_wind_velocity(total_sim_time)
            simulation_time = 0.0
            while simulation_time < mpc_period:
                simulation_time += simulation_dt
                total_sim_time += simulation_dt
                quad_mpc.simulate(ref_u, external_v=ext_v_k)

            s_after_sim = quad_mpc.get_state()
            v_body_out = world_to_body_velocity_mapping(s_after_sim.T)
            v_body_out = np.squeeze(v_body_out[:, 7:10])

            v_body_pred = quad_mpc.predict_model_step_accurately(
                current_state_np=s_before_sim,
                control_input_np=w_opt[:4],
                integration_period=simulation_time,
                use_model_idx=model_ind,
            )
            v_body_pred = world_to_body_velocity_mapping(np.expand_dims(v_body_pred, axis=0))
            v_body_pred = np.squeeze(v_body_pred[:, 7:10])

            residual_acc_body = (v_body_out - v_body_pred) / max(simulation_time, 1e-6)
            manager.update(v_body_in, residual_acc_body, timestamp=total_sim_time)
            manager.poll_for_results()

            snapshots.append(_capture_buffer_snapshot(manager, current_idx, total_sim_time))
            exec_traj[current_idx + 1, :] = my_quad.get_state(quaternion=True, stacked=True)
            current_idx += 1
    finally:
        manager.shutdown()

    eval_len = int(min(max_steps + 1, reference_traj.shape[0]))
    rmse = float(
        interpol_mse(
            reference_timestamps[:eval_len],
            reference_traj[:eval_len, :3],
            reference_timestamps[:eval_len],
            exec_traj[:eval_len, :3],
        )
    )
    max_vel = float(np.max(np.sqrt(np.sum(reference_traj[:eval_len, 7:10] ** 2, axis=1))))

    return {
        "method": method_key,
        "trajectory": args.trajectory,
        "speed": float(args.speed),
        "wind_profile": args.wind_profile,
        "max_steps": int(max_steps),
        "frame_stride": int(args.frame_stride),
        "seed": int(args.seed),
        "rmse": rmse,
        "max_vel": max_vel,
        "snapshots": snapshots,
    }


def _collect_axis_limits(trace: Dict) -> List[Tuple[float, float, float, float]]:
    limits = []
    for dim in range(3):
        xs, ys = [], []
        for snap in trace["snapshots"]:
            merged = snap["dims"][dim]["merged"]
            xs.extend([p[0] for p in merged])
            ys.extend([p[1] for p in merged])
        if xs:
            x_min, x_max = float(np.min(xs)), float(np.max(xs))
            y_min, y_max = float(np.min(ys)), float(np.max(ys))
            x_pad = 0.08 * max(1e-3, x_max - x_min)
            y_pad = 0.08 * max(1e-3, y_max - y_min)
            limits.append((x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad))
        else:
            limits.append((-1.0, 1.0, -1.0, 1.0))
    return limits


def _plot_dim_snapshot(ax, dim_snap: Dict, limits: Tuple[float, float, float, float], show_levels: bool) -> None:
    merged = dim_snap["merged"]
    levels = dim_snap["levels"]
    if show_levels and levels:
        for i, lvl in enumerate(levels):
            if not lvl:
                continue
            lv_x = [p[0] for p in lvl]
            lv_y = [p[1] for p in lvl]
            ax.scatter(
                lv_x,
                lv_y,
                s=18,
                alpha=0.30,
                color=LEVEL_COLORS[i % len(LEVEL_COLORS)],
                edgecolors="none",
                label=f"L{i}",
            )
    if merged:
        m_x = [p[0] for p in merged]
        m_y = [p[1] for p in merged]
        ax.scatter(
            m_x,
            m_y,
            s=28,
            alpha=0.90,
            color=MERGED_COLOR,
            edgecolors="black",
            linewidths=0.3,
            label="active",
        )
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.grid(True, alpha=0.25)


def _frame_indices(n_frames: int, stride: int) -> List[int]:
    idx = list(range(0, max(1, n_frames), max(1, stride)))
    if idx[-1] != n_frames - 1:
        idx.append(n_frames - 1)
    return idx


def _save_animation(anim_obj, out_stem: str, out_format: str, fps: int) -> List[str]:
    saved = []
    if out_format in ("gif", "both"):
        gif_path = f"{out_stem}.gif"
        anim_obj.save(gif_path, writer="pillow", fps=fps)
        saved.append(gif_path)
    if out_format in ("mp4", "both"):
        mp4_path = f"{out_stem}.mp4"
        try:
            writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
            anim_obj.save(mp4_path, writer=writer)
            saved.append(mp4_path)
        except Exception as exc:
            print(f"[warn] MP4 writer unavailable, skip mp4: {exc}")
    return saved


def render_method_animation(trace: Dict, out_dir: str, frame_stride: int, fps: int, out_format: str) -> List[str]:
    method = trace["method"]
    snapshots = trace["snapshots"]
    if not snapshots:
        return []
    frame_ids = _frame_indices(len(snapshots), frame_stride)
    limits = _collect_axis_limits(trace)
    show_levels = any(len(s["dims"][0]["levels"]) > 0 for s in snapshots)

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.6), sharex=False)
    fig.subplots_adjust(hspace=0.36)

    def _update(k: int):
        snap = snapshots[frame_ids[k]]
        for d, ax in enumerate(axes):
            ax.cla()
            _plot_dim_snapshot(ax, snap["dims"][d], limits[d], show_levels=show_levels)
            count = len(snap["dims"][d]["merged"])
            trained = snap["dims"][d]["trained"]
            ax.set_xlabel("Body velocity [m/s]")
            ax.set_ylabel("Residual [m/s²]")
            ax.set_title(
                f"{method.upper()} - dim {DIM_LABELS[d]} | points={count} | trained={trained}",
                fontsize=10,
            )
        fig.suptitle(
            f"Buffer Evolution ({method.upper()}) | step={snap['step']} | t={snap['time']:.2f}s",
            fontsize=12,
        )
        # legend only once
        if show_levels:
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                axes[0].legend(handles, labels, loc="upper right", frameon=False, fontsize=8)

    anim_obj = animation.FuncAnimation(fig, _update, frames=len(frame_ids), interval=1000 / max(1, fps))
    out_stem = os.path.join(out_dir, f"buffer_evolution_{method}")
    saved = _save_animation(anim_obj, out_stem, out_format, fps)
    plt.close(fig)
    return saved


def render_comparison_animation(
    traces: Dict[str, Dict],
    methods: List[str],
    out_dir: str,
    frame_stride: int,
    fps: int,
    out_format: str,
) -> List[str]:
    if len(methods) < 2:
        return []

    method_a, method_b = methods[0], methods[1]
    ta = traces[method_a]["snapshots"]
    tb = traces[method_b]["snapshots"]
    if not ta or not tb:
        return []

    frame_ids = _frame_indices(min(len(ta), len(tb)), frame_stride)
    limits_a = _collect_axis_limits(traces[method_a])
    limits_b = _collect_axis_limits(traces[method_b])
    show_levels_a = any(len(s["dims"][0]["levels"]) > 0 for s in ta)
    show_levels_b = any(len(s["dims"][0]["levels"]) > 0 for s in tb)

    fig, axes = plt.subplots(3, 2, figsize=(11.5, 8.8), sharex=False)
    fig.subplots_adjust(hspace=0.44, wspace=0.28)

    def _update(k: int):
        idx = frame_ids[k]
        sa = ta[idx]
        sb = tb[idx]
        for d in range(3):
            ax_l = axes[d, 0]
            ax_r = axes[d, 1]
            ax_l.cla()
            ax_r.cla()
            _plot_dim_snapshot(ax_l, sa["dims"][d], limits_a[d], show_levels_a)
            _plot_dim_snapshot(ax_r, sb["dims"][d], limits_b[d], show_levels_b)

            ca = len(sa["dims"][d]["merged"])
            cb = len(sb["dims"][d]["merged"])
            ax_l.set_title(f"{method_a.upper()} - dim {DIM_LABELS[d]} | points={ca}", fontsize=10)
            ax_r.set_title(f"{method_b.upper()} - dim {DIM_LABELS[d]} | points={cb}", fontsize=10)
            ax_l.set_xlabel("Body velocity [m/s]")
            ax_r.set_xlabel("Body velocity [m/s]")
            ax_l.set_ylabel("Residual [m/s²]")
            ax_r.set_ylabel("Residual [m/s²]")

        fig.suptitle(
            f"Buffer Evolution Comparison | step={sa['step']} | t≈{sa['time']:.2f}s",
            fontsize=12,
        )

    anim_obj = animation.FuncAnimation(fig, _update, frames=len(frame_ids), interval=1000 / max(1, fps))
    out_stem = os.path.join(out_dir, f"buffer_evolution_compare_{method_a}_vs_{method_b}")
    saved = _save_animation(anim_obj, out_stem, out_format, fps)
    plt.close(fig)
    return saved


def save_trace_files(trace: Dict, out_dir: str) -> List[str]:
    method = trace["method"]
    pkl_path = os.path.join(out_dir, f"buffer_trace_{method}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(trace, f)

    csv_path = os.path.join(out_dir, f"buffer_trace_counts_{method}.csv")
    max_levels = 0
    for snap in trace["snapshots"]:
        for d in snap["dims"]:
            max_levels = max(max_levels, len(d["levels"]))
    level_cols = [f"level{i}_count" for i in range(max_levels)]
    fieldnames = [
        "method",
        "step",
        "sim_time",
        "dim",
        "merged_count",
        "selected_size",
        "unique_ratio",
        "duplicate_ratio",
        "insert_accept_ratio",
        "insert_skip_ratio",
        "prune_old_count",
        "flip_delete_count",
        "prune_old_count_last",
        "flip_delete_count_last",
        "trained",
        "training_in_progress",
    ] + level_cols

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for snap in trace["snapshots"]:
            for d, dim_snap in enumerate(snap["dims"]):
                row = {
                    "method": method,
                    "step": snap["step"],
                    "sim_time": f"{snap['time']:.6f}",
                    "dim": DIM_LABELS[d],
                    "merged_count": len(dim_snap["merged"]),
                    "selected_size": int(dim_snap.get("selected_size", len(dim_snap["merged"]))),
                    "unique_ratio": (
                        f"{float(dim_snap.get('unique_ratio', np.nan)):.6f}"
                        if np.isfinite(float(dim_snap.get("unique_ratio", np.nan)))
                        else ""
                    ),
                    "duplicate_ratio": (
                        f"{float(dim_snap.get('duplicate_ratio', np.nan)):.6f}"
                        if np.isfinite(float(dim_snap.get("duplicate_ratio", np.nan)))
                        else ""
                    ),
                    "insert_accept_ratio": (
                        f"{float(dim_snap.get('insert_accept_ratio', np.nan)):.6f}"
                        if np.isfinite(float(dim_snap.get("insert_accept_ratio", np.nan)))
                        else ""
                    ),
                    "insert_skip_ratio": (
                        f"{float(dim_snap.get('insert_skip_ratio', np.nan)):.6f}"
                        if np.isfinite(float(dim_snap.get("insert_skip_ratio", np.nan)))
                        else ""
                    ),
                    "prune_old_count": (
                        f"{float(dim_snap.get('prune_old_count', np.nan)):.6f}"
                        if np.isfinite(float(dim_snap.get("prune_old_count", np.nan)))
                        else ""
                    ),
                    "flip_delete_count": (
                        f"{float(dim_snap.get('flip_delete_count', np.nan)):.6f}"
                        if np.isfinite(float(dim_snap.get("flip_delete_count", np.nan)))
                        else ""
                    ),
                    "prune_old_count_last": (
                        f"{float(dim_snap.get('prune_old_count_last', np.nan)):.6f}"
                        if np.isfinite(float(dim_snap.get("prune_old_count_last", np.nan)))
                        else ""
                    ),
                    "flip_delete_count_last": (
                        f"{float(dim_snap.get('flip_delete_count_last', np.nan)):.6f}"
                        if np.isfinite(float(dim_snap.get("flip_delete_count_last", np.nan)))
                        else ""
                    ),
                    "trained": int(dim_snap["trained"]),
                    "training_in_progress": int(dim_snap["training_in_progress"]),
                }
                for i in range(max_levels):
                    count = len(dim_snap["levels"][i]) if i < len(dim_snap["levels"]) else 0
                    row[f"level{i}_count"] = count
                writer.writerow(row)
    return [pkl_path, csv_path]


def save_rmse_summary(traces: Dict[str, Dict], methods: List[str], out_dir: str) -> str:
    """
    保存 FIFO/IVS 的 RMSE 对比摘要（CSV），便于论文表格直接引用。
    """
    csv_path = os.path.join(out_dir, "buffer_evolution_rmse_summary.csv")
    fieldnames = ["method", "rmse_m", "max_ref_speed_mps", "delta_vs_fifo_m", "improvement_vs_fifo_pct"]

    fifo_rmse = float(traces["fifo"]["rmse"]) if "fifo" in traces else np.nan
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method in methods:
            rmse = float(traces[method]["rmse"])
            delta = rmse - fifo_rmse if np.isfinite(fifo_rmse) else np.nan
            improve_pct = (
                (fifo_rmse - rmse) / fifo_rmse * 100.0
                if np.isfinite(fifo_rmse) and fifo_rmse > 1e-12
                else np.nan
            )
            writer.writerow(
                {
                    "method": method,
                    "rmse_m": f"{rmse:.6f}",
                    "max_ref_speed_mps": f"{float(traces[method]['max_vel']):.6f}",
                    "delta_vs_fifo_m": f"{delta:.6f}" if np.isfinite(delta) else "",
                    "improvement_vs_fifo_pct": f"{improve_pct:.3f}" if np.isfinite(improve_pct) else "",
                }
            )
    return csv_path


def parse_methods(methods_arg: str) -> List[str]:
    allowed = {"fifo", "ivs"}
    methods = [m.strip().lower() for m in methods_arg.split(",") if m.strip()]
    methods = [m for m in methods if m in allowed]
    if not methods:
        methods = ["fifo", "ivs"]
    # Keep order, remove duplicates
    dedup = []
    for m in methods:
        if m not in dedup:
            dedup.append(m)
    return dedup


def main():
    parser = argparse.ArgumentParser(description="可视化 FIFO/IVS 缓冲区点集演化，并输出 RMSE 对比结果。")
    # 基础实验配置：轨迹、风场、随机种子、输出格式等。
    parser.add_argument("--methods", type=str, default="fifo,ivs", help="对比方法（逗号分隔）：fifo,ivs")
    parser.add_argument("--trajectory", type=str, default="random", choices=["random", "loop", "lemniscate"],
                        help="参考轨迹类型：random / loop / lemniscate")
    parser.add_argument("--speed", type=float, default=3.0, help="轨迹目标速度（m/s）")
    parser.add_argument("--wind-profile", type=str, default="default", choices=["default", "regime_shift"],
                        help="风场模式：default 或 regime_shift")
    parser.add_argument("--seed", type=int, default=303, help="随机种子（用于可复现）")
    parser.add_argument("--max-steps", type=int, default=500, help="控制步数上限")
    parser.add_argument("--frame-stride", type=int, default=2, help="动画抽帧间隔（每 N 步取一帧）")
    parser.add_argument("--fps", type=int, default=12, help="动画帧率")
    parser.add_argument("--format", type=str, default="gif", choices=["gif", "mp4", "both"],
                        help="视频导出格式：gif / mp4 / both")
    parser.add_argument("--out-dir", type=str, default="outputs/figures/buffer_evolution",
                        help="输出目录（缓存快照、统计CSV、RMSE摘要、动图）")
    parser.add_argument("--no-video", action="store_true", help="仅导出数据，不渲染动画")

    # 在线训练模式开关：
    # 注意：除异步/同步外，其余 Online GP buffer 参数全部来自 build_online_gp_config，
    # 不再在本脚本中提供独立参数入口，确保与核心模块配置完全一致。
    parser.add_argument("--async-updates", action="store_true", default=True, help="异步在线训练（默认）")
    parser.add_argument("--sync-updates", action="store_true", help="同步在线训练（调试用）")

    args = parser.parse_args()
    if args.sync_updates:
        args.async_updates = False

    methods = parse_methods(args.methods)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    try:
        import multiprocessing
        multiprocessing.set_start_method("spawn", force=True)
    except (RuntimeError, ValueError):
        pass

    print("=" * 76)
    print("Buffer Evolution Visualization")
    print("=" * 76)
    print(
        f"Methods={methods}, Trajectory={args.trajectory}, Speed={args.speed}, "
        f"Wind={args.wind_profile}, Steps={args.max_steps}, "
        f"Async={args.async_updates}, Format={args.format}"
    )

    traces: Dict[str, Dict] = {}
    saved_paths: List[str] = []
    for method in methods:
        print(f"\n[run] method={method}")
        trace = run_buffer_trace(method, args)
        traces[method] = trace
        artifacts = save_trace_files(trace, out_dir)
        saved_paths.extend(artifacts)
        print(f"  saved trace: {artifacts[0]}")
        print(f"  saved counts: {artifacts[1]}")
        print(f"  RMSE: {trace['rmse']:.6f} m | MaxRefSpeed: {trace['max_vel']:.3f} m/s")

    rmse_summary_path = save_rmse_summary(traces, methods, out_dir)
    saved_paths.append(rmse_summary_path)
    print(f"  saved rmse summary: {rmse_summary_path}")

    if "fifo" in traces and "ivs" in traces:
        fifo_rmse = float(traces["fifo"]["rmse"])
        ivs_rmse = float(traces["ivs"]["rmse"])
        delta = ivs_rmse - fifo_rmse
        improve_pct = (fifo_rmse - ivs_rmse) / max(fifo_rmse, 1e-12) * 100.0
        print("\nRMSE comparison:")
        print(f"  FIFO RMSE: {fifo_rmse:.6f} m")
        print(f"  IVS  RMSE: {ivs_rmse:.6f} m")
        print(f"  IVS - FIFO: {delta:+.6f} m")
        print(f"  IVS improvement over FIFO: {improve_pct:+.3f}%")

    if not args.no_video:
        # 为减少运行时长，仅渲染 FIFO vs IVS 的总对比动画，不再输出单方法动画。
        if len(methods) >= 2:
            rendered = render_comparison_animation(
                traces=traces,
                methods=methods[:2],
                out_dir=out_dir,
                frame_stride=int(args.frame_stride),
                fps=int(args.fps),
                out_format=args.format,
            )
            saved_paths.extend(rendered)
            for p in rendered:
                print(f"  saved comparison: {p}")
        else:
            print("  skip video: methods < 2，无法生成对比动画。")

    print("\nDone. Generated files:")
    for p in saved_paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
