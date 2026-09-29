"""
Experiment 02: Baseline Connectome-Constrained LIF Simulation & Parameter Sensitivity Sweep.

Simulates Leaky Integrate-and-Fire dynamics on the real 1,000-neuron Drosophila connectome subnetwork.
Measures baseline activity, stability, and sensitivity to synaptic weight scaling.
Generates publication-quality raster plots, population rate profiles, heatmaps, and sweep tables.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from src.data_loader import extract_real_subnetwork
from src.simulation import LIFConfig, LIFNetwork, SimulationResult, run_simulation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_baseline_figures(
    result: SimulationResult,
    config: LIFConfig,
    figures_dir: Path,
) -> None:
    """
    Generates all required publication-quality figures:
    1. baseline_raster.png
    2. population_firing_rate.png
    3. firing_rate_distribution.png
    4. network_activity_heatmap.png
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Raster Plot: x=time (ms), y=neuron ID
    plt.figure(figsize=(10, 6), dpi=300)
    plt.style.use("tableau-colorblind10")
    if result.spike_times:
        times, nids = zip(*result.spike_times)
        # Map neuron IDs to contiguous rank for clear visualization
        id_map = {nid: rank for rank, nid in enumerate(result.neuron_ids)}
        y_vals = [id_map[nid] for nid in nids]
        plt.scatter(times, y_vals, s=2.5, c="#1f77b4", alpha=0.7, edgecolors="none")
    plt.title(
        f"Connectome-Constrained LIF Spike Raster (N={len(result.neuron_ids)}, scale={config.synaptic_weight_scale})",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    plt.xlabel("Time (ms)", fontsize=11)
    plt.ylabel("Neuron Index (sorted by connectivity degree)", fontsize=11)
    plt.xlim(0, config.duration)
    plt.ylim(-10, len(result.neuron_ids) + 10)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    raster_path = figures_dir / "baseline_raster.png"
    plt.savefig(raster_path, dpi=300)
    plt.close()
    logger.info(f"Saved raster plot to: {raster_path}")

    # 2. Population Firing Rate Plot: x=time (ms), y=spikes/sec per neuron (Hz)
    plt.figure(figsize=(10, 4.5), dpi=300)
    plt.plot(result.bin_centers, result.population_rate, color="#d62728", linewidth=1.8, label="Population Rate")
    mean_pop = np.mean(result.population_rate) if len(result.population_rate) > 0 else 0.0
    plt.axhline(mean_pop, color="#2ca02c", linestyle="--", linewidth=1.5, label=f"Mean: {mean_pop:.2f} Hz")
    if config.external_stimulus_mode == "pulse":
        plt.axvspan(config.pulse_start, config.pulse_end, color="#ffbb78", alpha=0.3, label="Stimulus ON Period")
    plt.title(
        f"Population Firing Rate Over Time (Bin Width = 10 ms | Mode: {config.external_stimulus_mode})",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    plt.xlabel("Time (ms)", fontsize=11)
    plt.ylabel("Population Rate (spikes/s/neuron)", fontsize=11)
    plt.xlim(0, config.duration)
    plt.ylim(bottom=0)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    pop_path = figures_dir / "population_firing_rate.png"
    plt.savefig(pop_path, dpi=300)
    plt.close()
    logger.info(f"Saved population firing rate plot to: {pop_path}")

    # 3. Firing Rate Distribution (Histogram)
    plt.figure(figsize=(8, 5), dpi=300)
    rates = result.firing_rates
    plt.hist(rates, bins=35, color="#2b5c8f", edgecolor="#142c47", alpha=0.85)
    plt.axvline(np.mean(rates), color="#d62728", linestyle="--", linewidth=1.8, label=f"Mean: {np.mean(rates):.1f} Hz")
    plt.axvline(np.median(rates), color="#2ca02c", linestyle=":", linewidth=1.8, label=f"Median: {np.median(rates):.1f} Hz")
    plt.title("Per-Neuron Firing Rate Distribution", fontsize=13, fontweight="bold", pad=10)
    plt.xlabel("Firing Rate (Hz)", fontsize=11)
    plt.ylabel("Number of Neurons", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(frameon=True)
    plt.tight_layout()
    dist_path = figures_dir / "firing_rate_distribution.png"
    plt.savefig(dist_path, dpi=300)
    plt.close()
    logger.info(f"Saved firing rate distribution to: {dist_path}")

    # 4. Network Activity Heatmap (2D matrix of time bins vs neuron index)
    plt.figure(figsize=(10, 6), dpi=300)
    bin_width_ms = 20.0
    n_bins = int(np.ceil(config.duration / bin_width_ms))
    heatmap_matrix = np.zeros((len(result.neuron_ids), n_bins), dtype=np.float32)

    id_to_idx = {nid: idx for idx, nid in enumerate(result.neuron_ids)}
    for t_spike, nid in result.spike_times:
        b = min(int(t_spike / bin_width_ms), n_bins - 1)
        neuron_idx = id_to_idx[nid]
        heatmap_matrix[neuron_idx, b] += 1.0

    # Convert counts per bin to Hz
    heatmap_matrix = heatmap_matrix / (bin_width_ms / 1000.0)

    im = plt.imshow(
        heatmap_matrix,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[0, config.duration, 0, len(result.neuron_ids)],
    )
    cbar = plt.colorbar(im)
    cbar.set_label("Firing Rate (Hz)", fontsize=11)
    plt.title(
        f"Network Spatio-Temporal Activity Heatmap (Bin = {bin_width_ms:.0f} ms)",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    plt.xlabel("Time (ms)", fontsize=11)
    plt.ylabel("Neuron Index", fontsize=11)
    if config.external_stimulus_mode == "pulse":
        plt.axvline(config.pulse_start, color="white", linestyle="--", alpha=0.7)
        plt.axvline(config.pulse_end, color="white", linestyle="--", alpha=0.7)
    plt.tight_layout()
    heatmap_path = figures_dir / "network_activity_heatmap.png"
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    logger.info(f"Saved network activity heatmap to: {heatmap_path}")


def run_parameter_sweep(
    network: LIFNetwork,
    scales: List[float],
    base_config: LIFConfig,
    tables_dir: Path,
    figures_dir: Path,
) -> pd.DataFrame:
    """
    Executes parameter sweep over synaptic_weight_scale values.
    Saves results to weight_scale_sweep.csv and renders weight_scale_vs_activity.png.
    """
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    sweep_records = []

    logger.info("=" * 65)
    logger.info(" STARTING SYNAPTIC WEIGHT SCALE PARAMETER SWEEP")
    logger.info("=" * 65)

    for scale in scales:
        cfg = LIFConfig(
            tau_m=base_config.tau_m,
            v_rest=base_config.v_rest,
            v_reset=base_config.v_reset,
            v_threshold=base_config.v_threshold,
            t_ref=base_config.t_ref,
            dt=base_config.dt,
            duration=base_config.duration,
            synaptic_weight_scale=scale,
            external_stimulus_mode=base_config.external_stimulus_mode,
            external_current=base_config.external_current,
            noise_sigma=base_config.noise_sigma,
            pulse_start=base_config.pulse_start,
            pulse_end=base_config.pulse_end,
            stimulus_fraction=base_config.stimulus_fraction,
            random_seed=base_config.random_seed,
            max_rate_threshold=base_config.max_rate_threshold,
        )

        t0 = time.perf_counter()
        res = network.simulate(cfg, record_samples=0)
        dur = time.perf_counter() - t0

        m = res.metrics
        status = "Stable" if res.is_stable else "Unstable"

        logger.info(
            f"Scale: {scale:<6.3f} | Spikes: {m['total_spikes']:<7d} | "
            f"Mean Rate: {m['mean_firing_rate_hz']:<6.2f} Hz | "
            f"Active: {m['active_neuron_fraction']*100:<5.1f}% | "
            f"Max Rate: {m['max_firing_rate_hz']:<6.1f} Hz | "
            f"Status: {status} ({dur:.2f}s)"
        )

        sweep_records.append(
            {
                "synaptic_weight_scale": scale,
                "total_spikes": m["total_spikes"],
                "spikes_per_neuron": m["spikes_per_neuron"],
                "mean_firing_rate_hz": m["mean_firing_rate_hz"],
                "median_firing_rate_hz": m["median_firing_rate_hz"],
                "max_firing_rate_hz": m["max_firing_rate_hz"],
                "active_neuron_fraction": m["active_neuron_fraction"],
                "is_stable": res.is_stable,
                "status_message": res.status_message,
                "runtime_seconds": round(dur, 3),
            }
        )

    sweep_df = pd.DataFrame(sweep_records)
    csv_path = tables_dir / "weight_scale_sweep.csv"
    sweep_df.to_csv(csv_path, index=False)
    logger.info(f"Saved parameter sweep results to: {csv_path}")

    # Generate weight_scale_vs_activity.png
    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    color_mean = "#1f77b4"
    color_max = "#d62728"
    color_active = "#2ca02c"

    scales_arr = sweep_df["synaptic_weight_scale"].values
    mean_rates = sweep_df["mean_firing_rate_hz"].values
    max_rates = sweep_df["max_firing_rate_hz"].values
    active_fractions = sweep_df["active_neuron_fraction"].values * 100.0

    ax1.plot(scales_arr, mean_rates, marker="o", linewidth=2.0, color=color_mean, label="Mean Firing Rate (Hz)")
    ax1.plot(scales_arr, max_rates, marker="s", linestyle="--", linewidth=1.8, color=color_max, label="Max Firing Rate (Hz)")
    ax1.set_xlabel("Synaptic Weight Scale", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Firing Rate (Hz)", fontsize=11, color=color_mean, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_mean)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(scales_arr, active_fractions, marker="^", linestyle=":", linewidth=2.0, color=color_active, label="Active Neurons (%)")
    ax2.set_ylabel("Active Neurons (%)", fontsize=11, color=color_active, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_active)
    ax2.set_ylim(-5, 105)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", frameon=True)

    plt.title("Connectome Dynamics: Synaptic Weight Scale Sensitivity Sweep", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()

    sweep_fig_path = figures_dir / "weight_scale_vs_activity.png"
    plt.savefig(sweep_fig_path, dpi=300)
    plt.close()
    logger.info(f"Saved weight scale sweep plot to: {sweep_fig_path}")

    return sweep_df


def main():
    parser = argparse.ArgumentParser(description="Baseline Connectome-Constrained LIF Simulation")
    parser.add_argument("--neurons", type=int, default=1000, help="Number of neurons in subnetwork (default: 1000)")
    parser.add_argument("--duration", type=float, default=1000.0, help="Simulation duration in ms (default: 1000.0)")
    parser.add_argument("--dt", type=float, default=0.5, help="Integration timestep in ms (default: 0.5)")
    parser.add_argument("--weight-scale", type=float, default=0.01, help="Synaptic weight scale (default: 0.01)")
    parser.add_argument("--stimulus", type=str, default="pulse", choices=["constant", "random", "pulse"], help="External stimulus mode")
    parser.add_argument("--external-current", type=float, default=18.0, help="External stimulus current amplitude (default: 18.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Path to processed Parquet directory")
    parser.add_argument("--tables-dir", type=str, default="results/tables", help="Directory for CSV tables")
    parser.add_argument("--figures-dir", type=str, default="results/figures", help="Directory for figures")
    parser.add_argument("--logs-dir", type=str, default="results/logs", help="Directory for run logs")
    parser.add_argument("--skip-sweep", action="store_true", help="Skip the parameter sweep")

    args = parser.parse_args()

    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)
    logs_dir = Path(args.logs_dir)

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print(" CONNECTOME-CONSTRAINED LIF SIMULATION (PHASE 3)")
    print("=" * 65)
    print(f" Target Subnetwork:          {args.neurons:,} neurons (highest_degree)")
    print(f" Simulation Duration:        {args.duration} ms (dt = {args.dt} ms)")
    print(f" Synaptic Weight Scale:      {args.weight_scale}")
    print(f" Stimulus Mode:              {args.stimulus} (Amplitude: {args.external_current})")
    print(f" Deterministic Random Seed:  {args.seed}")
    print("=" * 65 + "\n")

    # 1. Load Real Subnetwork
    logger.info(f"Extracting {args.neurons}-neuron real subnetwork from {args.processed_dir}...")
    sub_graph, diagnostics = extract_real_subnetwork(
        processed_dir=args.processed_dir,
        max_neurons=args.neurons,
        strategy="highest_degree",
        seed=args.seed,
        verbose=True,
    )

    network = LIFNetwork(sub_graph)

    # 2. Smoke Test: Short 50 ms run to verify integrity before full execution
    logger.info("Executing 50 ms smoke test...")
    smoke_cfg = LIFConfig(
        duration=50.0,
        dt=args.dt,
        synaptic_weight_scale=args.weight_scale,
        external_stimulus_mode=args.stimulus,
        external_current=args.external_current,
        random_seed=args.seed,
    )
    smoke_res = network.simulate(smoke_cfg, record_samples=2)
    assert smoke_res.is_stable, f"Smoke test failed stability check: {smoke_res.status_message}"
    logger.info("Smoke test passed successfully!")

    # 3. Full Baseline Simulation
    logger.info(f"Executing full {args.duration:.0f} ms baseline simulation...")
    full_cfg = LIFConfig(
        duration=args.duration,
        dt=args.dt,
        synaptic_weight_scale=args.weight_scale,
        external_stimulus_mode=args.stimulus,
        external_current=args.external_current,
        random_seed=args.seed,
    )

    tracemalloc.start()
    t_start = time.perf_counter()

    full_res = network.simulate(full_cfg, record_samples=5)

    t_end = time.perf_counter()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    runtime_sec = t_end - t_start
    peak_ram_mb = peak_mem / (1024 * 1024)

    # 4. Save Baseline Metrics CSV
    metrics = full_res.metrics
    baseline_records = {
        "dataset_name": ["Janelia FlyEM Drosophila Hemibrain"],
        "dataset_version": ["v1.2.1"],
        "subnetwork_neurons": [args.neurons],
        "subnetwork_edges": [sub_graph.number_of_edges()],
        "selection_strategy": ["highest_degree"],
        "random_seed": [args.seed],
        "duration_ms": [args.duration],
        "dt_ms": [args.dt],
        "num_time_steps": [int(args.duration / args.dt)],
        "synaptic_weight_scale": [args.weight_scale],
        "external_stimulus_mode": [args.stimulus],
        "external_current": [args.external_current],
        "total_spikes": [metrics["total_spikes"]],
        "spikes_per_neuron": [metrics["spikes_per_neuron"]],
        "mean_firing_rate_hz": [metrics["mean_firing_rate_hz"]],
        "median_firing_rate_hz": [metrics["median_firing_rate_hz"]],
        "max_firing_rate_hz": [metrics["max_firing_rate_hz"]],
        "firing_rate_variance": [metrics["firing_rate_variance"]],
        "active_neuron_fraction": [metrics["active_neuron_fraction"]],
        "is_stable": [full_res.is_stable],
        "status_message": [full_res.status_message],
        "runtime_seconds": [round(runtime_sec, 3)],
        "peak_ram_mb": [round(peak_ram_mb, 2)],
    }

    baseline_df = pd.DataFrame(baseline_records)
    csv_out = tables_dir / "baseline_metrics.csv"
    baseline_df.to_csv(csv_out, index=False)
    logger.info(f"Saved baseline metrics to: {csv_out}")

    # 5. Generate Figures
    generate_baseline_figures(full_res, full_cfg, figures_dir)

    # 6. Parameter Sweep (0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
    if not args.skip_sweep:
        scales = [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
        run_parameter_sweep(network, scales, full_cfg, tables_dir, figures_dir)

    print("\n" + "=" * 65)
    print(" BASELINE SIMULATION SUMMARY REPORT")
    print("=" * 65)
    print(f" Neurons:                 {args.neurons:,}")
    print(f" Edges:                   {sub_graph.number_of_edges():,}")
    print(f" Simulation Duration:     {args.duration:.1f} ms (Timestep dt: {args.dt} ms)")
    print(f" Number of Time Steps:    {int(args.duration / args.dt):,}")
    print(f" Total Spikes Emitted:    {metrics['total_spikes']:,}")
    print(f" Mean Firing Rate:        {metrics['mean_firing_rate_hz']:.2f} Hz")
    print(f" Median Firing Rate:      {metrics['median_firing_rate_hz']:.2f} Hz")
    print(f" Max Firing Rate:         {metrics['max_firing_rate_hz']:.2f} Hz")
    print(f" Active Neuron Fraction:  {metrics['active_neuron_fraction']*100:.1f}%")
    print(f" Numerical Stability:     {full_res.status_message}")
    print(f" Execution Runtime:       {runtime_sec:.3f} s")
    print(f" Peak Memory Overhead:    {peak_ram_mb:.2f} MB")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
