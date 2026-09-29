"""
Experiment 01: Real Connectome Exploration & Subnetwork Extraction
Investigates the Janelia FlyEM Drosophila Hemibrain v1.2.1 connectome.
Extracts a memory-safe subnetwork (default: 1,000 neurons), measures topology and resource usage,
and produces validation tables and publication-quality degree distribution figures.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import tracemalloc
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from src.data_loader import extract_real_subnetwork, validate_real_data
from src.graph_builder import calculate_degree, calculate_in_degree, calculate_out_degree, get_graph_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_connectome_exploration(
    neurons: int = 1000,
    strategy: str = "highest_degree",
    seed: int = 42,
    processed_dir: str = "data/processed",
    output_tables_dir: str = "results/tables",
    output_figures_dir: str = "results/figures",
    output_logs_dir: str = "results/logs",
) -> pd.DataFrame:
    """
    Executes real Drosophila connectome subnetwork extraction, measures peak memory
    and execution latency, generates summary metrics CSV, and renders degree plots.
    """
    tables_path = Path(output_tables_dir)
    figures_path = Path(output_figures_dir)
    logs_path = Path(output_logs_dir)

    tables_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)
    logs_path.mkdir(parents=True, exist_ok=True)

    # Start memory and latency tracing
    tracemalloc.start()
    t_start = time.perf_counter()

    logger.info(f"Extracting {neurons}-neuron real Drosophila subnetwork (strategy: {strategy}, seed: {seed})...")
    sub_graph, diagnostics = extract_real_subnetwork(
        processed_dir=processed_dir,
        max_neurons=neurons,
        strategy=strategy,
        seed=seed,
        verbose=True,
    )

    t_end = time.perf_counter()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    runtime_sec = t_end - t_start
    peak_mem_mb = peak_mem / (1024 * 1024)

    logger.info(f"Subnetwork extraction complete in {runtime_sec:.3f} s (Peak RAM: {peak_mem_mb:.2f} MB)")

    # 1. Topological Summary Metrics
    summary = get_graph_summary(sub_graph)
    in_degrees = calculate_in_degree(sub_graph, weighted=False)
    out_degrees = calculate_out_degree(sub_graph, weighted=False)
    tot_degrees = calculate_degree(sub_graph, weighted=False)

    weighted_in = calculate_in_degree(sub_graph, weighted=True)
    weighted_out = calculate_out_degree(sub_graph, weighted=True)

    results_data = {
        "dataset_name": ["Janelia FlyEM Drosophila Hemibrain"],
        "dataset_version": ["v1.2.1"],
        "selection_strategy": [strategy],
        "random_seed": [seed],
        "total_neurons_in_source": [diagnostics["total_neurons_in_source"]],
        "total_connections_in_source": [4259624],
        "selected_neurons": [sub_graph.number_of_nodes()],
        "selected_edges": [sub_graph.number_of_edges()],
        "total_synaptic_weight": [summary["total_synapses"]],
        "graph_density": [summary["density"]],
        "average_total_degree": [summary["average_degree"]],
        "max_total_degree": [summary["max_degree"]],
        "average_in_degree": [np.mean(list(in_degrees.values()))],
        "average_out_degree": [np.mean(list(out_degrees.values()))],
        "weakly_connected_components": [summary["num_weakly_connected"]],
        "strongly_connected_components": [summary["num_strongly_connected"]],
        "largest_wcc_size": [summary["largest_wcc_size"]],
        "largest_wcc_fraction": [summary["largest_wcc_fraction"]],
        "runtime_seconds": [round(runtime_sec, 3)],
        "peak_ram_mb": [round(peak_mem_mb, 2)],
    }

    results_df = pd.DataFrame(results_data)
    csv_out = tables_path / "real_connectome_summary.csv"
    results_df.to_csv(csv_out, index=False)
    logger.info(f"Saved real connectome summary to: {csv_out}")

    # 2. Publication-Quality Degree Distribution Plot
    logger.info("Generating publication-quality degree distribution figure...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    tot_vals = np.array(list(tot_degrees.values()))
    in_vals = np.array(list(in_degrees.values()))
    out_vals = np.array(list(out_degrees.values()))

    # Subplot 1: Total Degree Histogram
    ax0 = axes[0]
    ax0.hist(tot_vals, bins=40, color="#1f77b4", edgecolor="#0b3c61", alpha=0.85, log=True)
    ax0.set_title(f"Total Degree Distribution (N={len(tot_vals)})", fontsize=12, fontweight="bold", pad=10)
    ax0.set_xlabel("Degree $k$", fontsize=11)
    ax0.set_ylabel("Count (log scale)", fontsize=11)
    ax0.grid(True, linestyle="--", alpha=0.5)
    ax0.axvline(tot_vals.mean(), color="#d62728", linestyle="--", linewidth=1.8, label=f"Mean: {tot_vals.mean():.1f}")
    ax0.axvline(np.median(tot_vals), color="#2ca02c", linestyle=":", linewidth=1.8, label=f"Median: {np.median(tot_vals):.1f}")
    ax0.legend(frameon=True, facecolor="#F8F9FA", edgecolor="#CCCCCC")

    # Subplot 2: In-Degree vs Out-Degree Scatter / Density
    ax1 = axes[1]
    ax1.scatter(in_vals, out_vals, color="#6f42c1", alpha=0.55, edgecolors="none", s=25)
    ax1.set_title("In-Degree (Integrator) vs Out-Degree (Broadcaster)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("In-Degree $k_{in}$", fontsize=11)
    ax1.set_ylabel("Out-Degree $k_{out}$", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Reference diagonal line
    max_diag = max(in_vals.max(), out_vals.max())
    ax1.plot([0, max_diag], [0, max_diag], color="#888888", linestyle="--", alpha=0.7, label="Equal reciprocity ($k_{in}=k_{out}$)")
    ax1.legend(frameon=True, facecolor="#F8F9FA", edgecolor="#CCCCCC")

    # Subplot 3: Degree Rank Plot (Log-Log Scale - Heavy-tail inspection)
    ax2 = axes[2]
    sorted_deg = np.sort(tot_vals)[::-1]
    ranks = np.arange(1, len(sorted_deg) + 1)
    ax2.loglog(ranks, sorted_deg, marker="o", markersize=3, color="#e83e8c", linestyle="none", alpha=0.75, label="Subnetwork Neurons")
    ax2.set_title("Log-Log Degree Rank Plot (Hub Identification)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Rank (log scale)", fontsize=11)
    ax2.set_ylabel("Total Degree $k$ (log scale)", fontsize=11)
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend(frameon=True, facecolor="#F8F9FA", edgecolor="#CCCCCC")

    fig.suptitle(
        f"Drosophila Melanogaster Connectome Subnetwork (Janelia FlyEM Hemibrain v1.2.1 | N={neurons})",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    fig_out = figures_path / "real_connectome_degree_distribution.png"
    plt.savefig(fig_out, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved publication-quality figure to: {fig_out}")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real Connectome Exploration & Subnetwork Extraction")
    parser.add_argument("--neurons", type=int, default=1000, help="Number of neurons to extract (default: 1000)")
    parser.add_argument("--strategy", type=str, default="highest_degree", choices=["highest_degree", "random", "neighborhood"], help="Subnetwork selection strategy")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Path to processed Parquet directory")
    parser.add_argument("--tables-dir", type=str, default="results/tables", help="Directory for output CSV tables")
    parser.add_argument("--figures-dir", type=str, default="results/figures", help="Directory for output figures")
    parser.add_argument("--logs-dir", type=str, default="results/logs", help="Directory for experiment run logs")

    args = parser.parse_args()

    df = run_connectome_exploration(
        neurons=args.neurons,
        strategy=args.strategy,
        seed=args.seed,
        processed_dir=args.processed_dir,
        output_tables_dir=args.tables_dir,
        output_figures_dir=args.figures_dir,
        output_logs_dir=args.logs_dir,
    )

    print("\n" + "=" * 65)
    print(" EXPERIMENT 01 EXECUTION SUMMARY")
    print("=" * 65)
    for col in df.columns:
        print(f" {col:<32}: {df[col].iloc[0]}")
    print("=" * 65 + "\n")
