"""
Virtual lesion (ablation) module for Drosophila connectome graphs.
Implements reproducible random and targeted (hub) lesions with strict biological invariants.
All functions operate on graph copies and never mutate the original network.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from src.graph_builder import calculate_degree, get_graph_summary

logger = logging.getLogger(__name__)


def apply_random_lesion(
    graph: nx.DiGraph,
    lesion_fraction: float,
    seed: Optional[int] = 42,
) -> Tuple[nx.DiGraph, List[int]]:
    """
    Applies a random virtual lesion removing exactly floor(lesion_fraction * N) neurons.
    Removes the selected neurons and all incident incoming and outgoing synapses.

    Strict Invariants Maintained:
        1. Original graph is never mutated (returns an isolated copy).
        2. Resulting neuron count == original_count - num_lesioned.
        3. Resulting edge count <= original edge count (never creates new synapses).
        4. All remaining edges belong to the original edge set.
        5. Deterministic selection given the same seed.

    Parameters:
        graph: Original intact connectome DiGraph.
        lesion_fraction: Fraction between 0.0 and 1.0 (e.g. 0.05 for 5%).
        seed: Random seed for exact reproducibility.

    Returns:
        (lesioned_graph_copy, list_of_lesioned_neuron_ids)
    """
    if not (0.0 <= lesion_fraction <= 1.0):
        raise ValueError(f"lesion_fraction must be in range [0.0, 1.0], got {lesion_fraction}")

    total_neurons = graph.number_of_nodes()
    if total_neurons == 0 or lesion_fraction == 0.0:
        return graph.copy(), []

    # Exactly floor(fraction * N)
    num_to_remove = int(np.floor(total_neurons * lesion_fraction))
    num_to_remove = min(num_to_remove, total_neurons)

    if num_to_remove == 0:
        return graph.copy(), []

    rng = np.random.default_rng(seed)
    all_nodes = np.array(sorted(list(graph.nodes())))
    lesioned_neurons = rng.choice(all_nodes, size=num_to_remove, replace=False).tolist()
    lesioned_set = set(lesioned_neurons)

    lesioned_graph = graph.copy()
    lesioned_graph.remove_nodes_from(lesioned_set)

    # Invariant assertion
    assert lesioned_graph.number_of_nodes() == total_neurons - num_to_remove, (
        f"Lesion invariant violated: expected {total_neurons - num_to_remove} neurons, "
        f"got {lesioned_graph.number_of_nodes()}"
    )

    return lesioned_graph, lesioned_neurons


def apply_hub_lesion(
    graph: nx.DiGraph,
    lesion_fraction: float,
    weighted: bool = False,
    mode: str = "total",
) -> Tuple[nx.DiGraph, List[int]]:
    """
    Applies a targeted virtual lesion by ablating the highest-degree (hub) neurons.
    Calculates total degree on the ORIGINAL pre-lesion graph and removes exactly
    floor(lesion_fraction * N) highest-degree neurons.

    Tie-breaking rule:
        Neurons are ranked primarily by degree (descending).
        Ties in degree are broken deterministically by ascending numerical node ID
        (i.e. (-degree, node_id)), ensuring 100% reproducible selections.

    Parameters:
        graph: Original intact connectome DiGraph.
        lesion_fraction: Fraction of neurons to remove.
        weighted: If True, ranks hubs by synaptic weight (strength); otherwise unweighted degree.
        mode: 'total' (in+out), 'in' (integrator hubs), or 'out' (broadcaster hubs).

    Returns:
        (lesioned_graph_copy, list_of_lesioned_neuron_ids)
    """
    if not (0.0 <= lesion_fraction <= 1.0):
        raise ValueError(f"lesion_fraction must be in range [0.0, 1.0], got {lesion_fraction}")

    total_neurons = graph.number_of_nodes()
    if total_neurons == 0 or lesion_fraction == 0.0:
        return graph.copy(), []

    # Exactly floor(fraction * N)
    num_to_remove = int(np.floor(total_neurons * lesion_fraction))
    num_to_remove = min(num_to_remove, total_neurons)

    if num_to_remove == 0:
        return graph.copy(), []

    degrees = calculate_degree(graph, weighted=weighted)

    # Sort descending by degree; break ties deterministically by ascending node ID
    sorted_neurons = sorted(degrees.items(), key=lambda x: (-x[1], x[0]))
    lesioned_neurons = [node for node, _ in sorted_neurons[:num_to_remove]]
    lesioned_set = set(lesioned_neurons)

    lesioned_graph = graph.copy()
    lesioned_graph.remove_nodes_from(lesioned_set)

    # Invariant assertion
    assert lesioned_graph.number_of_nodes() == total_neurons - num_to_remove, (
        f"Hub lesion invariant violated: expected {total_neurons - num_to_remove} neurons, "
        f"got {lesioned_graph.number_of_nodes()}"
    )

    return lesioned_graph, lesioned_neurons


def apply_lesion(
    graph: nx.DiGraph,
    mode: str = "random",
    lesion_fraction: float = 0.05,
    seed: Optional[int] = 42,
    weighted: bool = False,
) -> Tuple[nx.DiGraph, List[int]]:
    """
    Unified interface for applying virtual lesions.

    Parameters:
        mode: 'random' or 'hub' (or 'targeted').
        lesion_fraction: Proportion of neurons to ablate.
        seed: Random seed for random lesions.
        weighted: Weight flag for hub ranking.
    """
    mode = mode.lower()
    if mode == "random":
        return apply_random_lesion(graph, lesion_fraction=lesion_fraction, seed=seed)
    elif mode in ("hub", "targeted"):
        return apply_hub_lesion(graph, lesion_fraction=lesion_fraction, weighted=weighted)
    else:
        raise ValueError(f"Unknown lesion mode '{mode}'. Choose 'random' or 'hub'.")


def compute_lesion_graph_metrics(
    baseline_graph: nx.DiGraph,
    lesioned_graph: nx.DiGraph,
    lesion_strategy: str,
    lesion_fraction: float,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Computes comparative topological metrics between baseline and lesioned graphs.
    Handles division-by-zero safely.
    """
    base_n = baseline_graph.number_of_nodes()
    base_e = baseline_graph.number_of_edges()
    base_weights = sum(d.get("weight", 1.0) for _, _, d in baseline_graph.edges(data=True))
    base_wcc = max(len(c) for c in nx.weakly_connected_components(baseline_graph)) if base_n > 0 else 0

    les_n = lesioned_graph.number_of_nodes()
    les_e = lesioned_graph.number_of_edges()
    les_weights = sum(d.get("weight", 1.0) for _, _, d in lesioned_graph.edges(data=True))
    wcc_components = list(nx.weakly_connected_components(lesioned_graph))
    les_wcc = max(len(c) for c in wcc_components) if les_n > 0 else 0

    lesion_count = base_n - les_n

    # Safe ratios
    network_robustness = (les_e / base_e) if base_e > 0 else 0.0
    weight_survival_fraction = (les_weights / base_weights) if base_weights > 0 else 0.0
    wcc_relative_fraction = (les_wcc / base_wcc) if base_wcc > 0 else 0.0

    return {
        "lesion_strategy": lesion_strategy,
        "lesion_fraction": lesion_fraction,
        "lesion_count": lesion_count,
        "seed": seed if seed is not None else "NA",
        "baseline_neurons": base_n,
        "remaining_neurons": les_n,
        "baseline_edges": base_e,
        "remaining_edges": les_e,
        "remaining_total_synaptic_weight": round(les_weights, 1),
        "largest_weak_component_size": les_wcc,
        "largest_wcc_relative_fraction": round(wcc_relative_fraction, 4),
        "network_robustness": round(network_robustness, 4),
        "remaining_edge_fraction": round(network_robustness, 4),
        "remaining_weight_fraction": round(weight_survival_fraction, 4),
        "num_weak_components": len(wcc_components),
    }
