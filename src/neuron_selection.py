"""
Subnetwork selection methods for Drosophila connectome graphs.
Provides strategies to extract biologically grounded, computationally tractable
subnetworks (1,000 - 5,000 neurons) respecting laptop hardware constraints.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from src.graph_builder import calculate_degree, calculate_in_degree, calculate_out_degree, get_graph_summary, get_subgraph

logger = logging.getLogger(__name__)


def random_selection(
    graph: nx.DiGraph,
    n_neurons: int = 1000,
    seed: Optional[int] = 42,
) -> List[int]:
    """
    Randomly selects n_neurons from the graph using a reproducible random seed.
    """
    all_nodes = list(graph.nodes())
    if len(all_nodes) <= n_neurons:
        return all_nodes

    rng = np.random.default_rng(seed)
    selected = rng.choice(all_nodes, size=n_neurons, replace=False)
    return [int(x) for x in selected]


def highest_degree_selection(
    graph: nx.DiGraph,
    n_neurons: int = 1000,
    weighted: bool = True,
    mode: str = "total",
) -> List[int]:
    """
    Selects the top n_neurons with the highest degree/strength.

    Parameters:
        graph: DiGraph connectome.
        n_neurons: Maximum number of neurons to select.
        weighted: If True, uses synapse count (strength); otherwise unweighted edge count.
        mode: 'total' (in+out), 'in' (integrator hubs), or 'out' (broadcaster hubs).
    """
    if mode == "in":
        deg = calculate_in_degree(graph, weighted=weighted)
    elif mode == "out":
        deg = calculate_out_degree(graph, weighted=weighted)
    else:
        deg = calculate_degree(graph, weighted=weighted)

    sorted_neurons = sorted(deg.items(), key=lambda item: item[1], reverse=True)
    selected = [node for node, _ in sorted_neurons[:n_neurons]]
    return selected


def cell_type_selection(
    graph: nx.DiGraph,
    target_cell_types: Iterable[str],
    max_neurons: Optional[int] = None,
) -> List[int]:
    """
    Selects neurons matching specified cell types or superclasses.

    Parameters:
        graph: DiGraph connectome.
        target_cell_types: Collection of cell type or superclass names (case-insensitive).
        max_neurons: Optional limit on the number of selected neurons.
    """
    targets = {ct.lower() for ct in target_cell_types}
    selected: List[int] = []

    for node, data in graph.nodes(data=True):
        ctype = str(data.get("cell_type", "")).lower()
        sclass = str(data.get("super_class", "")).lower()
        roi = str(data.get("roi", "")).lower()

        if ctype in targets or sclass in targets or roi in targets:
            selected.append(node)
            if max_neurons is not None and len(selected) >= max_neurons:
                break

    return selected


def neighborhood_selection(
    graph: nx.DiGraph,
    seed_neuron: int,
    max_hops: int = 2,
    max_neurons: Optional[int] = 5000,
) -> List[int]:
    """
    Extracts a local neighborhood subnetwork centered around a seed neuron
    using breadth-first traversal up to max_hops.

    Parameters:
        graph: DiGraph connectome.
        seed_neuron: Target focal neuron ID.
        max_hops: Maximum path distance from seed neuron.
        max_neurons: Maximum neurons to include.
    """
    if seed_neuron not in graph:
        raise ValueError(f"Seed neuron {seed_neuron} not found in graph.")

    visited: Set[int] = {seed_neuron}
    current_frontier: Set[int] = {seed_neuron}

    for _ in range(max_hops):
        next_frontier: Set[int] = set()
        for node in current_frontier:
            successors = set(graph.successors(node))
            predecessors = set(graph.predecessors(node))
            neighbors = successors.union(predecessors)
            for nbr in neighbors:
                if nbr not in visited:
                    visited.add(nbr)
                    next_frontier.add(nbr)
                    if max_neurons is not None and len(visited) >= max_neurons:
                        return list(visited)
        current_frontier = next_frontier
        if not current_frontier:
            break

    return list(visited)


def select_subnetwork(
    graph: nx.DiGraph,
    strategy: str = "highest_degree",
    max_neurons: int = 5000,
    seed: Optional[int] = 42,
    target_cell_types: Optional[List[str]] = None,
    seed_neuron: Optional[int] = None,
    max_hops: int = 2,
    verbose: bool = True,
) -> Tuple[nx.DiGraph, Dict[str, Any]]:
    """
    Master subnetwork extraction function. Applies chosen strategy, creates induced subgraph,
    computes and prints standardized network diagnostics.

    Returns:
        (sub_graph, diagnostics_dict)
    """
    total_available = graph.number_of_nodes()
    strategy = strategy.lower()

    if total_available <= max_neurons and strategy != "cell_type" and strategy != "neighborhood":
        selected_nodes = list(graph.nodes())
    elif strategy == "highest_degree":
        selected_nodes = highest_degree_selection(graph, n_neurons=max_neurons, weighted=True)
    elif strategy == "random":
        selected_nodes = random_selection(graph, n_neurons=max_neurons, seed=seed)
    elif strategy == "cell_type":
        if not target_cell_types:
            raise ValueError("target_cell_types must be provided when strategy='cell_type'")
        selected_nodes = cell_type_selection(graph, target_cell_types=target_cell_types, max_neurons=max_neurons)
    elif strategy == "neighborhood":
        if seed_neuron is None:
            # Auto-pick top hub neuron as seed if not specified
            top_hubs = highest_degree_selection(graph, n_neurons=1)
            seed_neuron = top_hubs[0] if top_hubs else list(graph.nodes())[0]
        selected_nodes = neighborhood_selection(graph, seed_neuron=seed_neuron, max_hops=max_hops, max_neurons=max_neurons)
    else:
        raise ValueError(
            f"Unknown selection strategy '{strategy}'. Options: ['highest_degree', 'random', 'cell_type', 'neighborhood']"
        )

    sub_g = get_subgraph(graph, selected_nodes)
    summary = get_graph_summary(sub_g)

    diagnostics = {
        "strategy": strategy,
        "total_neurons_available": total_available,
        "selected_neurons": sub_g.number_of_nodes(),
        "total_synapses": summary["total_synapses"],
        "total_edges": summary["total_edges"],
        "graph_density": summary["density"],
        "average_degree": summary["average_degree"],
        "max_degree": summary["max_degree"],
        "connected_components": summary["num_weakly_connected"],
        "largest_component_size": summary["largest_wcc_size"],
    }

    if verbose:
        print("\n" + "=" * 65)
        print(f" SUBNETWORK SELECTION DIAGNOSTICS (Strategy: {strategy})")
        print("=" * 65)
        print(f" Total neurons available:     {diagnostics['total_neurons_available']:,}")
        print(f" Selected neurons:            {diagnostics['selected_neurons']:,}")
        print(f" Total synapses:              {diagnostics['total_synapses']:,.0f}")
        print(f" Total directed edges:        {diagnostics['total_edges']:,}")
        print(f" Graph density:               {diagnostics['graph_density']:.6f}")
        print(f" Average degree:              {diagnostics['average_degree']:.2f}")
        print(f" Maximum degree:              {diagnostics['max_degree']:.2f}")
        print(f" Connected components:        {diagnostics['connected_components']}")
        print(f" Largest connected component: {diagnostics['largest_component_size']:,}")
        print("=" * 65 + "\n")

    return sub_g, diagnostics
