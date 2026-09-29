"""
Graph representation and topological analytics for Drosophila connectome.
Represents the connectome as a directed weighted graph G = (V, E).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

from src.data_loader import SCHEMA, load_processed_data

logger = logging.getLogger(__name__)


def build_graph(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
) -> nx.DiGraph:
    """
    Constructs a directed, weighted NetworkX graph G = (V, E) from nodes and edges DataFrames.

    Node attributes stored:
        - cell_type: specific morphologic/functional cell type (e.g. KC_gamma)
        - super_class: broad anatomical class (e.g. Kenyon_Cell, Projection_Neuron)
        - roi: neuropil region (e.g. Mushroom_Body, Central_Complex)
        - body_id: raw ID from source dataset
        - neurotransmitter: predicted neurotransmitter if available

    Edge attributes stored:
        - weight: synapse count (connection strength)
        - roi: neuropil where connection occurs
    """
    G = nx.DiGraph()

    # Add nodes with all associated metadata
    for row in nodes_df.itertuples(index=False):
        node_id = int(getattr(row, SCHEMA.NODE_ID))
        attr = {
            "body_id": getattr(row, "body_id", node_id),
            SCHEMA.CELL_TYPE: str(getattr(row, SCHEMA.CELL_TYPE, "unknown")),
            SCHEMA.SUPER_CLASS: str(getattr(row, SCHEMA.SUPER_CLASS, "unknown")),
            SCHEMA.NEUROPIL: str(getattr(row, SCHEMA.NEUROPIL, "unknown")),
            "neurotransmitter": str(getattr(row, "neurotransmitter", "unknown")),
        }
        G.add_node(node_id, **attr)

    # Add directed weighted edges (aggregating parallel multi-ROI edges)
    if not edges_df.empty and SCHEMA.SOURCE in edges_df.columns and SCHEMA.TARGET in edges_df.columns:
        agg_dict = {SCHEMA.WEIGHT: "sum"}
        if SCHEMA.NEUROPIL in edges_df.columns:
            agg_dict[SCHEMA.NEUROPIL] = "first"
        
        grouped = edges_df.groupby([SCHEMA.SOURCE, SCHEMA.TARGET], as_index=False, observed=True).agg(agg_dict)
        
        edge_tuples = [
            (
                int(getattr(row, SCHEMA.SOURCE)),
                int(getattr(row, SCHEMA.TARGET)),
                {
                    "weight": float(getattr(row, SCHEMA.WEIGHT)),
                    "roi": str(getattr(row, SCHEMA.NEUROPIL, "unknown")),
                },
            )
            for row in grouped.itertuples(index=False)
        ]
        G.add_edges_from(edge_tuples)

    logger.info(f"Constructed DiGraph G with {G.number_of_nodes()} neurons and {G.number_of_edges()} directed edges.")
    return G


def load_connectome(
    data_dir: str | Path = "data/processed",
) -> nx.DiGraph:
    """
    Loads processed connectome data files and constructs the directed graph.

    Parameters:
        data_dir: Directory containing 'neurons.parquet' and 'connections.parquet'.

    Returns:
        nx.DiGraph: Directed weighted connectome graph.
    """
    nodes_df, edges_df = load_processed_data(data_dir)
    return build_graph(nodes_df, edges_df)


def get_subgraph(
    graph: nx.DiGraph,
    neuron_ids: Iterable[int],
) -> nx.DiGraph:
    """
    Returns the induced subgraph for a given subset of neuron IDs,
    preserving all node and edge attributes.
    """
    valid_ids = set(neuron_ids).intersection(set(graph.nodes()))
    sub_g = graph.subgraph(valid_ids).copy()
    return sub_g


def calculate_degree(
    graph: nx.DiGraph,
    weighted: bool = True,
) -> Dict[int, float]:
    """
    Calculates total degree (in + out) for all neurons.
    If weighted=True, sums synaptic counts (strength); otherwise sums edge count.
    """
    if weighted:
        in_s = calculate_in_degree(graph, weighted=True)
        out_s = calculate_out_degree(graph, weighted=True)
        return {n: in_s.get(n, 0.0) + out_s.get(n, 0.0) for n in graph.nodes()}
    return dict(graph.degree())


def calculate_in_degree(
    graph: nx.DiGraph,
    weighted: bool = True,
) -> Dict[int, float]:
    """
    Calculates in-degree (post-synaptic connections receiving input).
    """
    if weighted:
        return dict(graph.in_degree(weight="weight"))
    return dict(graph.in_degree())


def calculate_out_degree(
    graph: nx.DiGraph,
    weighted: bool = True,
) -> Dict[int, float]:
    """
    Calculates out-degree (pre-synaptic connections projecting output).
    """
    if weighted:
        return dict(graph.out_degree(weight="weight"))
    return dict(graph.out_degree())


def calculate_centrality(
    graph: nx.DiGraph,
    metric: str = "degree",
    max_iter: int = 100,
) -> Dict[int, float]:
    """
    Calculates network centrality for all neurons.

    Supported metrics:
        - 'degree': Normalized total degree
        - 'in_degree': In-degree centrality (integrator neurons)
        - 'out_degree': Out-degree centrality (broadcaster neurons)
        - 'pagerank': PageRank centrality
        - 'betweenness': Betweenness centrality (uses k-sampling on graphs > 1000 nodes for speed)
    """
    metric = metric.lower()

    if metric == "degree":
        return nx.degree_centrality(graph)
    elif metric == "in_degree":
        return nx.in_degree_centrality(graph)
    elif metric == "out_degree":
        return nx.out_degree_centrality(graph)
    elif metric == "pagerank":
        return nx.pagerank(graph, weight="weight", max_iter=max_iter)
    elif metric == "betweenness":
        # On larger graphs (1000+ nodes), approximate betweenness using k-sample to avoid freezing CPU
        n_nodes = graph.number_of_nodes()
        k = min(200, n_nodes) if n_nodes > 500 else None
        return nx.betweenness_centrality(graph, k=k, weight="weight")
    else:
        raise ValueError(
            f"Unsupported centrality metric '{metric}'. Choose from ['degree', 'in_degree', 'out_degree', 'pagerank', 'betweenness']."
        )


def get_graph_summary(graph: nx.DiGraph) -> Dict[str, Any]:
    """
    Computes key topological summary statistics required for connectome inspection:
        - Total neurons available
        - Total synapses (sum of edge weights)
        - Total directed edges
        - Graph density
        - Average degree
        - Maximum degree
        - Strongly & Weakly connected components
    """
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    if n_nodes == 0:
        return {
            "total_neurons": 0,
            "total_synapses": 0,
            "total_edges": 0,
            "density": 0.0,
            "average_degree": 0.0,
            "max_degree": 0.0,
            "num_strongly_connected": 0,
            "num_weakly_connected": 0,
            "largest_wcc_size": 0,
            "largest_wcc_fraction": 0.0,
        }

    degrees = dict(graph.degree())
    weights = [data.get("weight", 1.0) for _, _, data in graph.edges(data=True)]
    total_synapses = sum(weights)

    density = nx.density(graph)
    avg_deg = sum(degrees.values()) / n_nodes
    max_deg = max(degrees.values())

    wcc = list(nx.weakly_connected_components(graph))
    scc = list(nx.strongly_connected_components(graph))
    largest_wcc = max(len(c) for c in wcc) if wcc else 0

    return {
        "total_neurons": n_nodes,
        "total_synapses": total_synapses,
        "total_edges": n_edges,
        "density": density,
        "average_degree": avg_deg,
        "max_degree": max_deg,
        "num_strongly_connected": len(scc),
        "num_weakly_connected": len(wcc),
        "largest_wcc_size": largest_wcc,
        "largest_wcc_fraction": largest_wcc / n_nodes,
    }


def print_graph_summary(graph: nx.DiGraph, title: str = "Connectome Subnetwork Summary") -> None:
    """Prints a structured summary table to console."""
    summary = get_graph_summary(graph)
    print("\n" + "=" * 60)
    print(f" {title.upper()}")
    print("=" * 60)
    print(f" Total neurons available:        {summary['total_neurons']:,}")
    print(f" Total directed edges:          {summary['total_edges']:,}")
    print(f" Total synaptic weight:         {summary['total_synapses']:,.0f}")
    print(f" Graph density:                 {summary['density']:.6f}")
    print(f" Average degree:                {summary['average_degree']:.2f}")
    print(f" Maximum degree:                {summary['max_degree']:.2f}")
    print(f" Weakly connected components:   {summary['num_weakly_connected']}")
    print(f" Strongly connected components: {summary['num_strongly_connected']}")
    print(f" Largest connected component:   {summary['largest_wcc_size']:,} ({summary['largest_wcc_fraction']*100:.1f}%)")
    print("=" * 60 + "\n")
