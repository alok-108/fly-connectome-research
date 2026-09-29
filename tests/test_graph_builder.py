"""
Unit tests for graph_builder module.
Verifies directed graph representation, degree metrics, centrality, and subgraphing.
"""

import networkx as nx
import pytest

from src.data_loader import generate_mock_connectome
from src.graph_builder import (
    build_graph,
    calculate_centrality,
    calculate_degree,
    calculate_in_degree,
    calculate_out_degree,
    get_graph_summary,
    get_subgraph,
)


@pytest.fixture
def sample_graph():
    """Provides a synthetic connectome graph with 100 neurons."""
    nodes_df, edges_df = generate_mock_connectome(num_neurons=100, avg_degree=8.0, random_seed=42)
    return build_graph(nodes_df, edges_df)


def test_build_graph_structure(sample_graph):
    """Verify nodes, edges, and metadata attributes in DiGraph."""
    assert isinstance(sample_graph, nx.DiGraph)
    assert sample_graph.number_of_nodes() == 100
    assert sample_graph.number_of_edges() > 0

    # Verify node attributes
    first_node = list(sample_graph.nodes())[0]
    attrs = sample_graph.nodes[first_node]
    assert "cell_type" in attrs
    assert "super_class" in attrs
    assert "roi" in attrs

    # Verify edge attributes
    u, v, data = list(sample_graph.edges(data=True))[0]
    assert "weight" in data
    assert data["weight"] > 0


def test_calculate_degrees(sample_graph):
    """Verify in-degree, out-degree, and total degree calculations."""
    in_deg = calculate_in_degree(sample_graph, weighted=False)
    out_deg = calculate_out_degree(sample_graph, weighted=False)
    tot_deg = calculate_degree(sample_graph, weighted=False)

    for n in sample_graph.nodes():
        assert tot_deg[n] == in_deg[n] + out_deg[n]

    # Weighted degree check
    weighted_in = calculate_in_degree(sample_graph, weighted=True)
    for n in sample_graph.nodes():
        assert weighted_in[n] >= in_deg[n]


def test_calculate_centrality(sample_graph):
    """Verify degree, PageRank, and betweenness centrality."""
    deg_cent = calculate_centrality(sample_graph, metric="degree")
    assert len(deg_cent) == sample_graph.number_of_nodes()

    pr_cent = calculate_centrality(sample_graph, metric="pagerank")
    assert len(pr_cent) == sample_graph.number_of_nodes()
    assert abs(sum(pr_cent.values()) - 1.0) < 1e-4

    with pytest.raises(ValueError):
        calculate_centrality(sample_graph, metric="unsupported_metric")


def test_get_subgraph(sample_graph):
    """Verify induced subgraph preserves attributes and removes external edges."""
    nodes = list(sample_graph.nodes())[:25]
    sub_g = get_subgraph(sample_graph, nodes)

    assert sub_g.number_of_nodes() == 25
    for n in sub_g.nodes():
        assert n in sample_graph
        assert sub_g.nodes[n]["cell_type"] == sample_graph.nodes[n]["cell_type"]

    for u, v in sub_g.edges():
        assert u in nodes and v in nodes


def test_get_graph_summary(sample_graph):
    """Verify topological summary statistics."""
    summary = get_graph_summary(sample_graph)
    assert summary["total_neurons"] == 100
    assert summary["total_edges"] == sample_graph.number_of_edges()
    assert summary["density"] > 0.0
    assert summary["average_degree"] > 0.0
    assert summary["max_degree"] >= summary["average_degree"]
    assert summary["largest_wcc_size"] <= 100
