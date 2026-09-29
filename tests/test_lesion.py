"""
Unit tests for lesion module.
Verifies critical scientific invariants:
1. Lesioning never increases neuron count.
2. Lesioning never creates new synapses or edges.
3. Random seed produces reproducible selections.
4. Edge sets of lesioned graphs are strict subsets of intact graph.
"""

import pytest

from src.data_loader import generate_mock_connectome
from src.graph_builder import build_graph
from src.lesion import apply_hub_lesion, apply_lesion, apply_random_lesion


@pytest.fixture
def test_network():
    nodes_df, edges_df = generate_mock_connectome(num_neurons=200, avg_degree=10.0, random_seed=42)
    return build_graph(nodes_df, edges_df)


def test_random_lesion_never_increases_neurons(test_network):
    """Invariant 1: Node count strictly decreases or remains equal."""
    orig_n = test_network.number_of_nodes()

    for p in [0.0, 0.01, 0.05, 0.10, 0.20, 0.50]:
        lesioned_g, lesioned_nodes = apply_random_lesion(test_network, lesion_fraction=p, seed=42)
        assert lesioned_g.number_of_nodes() <= orig_n
        assert len(lesioned_nodes) == orig_n - lesioned_g.number_of_nodes()


def test_hub_lesion_never_increases_neurons(test_network):
    """Invariant 1 (hub): Targeted hub removal strictly decreases neuron count."""
    orig_n = test_network.number_of_nodes()

    for p in [0.01, 0.05, 0.10, 0.20]:
        lesioned_g, lesioned_nodes = apply_hub_lesion(test_network, lesion_fraction=p)
        assert lesioned_g.number_of_nodes() < orig_n
        assert len(lesioned_nodes) == orig_n - lesioned_g.number_of_nodes()


def test_lesion_never_creates_new_synapses(test_network):
    """Invariant 2: Remaining edges must be a strict subset of original edges."""
    orig_edges = set(test_network.edges())
    orig_weight = sum(d.get("weight", 1.0) for _, _, d in test_network.edges(data=True))

    for p in [0.05, 0.15]:
        lesioned_g, _ = apply_random_lesion(test_network, lesion_fraction=p, seed=123)
        lesioned_edges = set(lesioned_g.edges())

        # Subset check
        assert lesioned_edges.issubset(orig_edges)

        # Synaptic weight check
        lesioned_weight = sum(d.get("weight", 1.0) for _, _, d in lesioned_g.edges(data=True))
        assert lesioned_weight <= orig_weight


def test_random_lesion_reproducibility(test_network):
    """Invariant 3: Identical seed produces identical lesion; different seed differs."""
    g1, nodes1 = apply_random_lesion(test_network, lesion_fraction=0.10, seed=777)
    g2, nodes2 = apply_random_lesion(test_network, lesion_fraction=0.10, seed=777)
    g3, nodes3 = apply_random_lesion(test_network, lesion_fraction=0.10, seed=888)

    assert nodes1 == nodes2
    assert set(g1.nodes()) == set(g2.nodes())
    assert set(g1.edges()) == set(g2.edges())

    # Different seeds should yield different ablated nodes
    assert nodes1 != nodes3


def test_hub_lesion_removes_highest_degree(test_network):
    """Verifies that hub lesion removes top degree nodes."""
    from src.graph_builder import calculate_degree
    degrees = calculate_degree(test_network, weighted=False)
    sorted_nodes = sorted(degrees.items(), key=lambda x: (-x[1], x[0]))

    num_to_remove = int(len(test_network) * 0.05)
    expected_hubs = [n for n, _ in sorted_nodes[:num_to_remove]]
    lesioned_g, lesioned_nodes = apply_hub_lesion(test_network, lesion_fraction=0.05, weighted=False)

    assert set(expected_hubs) == set(lesioned_nodes)

