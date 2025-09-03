# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 16:03:24 2025

@author: wwj19
"""

from typing import Any, Dict, Hashable, List, Optional, Set, Tuple
import networkx as nx
import numpy as np

# node_dict: {node_id: (time_series: np.ndarray, value: float, other_info: dict)}
NodeDict = Dict[Hashable, Tuple[np.ndarray, float, Dict[str, Any]]]

# Nonlinear parameter for similarity mapping
A_NONLINEAR: float = 0.9


def _sim_fds(ts1: np.ndarray, ts2: np.ndarray, a: float = A_NONLINEAR) -> float:
    """
    Frequency-Domain Similarity (FDS) + monotonic mapping, returns a score in [-1, 1].

    Steps:
      1) Mean-center both time series
      2) Take FFT magnitudes (first half spectrum)
      3) L2 normalize and compute cosine similarity
      4) Apply nonlinear mapping: (s - a) / (1 - a*s)
    """
    f = np.asarray(ts1, dtype=np.float64) - float(np.mean(ts1))
    g = np.asarray(ts2, dtype=np.float64) - float(np.mean(ts2))

    F = np.abs(np.fft.fft(f))[: len(f) // 2]
    G = np.abs(np.fft.fft(g))[: len(g) // 2]

    nF, nG = np.linalg.norm(F), np.linalg.norm(G)
    if nF == 0.0 or nG == 0.0:
        return 0.0

    s = float(np.dot(F / nF, G / nG))  # in [-1, 1]
    return float((s - a) / (1.0 - a * s))


def expand_subgraph_greedy(
    G: nx.DiGraph,
    node_dict: NodeDict,
    root: Hashable,
) -> Tuple[Set[Hashable], float]:
    """
    Greedily expand a subgraph from `root` to maximize:

        g(S) = (Σ_{v∈S} sim(ts_root, ts_v)) * (Σ_{v∈S} value_v) / |S|

    Fixed policy:
      - Only expand within descendants(root) ∪ {root}
      - Only consider out-neighbors (successors)
      - Deterministic tie-break: nodes sorted by str(node_id)

    Returns:
        (selected_nodes, best_g_value)
    """
    if root not in node_dict or root not in G:
        return set(), 0.0

    try:
        scope: Set[Hashable] = set(nx.descendants(G, root)) | {root}
    except Exception:
        scope = {root}

    ts_root = node_dict[root][0]
    sim_cache: Dict[Hashable, float] = {root: 1.0}
    for u in scope:
        if u == root:
            continue
        sim_cache[u] = _sim_fds(ts_root, node_dict[u][0])

    # Initial subgraph
    S: Set[Hashable] = {root}
    S_sim_sum = 1.0
    S_val_sum = float(node_dict[root][1])
    S_size = 1
    g_best = (S_sim_sum * S_val_sum) / S_size

    # Iterative expansion
    while True:
        frontier: Set[Hashable] = set()
        for u in S:
            for v in G.successors(u):
                if v in scope and v not in S:
                    frontier.add(v)

        if not frontier:
            break

        best_choice: Optional[Hashable] = None
        best_new_g = g_best
        best_new_sim_sum, best_new_val_sum, best_new_size = S_sim_sum, S_val_sum, S_size

        for n in sorted(frontier, key=lambda x: str(x)):
            sim_n = sim_cache[n]
            val_n = float(node_dict[n][1])

            new_sim_sum = S_sim_sum + sim_n
            new_val_sum = S_val_sum + val_n
            new_size = S_size + 1
            new_g = (new_sim_sum * new_val_sum) / new_size

            if new_g > best_new_g or (
                abs(new_g - best_new_g) < 1e-15 and best_choice is not None and str(n) < str(best_choice)
            ):
                best_choice = n
                best_new_g = new_g
                best_new_sim_sum, best_new_val_sum, best_new_size = new_sim_sum, new_val_sum, new_size

        if best_choice is None or best_new_g <= g_best:
            break

        # Commit choice
        S.add(best_choice)
        S_sim_sum, S_val_sum, S_size = best_new_sim_sum, best_new_val_sum, best_new_size
        g_best = best_new_g

    return S, float(g_best)


def find_k_best_subgraphs(
    G: nx.DiGraph,
    node_dict: NodeDict,
    k: int,
) -> Tuple[List[Tuple[Set[Hashable], float]], float]:
    """
    Iteratively run expand_subgraph_greedy to extract k disjoint subgraphs.
    After selecting one, its nodes are marked as used, and all incident edges
    are removed from the working graph to prevent cross-subgraph expansion.

    Returns:
        (subgraphs, total_g)
        subgraphs = [({nodes}, g_value), ...] in selection order.
    """
    if k <= 0:
        return [], 0.0

    used: Set[Hashable] = set()
    results: List[Tuple[Set[Hashable], float]] = []
    total_g = 0.0
    G_work = G.copy()

    seeds = [u for u in node_dict.keys() if u in G_work]
    seeds.sort(key=lambda x: str(x))

    for _ in range(k):
        best_S: Optional[Set[Hashable]] = None
        best_g = float("-inf")

        for u in seeds:
            if u in used:
                continue
            S_u, g_u = expand_subgraph_greedy(G_work, node_dict, u)
            if g_u > best_g:
                best_g = g_u
                best_S = S_u

        if not best_S:
            break

        results.append((best_S, float(best_g)))
        total_g += float(best_g)
        used.update(best_S)

        for n in best_S:
            if G_work.has_node(n):
                G_work.remove_edges_from(list(G_work.in_edges(n)) + list(G_work.out_edges(n)))

        seeds = [u for u in seeds if u not in used]
        if not seeds:
            break

    return results, float(total_g)
