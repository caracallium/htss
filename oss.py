# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 17:50:55 2025

@author: wwj19
"""

import networkx as nx
import numpy as np
import heapq
from typing import Dict, Hashable, List, Optional, Set, Tuple


# node_dict: {node_id: (time_series: np.ndarray, value: float, other_info: dict)}
NodeDict = Dict[Hashable, Tuple[np.ndarray, float, Dict[str, Any]]]

# Nonlinear parameter for similarity mapping
NONLINEAR: float = 0.9


def _sim_fds(ts1: np.ndarray, ts2: np.ndarray, a: float = NONLINEAR) -> float:
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


def expand_subgraph_hybrid_oss(G, node_dict, node_x):
    """
    Exact subtree convolution with state dominance filtering.
    Always does full enumeration (no beam search).
    Returns:
        (selected_nodes, best_g_value)
    """
    try:
        descendants = nx.descendants(G, node_x)
    except Exception:
        return set(), 0.0

    target_nodes = descendants.union({node_x})

    # Precompute sim(node_x, ·)
    sim_cache = {}
    ts_x = node_dict[node_x][0]
    for other in target_nodes:
        if other == node_x:
            sim_cache[other] = 1.0
        else:
            ts_other = node_dict[other][0]
            sim_cache[other] = sim(ts_x, ts_other)

    # state table: node -> {(s, t): (sim_sum, sel)}
    oss: Dict[Hashable, Dict[Tuple[int, float], Tuple[float, List]]] = {}

    def dominance_filter(table):
        """Remove dominated states under the 3D rule."""
        items = []
        for (s, t), (sim_sum, sel) in table.items():
            items.append((s, t, sim_sum, sel))
        # order: s↑, t↓, sim↓
        items.sort(key=lambda x: (x[0], -x[1], -x[2]))

        kept = []
        for s, t, sim_sum, sel in items:
            dominated = False
            for s2, t2, sim2, _ in kept:
                if (s2 <= s) and (t2 >= t) and (sim2 >= sim_sum) and (
                    (s2 < s) or (t2 > t) or (sim2 > sim_sum)
                ):
                    dominated = True
                    break
            if not dominated:
                kept.append((s, t, sim_sum, sel))

        out = {}
        for s, t, sim_sum, sel in kept:
            prev = out.get((s, t))
            if (prev is None) or (sim_sum > prev[0]):
                out[(s, t)] = (sim_sum, sel)
        return out

    def merge_with_child(cur, oss_u, u):
        """
        Merge current table with child table oss_u.
        Keep skip option, then do full cross merging, then filter.
        """
        next_conf = {}
        # skip
        for (s1, t1), (sim1, sel1) in cur.items():
            prev = next_conf.get((s1, t1))
            if (prev is None) or (sim1 > prev[0]):
                next_conf[(s1, t1)] = (sim1, sel1)

        # full enumeration
        for (s1, t1), (sim1, sel1) in cur.items():
            for (s2, t2), (sim2, _sel2) in oss_u.items():
                s_ = s1 + s2
                t_ = t1 + t2
                sim_ = sim1 + sim2
                sel_ = sel1 + [(u, (s2, t2))]
                prev = next_conf.get((s_, t_))
                if (prev is None) or (sim_ > prev[0]):
                    next_conf[(s_, t_)] = (sim_, sel_)

        return dominance_filter(next_conf)

    def build(x):
        children = [u for u in G.successors(x) if u in target_nodes]
        for u in children:
            build(u)

        base = {(1, node_dict[x][1]): (sim_cache[x], [])}
        cur = base
        for u in children:
            cur = merge_with_child(cur, oss[u], u)

        oss[x] = cur

    def reconstruct(x, key):
        S = {x}
        _, sel_list = oss[x][key]
        for (u, key_u) in sel_list:
            S |= reconstruct(u, key_u)
        return S

    build(node_x)

    # choose best
    best_key = None
    best_score = float("-inf")
    for (s, t), (sim_sum, _sel) in oss[node_x].items():
        score = (sim_sum * t) / float(s)
        if score > best_score:
            best_score = score
            best_key = (s, t)

    if best_key is None:
        return {node_x}, (sim_cache[node_x] * node_dict[node_x][1]) / 1.0

    current_subgraph_nodes = reconstruct(node_x, best_key)
    current_g_value = best_score
    return current_subgraph_nodes, current_g_value

def find_k_best_subgraphs_lazy(
    G: nx.DiGraph,
    node_dict: NodeDict,
    k: int,
) -> Tuple[List[Tuple[Set[Hashable], float]], float]:
    """
    Select up to k non-overlapping subgraphs using a LAZY max-heap over seeds.

    Core ideas:
      - Work on a mutable copy G_unused.
      - Maintain a cache: seed -> (sub_nodes, g_value).
      - Maintain a version counter per seed for lazy invalidation.
      - Use a max-heap of entries (-g_value, str(seed), version) to always pop
        the current best. Discard heap entries if they are stale (version mismatch),
        disappeared from the graph, or no longer in node_dict.

      - After selecting a subgraph:
          * Compute union of its ancestors in G_unused (BEFORE removal),
            and mark these seeds as "affected" → recompute and push fresh entries.
          * Remove selected nodes from G_unused to enforce disjointness.
          * Drop cache entries for seeds that disappeared.

      - Early stop if the current best g ≤ 0.

    Returns:
        (subgraphs, total_g),
        where subgraphs = [({nodes}, g_value), ...] in selection order.
    """
    if k <= 0:
        return [], 0.0

    results: List[Tuple[Set[Hashable], float]] = []
    total_g = 0.0

    G_unused = G.copy()
    cache: Dict[Hashable, Tuple[Set[Hashable], float]] = {}     # seed -> (nodes, g)
    version: Dict[Hashable, int] = {}                           # seed -> int
    heap: List[Tuple[float, str, int, Hashable]] = []           # (-g, str(seed), ver, seed)

    def recompute_seed(seed: Hashable) -> None:
        """Recompute (sub_nodes, g) for `seed` on current G_unused and push to heap."""
        if (seed not in node_dict) or (not G_unused.has_node(seed)):
            return
        sub_nodes, g_value = expand_subgraph_pgreedy_tree(G_unused, node_dict, seed)
        cache[seed] = (sub_nodes, g_value)
        version[seed] = version.get(seed, 0) + 1
        heapq.heappush(heap, (-float(g_value), str(seed), version[seed], seed))

    # Initialize: compute once for all valid seeds (lazy heap mainly saves re-scans across rounds)
    for seed in list(G_unused.nodes):
        if seed in node_dict:
            recompute_seed(seed)

    rounds = 0
    while heap and rounds < k:
        # Pop until a valid top is found
        while heap:
            neg_g, seed_str, ver, seed = heap[0]
            # Validate seed presence and version
            if (seed not in node_dict) or (not G_unused.has_node(seed)) or (version.get(seed, -1) != ver):
                heapq.heappop(heap)  # stale entry
                continue
            # Pull current cached value
            sub_nodes, g_value = cache.get(seed, (set(), float("-inf")))
            # If cached vanished or seed not present anymore, discard
            if not sub_nodes and g_value == float("-inf"):
                heapq.heappop(heap)
                continue
            break

        if not heap:
            break

        # Peek is valid; check the best
        neg_g, seed_str, ver, seed = heapq.heappop(heap)
        sub_nodes, g_value = cache[seed]

        if g_value <= 0.0 + EPS:
            break  # early stop

        # Accept this subgraph
        results.append((sub_nodes, float(g_value)))
        total_g += float(g_value)
        rounds += 1

        # Gather ancestors BEFORE removal (relative to current G_unused)
        affected_ancestors: Set[Hashable] = set()
        for n in sub_nodes:
            if G_unused.has_node(n):
                affected_ancestors |= nx.ancestors(G_unused, n)

        # Remove selected nodes (enforce non-overlap)
        G_unused.remove_nodes_from(sub_nodes)

        # Invalidate cache entries that disappeared; keep others
        for s in list(cache.keys()):
            if (s in sub_nodes) or (not G_unused.has_node(s)):
                cache.pop(s, None)
                version.pop(s, None)  # drop version; any heap entries for it will be treated stale

        # Recompute affected seeds that still exist
        for s in sorted(affected_ancestors, key=lambda x: str(x)):
            if G_unused.has_node(s) and s in node_dict:
                recompute_seed(s)

        # Also, some seeds may still be in heap with old versions; they will be lazily discarded.

    return results, float(total_g)
