# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 17:33:10 2025

@author: wwj19
"""

from typing import Any, Dict, Hashable, List, Optional, Set, Tuple
import heapq
import networkx as nx
import numpy as np

# node_dict schema: {node_id: (time_series: np.ndarray, value: float, extra_info: dict)}
NodeDict = Dict[Hashable, Tuple[np.ndarray, float, Dict[str, Any]]]

A_NONLINEAR: float = 0.9
EPS: float = 1e-12


def sim_fds(ts1: np.ndarray, ts2: np.ndarray, a: float = A_NONLINEAR) -> float:
    """
    Frequency-Domain Similarity (FDS) with a monotonic mapping to [-1, 1].

    Steps:
      1) Mean-center each time series
      2) FFT magnitudes (first half spectrum)
      3) L2-normalize and cosine similarity
      4) Mapping: (s - a) / (1 - a*s)
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


def expand_subgraph_pgreedy_tree(
    G: nx.DiGraph,
    node_dict: NodeDict,
    root: Hashable,
) -> Tuple[Set[Hashable], float]:
    """
    P-Greedy on a TREE with a global candidate set (matches the pseudocode).

    Policy:
      - Scope = descendants(root) ∪ {root}
      - Precompute cumulative triplets along the unique path from `root`:
            pref_val[u], pref_sim[u], depth[u]
        where sim is sim(root, u) and depth counts nodes on (root, u].
      - While Q != ∅:
          * For each u in Q:
                b = nearest node to u on path(root, u) that is already in S
                A_u = path(b -> u]  (suffix after b)
                Gain = Score(S ∪ A_u) - Score(S), computed via prefix diffs
          * Pick argmax; if gain ≤ 0, stop; else merge the whole A_u into S

    Objective:
        g(S) = (sum_{v in S} sim(ts_root, ts_v)) * (sum_{v in S} value_v) / |S|

    Returns:
        (selected_nodes, best_g_value)
    """
    if root not in G or root not in node_dict:
        return set(), 0.0

    # Subtree scope
    try:
        scope: Set[Hashable] = set(nx.descendants(G, root)) | {root}
    except Exception:
        scope = {root}

    H = G.subgraph(scope).copy()
    H_undirected = H.to_undirected(as_view=False)

    # Precompute sim(root, u)
    ts_root = node_dict[root][0]
    sim_cache: Dict[Hashable, float] = {root: 1.0}
    for u in scope:
        if u == root:
            continue
        sim_cache[u] = sim_fds(ts_root, node_dict[u][0])

    # Cumulative triplets via a single DFS along successors
    pref_val: Dict[Hashable, float] = {root: 0.0}
    pref_sim: Dict[Hashable, float] = {root: 0.0}
    depth: Dict[Hashable, int] = {root: 0}

    children: Dict[Hashable, List[Hashable]] = {}
    for u in scope:
        children[u] = [v for v in H.successors(u) if v in scope]

    def dfs(u: Hashable) -> None:
        for v in children[u]:
            pref_val[v] = pref_val[u] + float(node_dict[v][1])
            pref_sim[v] = pref_sim[u] + float(sim_cache[v])
            depth[v] = depth[u] + 1
            dfs(v)

    dfs(root)

    # Current subtree S and its sufficient statistics
    S: Set[Hashable] = {root}
    sum_sim, sum_val, count = 1.0, float(node_dict[root][1]), 1
    g_best = (sum_sim * sum_val) / count

    def nearest_on_path(u: Hashable) -> Hashable:
        """Return the last node on path(root, u) that already belongs to S."""
        path_ru: List[Hashable] = nx.shortest_path(H_undirected, root, u)
        last = root
        for x in path_ru:
            if x in S:
                last = x
            else:
                break
        return last

    # Global candidate set (all nodes in scope minus S)
    Q: Set[Hashable] = set(scope)
    Q.discard(root)

    while Q:
        best_u: Optional[Hashable] = None
        best_gain = float("-inf")
        best_tuple: Optional[Tuple[float, float, int]] = None  # (new_sum_sim, new_sum_val, new_count)

        for u in sorted(Q, key=lambda x: str(x)):  # deterministic order
            b = nearest_on_path(u)
            if b == u:
                continue

            ds = pref_sim[u] - pref_sim[b]
            dv = pref_val[u] - pref_val[b]
            dc = depth[u] - depth[b]
            if dc <= 0:
                continue

            new_sum_sim = sum_sim + ds
            new_sum_val = sum_val + dv
            new_count = count + dc
            new_g = (new_sum_sim * new_sum_val) / new_count
            gain = new_g - g_best

            if (gain > best_gain + EPS) or (
                abs(gain - best_gain) <= EPS and best_u is not None and str(u) < str(best_u)
            ):
                best_u = u
                best_gain = gain
                best_tuple = (new_sum_sim, new_sum_val, new_count)
            elif best_u is None:
                best_u = u
                best_gain = gain
                best_tuple = (new_sum_sim, new_sum_val, new_count)

        if best_u is None or best_tuple is None or best_gain <= 0.0 + EPS:
            break

        # Merge the suffix A_{best_u} = (b, best_u]
        b_star = nearest_on_path(best_u)
        path_r_u = nx.shortest_path(H_undirected, root, best_u)
        idx = path_r_u.index(b_star)
        to_add = path_r_u[idx + 1 :]

        for x in to_add:
            S.add(x)

        sum_sim, sum_val, count = best_tuple
        g_best = (sum_sim * sum_val) / count
        Q = set(scope) - S  # update global candidates

    return S, float(g_best)


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
