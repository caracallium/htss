# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 17:50:55 2025

@author: wwj19
"""

import networkx as nx
import numpy as np
from typing import Dict, Hashable, List, Optional, Set, Tuple


def sim(ts1, ts2, a: float = 0.9):
    f = np.asarray(ts1, dtype=np.float64) - float(np.mean(ts1))
    g = np.asarray(ts2, dtype=np.float64) - float(np.mean(ts2))
    F = np.abs(np.fft.fft(f))[: len(f) // 2]
    G = np.abs(np.fft.fft(g))[: len(g) // 2]
    nF, nG = np.linalg.norm(F), np.linalg.norm(G)
    if nF == 0.0 or nG == 0.0:
        return 0.0
    s = float(np.dot(F / nF, G / nG))
    return (s - a) / (1.0 - a * s)


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
