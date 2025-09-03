# Subgraph Selection Algorithms

This repository implements three strategies for selecting rooted subgraphs in directed trees or DAGs.  
Each node is represented as:

node_dict[node_id] = (time_series: np.ndarray, value: float, extra_info: dict)

The optimization objective for a node set **S** rooted at **r** is:

\[
g(S) = \frac{\left(\sum_{v\in S}\text{sim}(ts_r, ts_v)\right)\left(\sum_{v\in S}\text{value}_v\right)}{|S|}
\]

where `sim(ts_r, ts_v)` is the similarity between the root’s time series and node v’s time series.

---

## Similarity Function

**Function**: `sim(ts1, ts2, a=0.9)`

- Frequency-Domain Similarity (FDS) with monotonic mapping.  
- **Steps**:  
  1. Mean-center both series.  
  2. Take FFT magnitudes (first half spectrum).  
  3. Compute cosine similarity.  
  4. Map via `(s - a) / (1 - a*s)` (default `a = 0.9`).  
- **Returns**: float in range `[-1, 1]`.

---

## V-Greedy

**Library name**: `V-Greedy`  

**Functions**:
- `expand_subgraph_greedy(G, node_dict, root, sim)`
- `find_k_best_subgraphs(G, node_dict, k, sim)`

**Description**:
- Greedy expansion from the root, considering only one-step frontier nodes (direct successors).  
- Iteratively expands while the score improves.  
- `find_k_best_subgraphs` applies the greedy expansion up to **k** times, selecting disjoint subgraphs. After each pick, incident edges are removed to avoid overlap.  
- **Advantages**: simple, fast heuristic.

---

## P-Greedy

**Library name**: `P-Greedy`  

**Functions**:
- `expand_subgraph_pgreedy_tree(G, node_dict, root)`
- `find_k_best_subgraphs_lazy(G, node_dict, k)`

**Description**:
- Works on trees with a global candidate set.  
- At each iteration, evaluates for every candidate **u** the entire path suffix \(A_u\) (from boundary node **b** to **u**).  
- Selects the \(A_u\) that maximizes score gain.  
- Expansion continues until no candidate improves the score.  
- `find_k_best_subgraphs_lazy` manages multiple subgraphs using a lazy max-heap with caching:  
  - Keeps a working copy of the graph.  
  - Seeds are cached with version numbers.  
  - Uses lazy invalidation of heap entries.  
  - After selecting a subgraph, recomputes only ancestors of selected nodes.  
- **Advantages**: more global and path-aware than V-Greedy, still efficient for trees.

---

## OSS

**Library name**: `OSS`  

**Function**:
- `expand_subgraph_hybrid_oss(G, node_dict, root)`

**Description**:
- Exact bottom-up dynamic programming on a single root’s subtree.  
- Each node stores a state table:  
  `(count s, sum_value t) → (sum_sim, selection)`  
- For each child, merge its state table with the parent’s by full cross enumeration.  
- Apply dominance filtering: keep `(s, t, sim)` only if no other state strictly dominates it (`s' <= s`, `t' >= t`, `sim' >= sim`).  
- Reconstruct the selected node set from the best key.  
- **Advantages**: produces the exact optimal solution for a given root, but computationally heavier on large subtrees.

---

## Comparison

- **V-Greedy**: fastest, lightweight heuristic; local frontier only.  
- **P-Greedy**: path-aware greedy; considers entire root-to-node suffix; better than V-Greedy on trees.  
- **OSS**: exact, but heavier on large subtrees.  

---

## Notes

- Node values may be preprocessed (e.g., log transform: `log2(value + 1)`) before running.  
- Tie-breaking is deterministic: candidates are ordered by `str(node_id)`.  
- **Guidelines**:  
  - Use V-Greedy for quick approximate results.  
  - Use P-Greedy for tree-structured data and better quality.  
  - Use OSS when exact optimality is required.
