# LouvainDP Algorithm

LouvainDP is a differentially private adaptation of the Louvain method for community detection in graphs. It adds noise to the graph structure to ensure privacy while maintaining meaningful community partitions. The algorithm operates by grouping nodes into supernodes, introducing edge noise for privacy, and applying the Louvain method for modularity optimization.

---

## Key Features
- Ensures differential privacy with adjustable privacy budget (\( \epsilon \)).
- Preserves community detection quality despite noise addition.

---

## Inputs and Outputs
- **Input:** Graph \( G \), group size \( k \), privacy budget \( \epsilon \).
- **Output:** Partition \( C \), representing the graph's noisy communities.
