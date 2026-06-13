# federated-ternary

**Federated learning over ternary weight spaces** — multiple nodes train {-1, 0, +1} weight vectors locally, then merge via element-wise majority vote. The merge operation is commutative, associative, and idempotent (CRDT properties), making it Byzantine-tolerant without coordination.

## Why It Matters

Standard federated learning (FedAvg) averages continuous-valued weights — but averaging is vulnerable to Byzantine participants (a single malicious node can shift the average arbitrarily). Ternary federated learning replaces averaging with **majority voting**, which is far more robust:

- **Byzantine tolerance**: With n nodes and f < n/2 Byzantine, the majority still converges to the correct weights. FedAvg requires robust aggregation (Krum, median-of-means) to achieve similar guarantees.
- **Communication efficiency**: Each weight is a single trit (2 bits) vs. 32 bits for FP32. A 1-billion-parameter model transmits 250 MB vs. 4 GB per round.
- **CRDT merge semantics**: The majority-vote merge is commutative (order-independent), associative (grouping-independent), and idempotent (re-merging is a no-op). This means nodes can merge in any order, over unreliable networks, with duplicate messages — and they still converge.

## How It Works

### Local Training

Each node trains its ternary weight vector toward a target using gradient-free updates:

$$w_i' = \text{clip}_{[-1, +1]}\left(w_i + \text{round}\left((t_i - w_i) \times \eta\right)\right)$$

Where t_i is the target weight, η is the learning rate, and the result is clipped to {-1, 0, +1}. The `round()` function ensures ternary output.

**Complexity**: O(L) per training step, where L = weight vector length.

### Majority Merge (CRDT)

Given n weight vectors, the merge computes element-wise:

$$w_i^{merged} = \text{sign}\left(\sum_{j=1}^{n} w_i^{(j)}\right) = \begin{cases} +1 & \text{if } \sum > 0 \\ -1 & \text{if } \sum < 0 \\ 0 & \text{if } \sum = 0 \end{cases}$$

This is the **majority function** — the most common non-zero value wins, with ties resolving to 0.

### CRDT Properties

**Commutative**: `merge(A, B) = merge(B, A)` — order doesn't matter.

**Associative**: `merge(merge(A, B), C) = merge(A, merge(B, C))` — grouping doesn't matter.

**Idempotent**: `merge(X, X) = X` — duplicate merges are safe.

These three properties guarantee eventual convergence: regardless of network topology, message ordering, or duplication, all nodes that receive the same set of updates converge to identical state.

### Byzantine Tolerance Analysis

Suppose f out of n nodes are Byzantine (adversarial). For a single weight position:

- Honest nodes produce the same target value t ∈ {-1, 0, +1}
- Byzantine nodes produce arbitrary values b ∈ {-1, 0, +1}

The majority is correct if:

$$n_{honest}^{+} > \frac{n}{2} \quad \text{or} \quad n_{honest}^{-} > \frac{n}{2}$$

This holds when honest nodes agree and f < n/2. With 7 nodes and 2 Byzantine (29% Byzantine), the 5 honest nodes always outvote the 2 attackers.

### Convergence Tracking

After each round, accuracy is measured as:

$$\text{accuracy} = \frac{|\{i : w_i^{merged} = t_i\}|}{L}$$

The crate also detects convergence by checking if consecutive merged vectors are identical — a sign that training has stabilized.

### Complexity Analysis

| Operation | Time | Space |
|-----------|------|-------|
| `train_step` | O(L) | O(1) |
| `majority_merge` | O(n × L) | O(L) |
| `round` (train + merge + redistribute) | O(n × L) | O(L) |
| `train` (R rounds) | O(R × n × L) | O(R) for accuracy history |

Where n = node count, L = weight vector length, R = rounds.

## Quick Start

```rust
use federated_ternary::*;

// 5 nodes, 8 weights each
let mut fed = FederatedRound::new(5, 8);
let target = vec![1, -1, 1, -1, 0, 0, 1, -1];

// Train for 10 rounds
let accuracies = fed.train(&target, 10);
println!("Final accuracy: {:.1}%", accuracies.last().unwrap() * 100.0);

// Byzantine tolerance: mark 2 of 7 nodes as adversarial
let mut fed2 = FederatedRound::new(7, 4);
fed2.set_byzantine(&[5, 6]);
let acc = fed2.train(&vec![1, 1, 1, 1], 5);
// Honest majority (5/7) still converges
```

## API

### `TernaryWeights`
- `new(node_id, values: Vec<i8>) -> Self`
- `random_biased(node_id, len, bias: i8) -> Self`
- `train_step(&mut self, target: &[i8], lr: f64)` — Gradient-free ternary update
- `len() -> usize`, `is_empty() -> bool`

### `FederatedRound`
- `new(node_count, weight_len) -> Self`
- `set_byzantine(&mut self, ids: &[u32])` — Mark nodes as adversarial
- `round(&mut self, target: &[i8]) -> TernaryWeights` — One federated round
- `train(&mut self, target, rounds) -> Vec<f64>` — Multiple rounds with accuracy tracking
- `convergence_round(&self) -> Option<usize>` — First round where merge stabilized

### Free Functions
- `majority_merge(weights: &[&TernaryWeights]) -> TernaryWeights` — CRDT merge
- `accuracy(predicted: &[i8], target: &[i8]) -> f64` — Fraction of matching weights

## Architecture Notes

The federated ternary merge implements the γ + η = C conservation link:

- **γ** (gamma) = the merged global weight vector (agreement)
- **η** (eta) = per-node deviations from the global vector (disagreement)
- **C** (constant) = the target weight vector

As training proceeds, γ → C and η → 0. The rate of convergence depends on the learning rate, Byzantine fraction, and the difficulty of the target pattern.

See the full architecture: [ARCHITECTURE.md](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md)

## References

1. McMahan, B., et al. (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data." *AISTATS 2017.* — FedAvg, the original federated learning paper.
2. Blanchard, P., et al. (2017). "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent." *NeurIPS 2017.* — Krum aggregation for Byzantine robustness.
3. Wang, H., et al. (2023). "BitNet: Scaling 1-bit Transformers for Large Language Models." *arXiv:2310.11453.* — Ternary weight quantization.
4. Shapiro, M., et al. (2011). "A Comprehensive Study of Convergent and Commutative Replicated Data Types." *INRIA RR-7506.* — CRDT formal framework.
5. Lamport, L., Shostak, R., & Pease, M. (1982). "The Byzantine Generals Problem." *ACM TOPLAS, 4(3).* — Byzantine fault tolerance foundations.

## License

Apache-2.0
