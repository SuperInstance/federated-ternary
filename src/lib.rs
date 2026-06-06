//! # federated-ternary
//!
//! Federated ternary learning: multiple nodes train ternary {-1, 0, +1} weights
//! locally, then merge via element-wise majority vote. Byzantine tolerant.

use std::collections::HashMap;

/// A ternary weight vector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TernaryWeights {
    pub values: Vec<i8>,
    pub node_id: u32,
    pub round: u32,
}

impl TernaryWeights {
    pub fn new(node_id: u32, values: Vec<i8>) -> Self {
        Self { values, node_id, round: 0 }
    }

    pub fn random_biased(node_id: u32, len: usize, bias: i8) -> Self {
        let values = (0..len).enumerate().map(|(i, _)| {
            if bias > 0 { [1, 1, 1, 0][i % 4] }
            else if bias < 0 { [-1, -1, -1, 0][i % 4] }
            else { [1, 0, -1][i % 3] }
        }).collect();
        Self { values, node_id, round: 0 }
    }

    /// Local training step: shift weights toward a target.
    pub fn train_step(&mut self, target: &[i8], lr: f64) {
        for (w, &t) in self.values.iter_mut().zip(target) {
            let shift = ((t as f64 - *w as f64) * lr).round() as i8;
            *w = (*w + shift).clamp(-1, 1);
        }
        self.round += 1;
    }

    pub fn len(&self) -> usize { self.values.len() }
    pub fn is_empty(&self) -> bool { self.values.is_empty() }
}

/// Element-wise majority vote across multiple weight sets.
pub fn majority_merge(weights: &[&TernaryWeights]) -> TernaryWeights {
    if weights.is_empty() { return TernaryWeights::new(0, vec![]); }
    let len = weights[0].values.len();
    let max_round = weights.iter().map(|w| w.round).max().unwrap_or(0);

    let values: Vec<i8> = (0..len).map(|i| {
        let mut sum = 0i32;
        for w in weights { sum += w.values[i] as i32; }
        if sum > 0 { 1 } else if sum < 0 { -1 } else { 0 }
    }).collect();

    TernaryWeights { values, node_id: u32::MAX, round: max_round + 1 }
}

/// Federated learning round: collect, merge, redistribute.
pub struct FederatedRound {
    pub nodes: Vec<TernaryWeights>,
    pub byzantine_ids: Vec<u32>,
    pub merge_history: Vec<TernaryWeights>,
}

impl FederatedRound {
    pub fn new(node_count: usize, weight_len: usize) -> Self {
        let nodes = (0..node_count)
            .map(|i| TernaryWeights::random_biased(i as u32, weight_len, 0))
            .collect();
        Self { nodes, byzantine_ids: Vec::new(), merge_history: Vec::new() }
    }

    pub fn set_byzantine(&mut self, ids: &[u32]) { self.byzantine_ids = ids.to_vec(); }

    /// Run one round: local training + merge.
    pub fn round(&mut self, target: &[i8]) -> TernaryWeights {
        // Local training
        for node in &mut self.nodes {
            if self.byzantine_ids.contains(&node.node_id) {
                // Byzantine: train toward opposite
                let anti_target: Vec<i8> = target.iter().map(|&t| -t).collect();
                node.train_step(&anti_target, 0.5);
            } else {
                node.train_step(target, 0.5);
            }
        }

        // Merge
        let refs: Vec<&TernaryWeights> = self.nodes.iter().collect();
        let merged = majority_merge(&refs);
        self.merge_history.push(merged.clone());

        // Redistribute merged weights to all nodes
        for node in &mut self.nodes {
            node.values = merged.values.clone();
        }

        merged
    }

    /// Run multiple rounds and track convergence.
    pub fn train(&mut self, target: &[i8], rounds: usize) -> Vec<f64> {
        let mut accuracies = Vec::with_capacity(rounds);
        for _ in 0..rounds {
            let merged = self.round(target);
            let accuracy = accuracy(&merged.values, target);
            accuracies.push(accuracy);
        }
        accuracies
    }

    pub fn convergence_round(&self) -> Option<usize> {
        self.merge_history.windows(2).position(|w| w[0] == w[1])
    }
}

/// Accuracy: fraction of matching ternary values.
pub fn accuracy(predicted: &[i8], target: &[i8]) -> f64 {
    if predicted.is_empty() { return 1.0; }
    let matches = predicted.iter().zip(target).filter(|(&p, &t)| p == t).count();
    matches as f64 / predicted.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_majority_merge_all_same() {
        let w1 = TernaryWeights::new(0, vec![1, -1, 0]);
        let w2 = TernaryWeights::new(1, vec![1, -1, 0]);
        let w3 = TernaryWeights::new(2, vec![1, -1, 0]);
        let merged = majority_merge(&[&w1, &w2, &w3]);
        assert_eq!(merged.values, vec![1, -1, 0]);
    }

    #[test]
    fn test_majority_merge_majority_wins() {
        let w1 = TernaryWeights::new(0, vec![1, -1, 1]);
        let w2 = TernaryWeights::new(1, vec![1, -1, -1]);
        let w3 = TernaryWeights::new(2, vec![1, 1, -1]);
        let merged = majority_merge(&[&w1, &w2, &w3]);
        assert_eq!(merged.values[0], 1); // 3/3 agree
        assert_eq!(merged.values[1], -1); // 2/3 for -1
    }

    #[test]
    fn test_merge_commutative() {
        let w1 = TernaryWeights::new(0, vec![1, -1, 0]);
        let w2 = TernaryWeights::new(1, vec![1, 1, -1]);
        let w3 = TernaryWeights::new(2, vec![-1, 1, 0]);
        let m1 = majority_merge(&[&w1, &w2, &w3]);
        let m2 = majority_merge(&[&w3, &w1, &w2]);
        assert_eq!(m1.values, m2.values);
    }

    #[test]
    fn test_merge_idempotent() {
        let w1 = TernaryWeights::new(0, vec![1, -1, 0]);
        let w2 = TernaryWeights::new(1, vec![1, -1, 0]);
        let m1 = majority_merge(&[&w1, &w2]);
        let m2 = majority_merge(&[&m1, &m1]);
        assert_eq!(m1.values, m2.values);
    }

    #[test]
    fn test_training_converges() {
        let mut fed = FederatedRound::new(5, 8);
        let target = vec![1, -1, 1, -1, 0, 0, 1, -1];
        let accuracies = fed.train(&target, 10);
        assert!(accuracies.last().unwrap() > &accuracies[0]);
    }

    #[test]
    fn test_byzantine_doesnt_prevent_progress() {
        let mut fed = FederatedRound::new(7, 4);
        fed.set_byzantine(&[5, 6]);
        let target = vec![1, 1, 1, 1];
        let accuracies = fed.train(&target, 5);
        // With 5/7 honest, should make progress (not necessarily converge perfectly)
        assert!(accuracies.last().unwrap() >= &0.5);
    }
}
