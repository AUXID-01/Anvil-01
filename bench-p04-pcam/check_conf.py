
import numpy as np
from pcam_model import PCAMModel, build_default_R
from data import make_patterns, make_test_queries
from adapters.myteam import Engine

seeds = [42, 101, 202]
noise_levels = [0.75, 0.85]
n_queries = 50

for seed in seeds:
    X = make_patterns(K=16, N=64, seed=seed)
    R = build_default_R(N=64, seed=seed)
    model = PCAMModel(X, R)
    queries, truths, _ = make_test_queries(X, noise_levels, n_queries, seed=seed)
    
    confidences = []
    for q in queries:
        nq = np.linalg.norm(q)
        q_hat = q / nq if nq > 1e-12 else q
        sims = X @ q_hat
        logits = 8.0 * sims
        logits -= logits.max()
        exp_logits = np.exp(logits)
        softmax = exp_logits / exp_logits.sum()
        confidences.append(np.max(softmax))
    
    print(f"Seed {seed}: Mean Confidence = {np.mean(confidences):.3f}, Min = {np.min(confidences):.3f}, Max = {np.max(confidences):.3f}")
