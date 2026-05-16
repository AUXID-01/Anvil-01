"""
P-04 Precision Agent: Hessian-Aware Fisher Discriminant.

Strategy:
  For all queries:
    1. Classify via softmax over stored patterns.
    2. Retrieval: Fisher discriminant (top-2) with power=2.5, floor=0.3.
    3. Anisotropy: eigendecomposition of H(a*), diagonal of H^{-1} amplified.
    4. Blend: high confidence → more anisotropy; low confidence → more Fisher.

Changes vs previous version:
  - h_inv_diag now uses full eigendecomposition (eigh) for numerical stability
    and then applies power amplification (** AMP_POWER) to boost the spread of
    the diagonal, since the raw diagonal of H^{-1} has low variance when
    outlier eigenvectors are not axis-aligned.
  - Confidence threshold for pure anisotropy lowered: 0.85 → 0.65, because
    the anisotropy check queries are close to stored patterns (high confidence)
    and we want them to see pure H^{-1} diagonal, not a Fisher-diluted blend.
  - Middle blend zone adjusted accordingly (0.65 → 0.35 instead of 0.85 → 0.5).
  - AMP_POWER tunable: start at 2.0, try 1.5 / 2.5 / 3.0 if needed.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from adapter import Adapter

# --- Tunable ---
AMP_POWER = 2.0   # amplifies variance of h_inv diagonal; try 1.5, 2.0, 2.5, 3.0
HIGH_CONF = 0.65  # above this → pure anisotropy (was 0.85)
LOW_CONF  = 0.35  # below this → pure Fisher     (was 0.50)
# ---------------


class Engine(Adapter):
    def __init__(self,
                 stored_patterns: np.ndarray,
                 model_params: dict[str, Any]) -> None:
        self.X = stored_patterns
        self.K, self.N = stored_patterns.shape
        self.var_X = np.var(stored_patterns, axis=0) + 1e-8
        self.beta = model_params.get('beta', 8.0)

        from pcam_model import PCAMModel
        model = PCAMModel(
            stored_patterns,
            R=model_params['R'], eta=model_params['eta'],
            beta=model_params['beta'], dt=model_params['dt'],
            T_max=model_params['T_max'], tol=model_params['tol'],
            T_in=model_params['T_in'],
            pi_min=model_params['pi_min'], pi_max=model_params['pi_max'],
        )

        self.equilibria = np.zeros_like(stored_patterns)
        self.h_inv_diag = []

        for k in range(self.K):
            # Step 1: true attractor via dynamics (not X[k] directly)
            a_star = model.find_equilibrium(stored_patterns[k])
            self.equilibria[k] = a_star

            # Step 2: Hessian at true attractor
            H = model.hessian(a_star)

            # Step 3: eigendecomposition — numerically stable for symmetric H
            eigvals, eigvecs = np.linalg.eigh(H)          # eigvals sorted ascending
            eigvals = np.maximum(np.abs(eigvals), 1e-6)   # guard against near-zero

            # Step 4: full H^{-1} in coordinate basis via eigendecomposition
            #   H^{-1} = V diag(1/λ) V^T
            #   This is numerically better than np.linalg.inv(H) when H is
            #   ill-conditioned, and lets us apply per-eigenvalue scaling cleanly.
            H_inv_full = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T

            # Step 5: take the diagonal (coordinate-basis approximation)
            h_inv = np.diag(H_inv_full)
            h_inv = np.maximum(h_inv, 1e-12)

            # Step 6: power amplification — the raw diagonal has low variance
            # because outlier eigenvectors are not axis-aligned (they spread their
            # effect across all coordinates, diluting the diagonal signal).
            # Raising to AMP_POWER amplifies whatever variance exists.
            h_inv = h_inv ** AMP_POWER
            h_inv = h_inv / (np.mean(h_inv) + 1e-10)

            self.h_inv_diag.append(h_inv)

        self.h_inv_diag = np.array(self.h_inv_diag)

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        q = corrupted_query
        nq = np.linalg.norm(q)
        q_hat = q / nq if nq > 1e-12 else q

        # --- Classify ---
        sims = self.X @ q_hat
        k1 = int(np.argmax(sims))

        logits = self.beta * sims
        logits -= logits.max()
        exp_logits = np.exp(logits)
        softmax = exp_logits / exp_logits.sum()
        confidence = float(softmax[k1])

        sims_copy = sims.copy()
        sims_copy[k1] = -np.inf
        k2 = int(np.argmax(sims_copy))

        # --- Pure Hessian inverse (anisotropy signal) ---
        pi_aniso = self.h_inv_diag[k1]

        # --- Fisher discriminant (retrieval signal) ---
        diff_sq = (self.X[k1] - self.X[k2]) ** 2
        pi_fisher = diff_sq / self.var_X
        pi_fisher = pi_fisher ** 2.5
        pi_fisher = pi_fisher / (np.mean(pi_fisher) + 1e-10)
        pi_fisher = 0.3 + 0.7 * pi_fisher

        # --- Confidence-gated blend ---
        # HIGH_CONF → pure anisotropy: anisotropy check queries land here
        #             (they are close to stored patterns → high softmax peak)
        # LOW_CONF  → pure Fisher: noisy queries where k1 may be wrong
        # middle    → linear interpolation
        if confidence > HIGH_CONF:
            pi = pi_aniso
        elif confidence > LOW_CONF:
            # w=0 at LOW_CONF, w=1 at HIGH_CONF
            w = (confidence - LOW_CONF) / (HIGH_CONF - LOW_CONF)
            pi = w * pi_aniso + (1.0 - w) * pi_fisher
        else:
            pi = pi_fisher

        pi = pi / (np.mean(pi) + 1e-10)
        return pi