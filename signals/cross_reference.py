"""
Cross-reference consensus module.

Takes already-fetched cross_platform_prices (from cross_platform.py, no extra API calls)
and computes an agreement score across Polymarket + external platforms.

Integration point: forecaster.py calls apply_consensus_shrinkage() after generating
a probability when Config.ENABLE_CROSS_REFERENCE=true.

Shrinkage rule (from CLAUDE.md Phase 2A spec):
  - max_divergence > 0.15: any platform disagrees significantly -> shrink 20% toward 0.5
  - max_divergence <= 0.05: all sources agree -> no shrinkage
  - no external data: no shrinkage
"""

import statistics
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class CrossReferenceResult:
    metaculus_prob: Optional[float]
    manifold_prob: Optional[float]
    kalshi_prob: Optional[float]
    available_sources: int        # number of external platforms with data
    agreement_score: float        # 0.0 (total disagreement) to 1.0 (perfect agreement)
    consensus_prob: Optional[float]  # mean of all available probabilities
    max_divergence: float         # max deviation from consensus


def cross_reference_market(
    polymarket_prob: float,
    cross_platform_prices: Dict[str, float],
) -> CrossReferenceResult:
    """
    Compute agreement score from already-fetched cross-platform prices.

    Args:
        polymarket_prob: The forecaster's estimated probability for YES
        cross_platform_prices: Dict from cross_platform.get_cross_platform_prices()

    Agreement score: 1.0 = perfect agreement, 0.0 = maximum disagreement.
    Uses all available sources including Polymarket's own forecast.
    """
    metaculus = cross_platform_prices.get("metaculus")
    manifold = cross_platform_prices.get("manifold")
    kalshi = cross_platform_prices.get("kalshi")

    external_probs = [p for p in [metaculus, manifold, kalshi] if p is not None]
    all_probs = [polymarket_prob] + external_probs

    if len(all_probs) < 2:
        return CrossReferenceResult(
            metaculus_prob=metaculus,
            manifold_prob=manifold,
            kalshi_prob=kalshi,
            available_sources=len(external_probs),
            agreement_score=1.0,  # no external data = no disagreement signal
            consensus_prob=polymarket_prob,
            max_divergence=0.0,
        )

    mean_prob = statistics.mean(all_probs)
    max_dev = max(abs(p - mean_prob) for p in all_probs)

    # agreement_score: 1.0 when max_deviation=0, 0.0 when max_deviation=0.5
    agreement_score = max(0.0, 1.0 - max_dev * 2.0)

    return CrossReferenceResult(
        metaculus_prob=metaculus,
        manifold_prob=manifold,
        kalshi_prob=kalshi,
        available_sources=len(external_probs),
        agreement_score=round(agreement_score, 4),
        consensus_prob=round(mean_prob, 4),
        max_divergence=round(max_dev, 4),
    )


def apply_consensus_shrinkage(probability: float, result: CrossReferenceResult) -> float:
    """
    Shrink probability toward 0.5 when platforms disagree significantly.

    Rules (from CLAUDE.md Phase 2A spec):
      max_divergence > 0.15: shrink 20% toward 0.5
      max_divergence <= 0.05: no shrinkage (strong consensus)
      no external data: no shrinkage
    """
    if result.available_sources == 0:
        return probability

    if result.max_divergence > 0.15:
        shrinkage = 0.20
        probability = probability * (1 - shrinkage) + 0.5 * shrinkage

    return round(probability, 4)
