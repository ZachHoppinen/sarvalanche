# v3 Single-Pair Debris Detector — Notes

## Why CNN over thresholding?

Tested melt-filtered d_empirical + manual thresholds + FlowPy/size/glacier filters
against the v2 CNN (human_only_4ch) on the same held-out validation set (87 obs,
85 matched, after 2026-01-28).

| Metric | Threshold | CNN | CNN advantage |
|--------|-----------|-----|---------------|
| Det@0.2 | 39% | 44% | +5% |
| F1@0.2 | 0.101 | 0.200 | **2× better** |
| FPR@0.2 | 39.4% | 14.5% | **2.7× fewer FP** |
| FPR@0.5 | 22.0% | 7.2% | **3.1× fewer FP** |

Detection by D-size (@0.2 threshold):

| D-size | Threshold | CNN |
|--------|-----------|-----|
| D1 | 43% | 29% |
| D1.5 | 29% | 57% |
| D2 | 23% | 46% |
| D2.5 | 20% | 67% |
| D3 | **0%** | **100%** |

**The CNN earns its keep through precision, not recall.** Similar overall detection
rates but 2.7× fewer false positives. The threshold fires on small noisy patches
everywhere (39% FPR) while the CNN focuses on terrain-plausible debris deposits.

The D3=0% vs 100% result is striking — the threshold approach misses large avalanches
entirely because its "probability" is just a scaled d_empirical value that doesn't
account for morphology. The CNN learns that large contiguous bright regions in runout
zones on appropriate slopes are debris.

**Conclusion:** The CNN adds real value through spatial context and terrain morphology.
Simple thresholding + filtering is a useful baseline but not sufficient for operational
detection. Proceed with v3 single-pair architecture.

## Cross-site transfer test (TODO)

Train v3 on AK, run on Idaho (SNFAC) without retraining. Critical for understanding
whether v2's SNFAC pretrain failure indicates overfitting or genuine physical
differences between snow climates. If v3's simpler per-pair evaluation transfers
better than v2's attention architecture, that validates the design.

## Architecture decisions

- **base_ch=16** recommended (~930k params). base_ch=32 gives 3.7M params which is
  too large for ~21k training samples.
- **20 static channels** (7 SAR + 13 static) per forward pass
- Curvature and TPI computation has a signature bug — needs fixing before production
- Auto-labeling dropped — consistently hurts in every experiment
- SNFAC pretrain dropped — domain gap too large for AK
