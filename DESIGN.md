# DESIGN — Correlated capacity: the realistic bound

## The question

`membrane-capacity` measured classification capacity with random
(near-orthogonal) patterns — the easy, best-case bound. Real
pattern families are correlated. How does P90 degrade?

## Family construction

**Shared-direction family.** `p_i = normalize(√(1−ρ)·r_i +
√(ρ·2N)·c)`: every pattern carries energy fraction ρ in one fixed
direction c, the rest random. ρ ∈ {0, 0.3, 0.6, 0.9}.

A note on the brief's sketch (`p_i = normalize(r_i + ρ·c)`): with
ρ as a bare amplitude against ‖r_i‖² ≈ 2N = 128, even ρ=0.9 would
move mean |overlap|² by 0.0001 — the experiment would have measured
nothing. So ρ is calibrated as the *energy fraction* in the shared
direction, giving E|<p_i|p_j>|² ≈ (1−ρ²)/N + ρ². The calibration is
measured, not assumed:

| ρ   | measured mean |⟨a|b⟩|² | theory |
|-----|----------------------|--------|
| 0.0 | 0.0156               | 0.0156 |
| 0.3 | 0.1062               | 0.1042 |
| 0.6 | 0.3734               | 0.3700 |
| 0.9 | 0.8146               | 0.8130 |

**Rank-R family.** Patterns confined to a fixed random 8-dim
subspace (N=64): measured mean |overlap|² = 0.1238 ≈ 1/8. One
structured-family data point.

## Measured P90 (N=64; P90 = largest P with accuracy ≥ 0.90)

| family | σ=0.15 | σ=0.30 |
|--------|--------|--------|
| ρ=0.0  | >1024  | 4      |
| ρ=0.3  | >1024  | <4     |
| ρ=0.6  | 128    | <4     |
| ρ=0.9  | <4     | <4     |
| rank-8 | 512    | <4     |

The ρ=0 column reproduces `membrane-capacity` exactly (>1024 /
4) — the sanity anchor holds.

## The realistic-bound interpretation

- **Random patterns were the best case, and the bound is tight
  only there.** At σ=0.15, mild correlation (ρ=0.3, mean
  overlap² 0.10) costs nothing — P90 stays >1024. The degradation
  has a knee: ρ=0.6 (overlap² 0.37) drops P90 to 128, ρ=0.9
  (overlap² 0.81) to <4.
- **At high noise there is no headroom to lose.** σ=0.30 already
  gives P90=4 for random patterns; any correlation pushes it
  below 4. Noise, not crosstalk, is the binding constraint — same
  conclusion as the reference work, sharpened.
- **The rank-8 point lands where effective correlation predicts:**
  its mean overlap² (0.124) sits between ρ=0.3 (0.104) and ρ=0.6
  (0.370), and its P90 (512) sits between theirs (>1024, 128).
  Mean pairwise |overlap|² is a decent single-number predictor of
  capacity across family *shapes*, not just within one family.

## The bug the measurement caught

First demo run printed every ρ row identical. The results dict
was keyed `(family, sigma, P)` — family was `"shared"` for all ρ,
so each ρ overwrote the last and all rows showed ρ=0.9. Keying
now includes the family parameters. The numbers above are from
the fixed run.

## Honest limits

- **One-shot classification only**, not recall; argmax-overlap,
  the recommended classifier per the recall-drift finding.
- **P90's 0.90 threshold is a reporting choice.** The full curves
  are in the demo output; a different threshold moves the
  numbers, not the ordering.
- **Toy scale** (N=64). The shared-direction family is one
  correlation model — real families have richer structure
  (clusters, hierarchies); the rank-8 point is one step toward
  that, not the destination.
- **ρ is an energy fraction**, a calibration choice documented
  above — not the brief's literal amplitude sketch, which would
  have produced no measurable correlation at these values.
- 200 trials for P≤128, 100 for P≥256: the large-P cells carry
  ±0.05-ish binomial noise; P90 boundaries near 0.90 should be
  read with that in mind.
