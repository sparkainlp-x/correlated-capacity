# Correlated Capacity

**The realistic capacity bound: how classification capacity of a
holographic membrane collapses under correlated pattern families.**

Same one-shot protocol as [`membrane-capacity`](https://github.com/sparkainlp-x/membrane-capacity) (copied, not
imported): N=64 oscillators, P patterns stored by outer product,
probe = stored pattern + noise, classifier = argmax |overlap|,
P90 = largest P with accuracy ≥ 0.90. Stdlib-only Python. AGPL-3.0.

Families: a shared-direction family (energy fraction ρ in one
fixed direction; measured mean |⟨a|b⟩|² ≈ (1−ρ²)/N + ρ²) and a
rank-8 subspace family (mean |⟨a|b⟩|² = 1/8).

> Synthetic toy simulation; classical software; not physical holography
> hardware. "Oscillators" and "membrane" are classically simulated
> complex vectors.

## Measured in this repo (SYNTHETIC, seeded)

- Correlation calibration matches theory to <2% (measured, not
  assumed)
- ρ=0 column reproduces membrane-capacity: P90 > 1024 at σ=0.15,
  tiny capacity at σ=0.30
- P90 never rises with ρ (non-increasing: >1024, >1024, 128, <4 at
  σ=0.15; 4, <4, <4, <4 at σ=0.30)
- 9/9 tests pass

**Uncertainty.** These are single seeded runs of `demo.py` with no repeated seeds and no interval, so "matches theory to <2%" and "P90 never rises with ρ" describe one draw per cell. P90 is read off a coarse grid, which is why several entries are bounds (">1024", "<4"). Spread across seeds would need new runs and is not measured here.

## Run it

```bash
python tests.py   # 9 tests
python demo.py    # calibration check + P90 table (a few minutes)
```

## Honest limits

One-shot classification only (not recall); P90's 0.90 threshold is
a reporting choice; toy scale (N=64); the shared-direction family
is one correlation model, not all of them; ρ is an energy fraction
— see DESIGN.md for the calibration choice. Full account there.
