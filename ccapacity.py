"""Correlated capacity: classification capacity of the complex
outer-product holographic memory under correlated pattern families.

Same one-shot protocol as membrane-capacity (copied, not imported):
  - N oscillators, P patterns stored via outer product.
  - Probe = stored pattern + complex Gaussian noise (sigma).
  - Classifier: argmax |overlap| (one-shot project -- iterate and the
    answer dissolves, per the recall-drift finding).
  - P90 = largest P with accuracy >= 0.90.

Families:
  - "shared": p_i = normalize(sqrt(1-rho)*r_i + sqrt(rho*2N)*c),
    r_i i.i.d. complex Gaussian, c one fixed unit direction.
    rho is the ENERGY FRACTION in the shared direction, so
    E|<p_i|p_j>|^2 ~= (1-rho^2)/N + rho^2. Measured, not assumed.
  - "rankR": p_i confined to a fixed random rank-R subspace;
    E|<p_i|p_j>|^2 = 1/R.

All seeded, stdlib only.
"""

import math
import random


def normalize(v):
    n = math.sqrt(sum(abs(c) ** 2 for c in v))
    return [c / n for c in v]


def overlap(field, pattern):
    return sum(p.conjugate() * f for p, f in zip(pattern, field))


def noisy(state, sigma, rng):
    return normalize([c + complex(rng.gauss(0, sigma), rng.gauss(0, sigma))
                      for c in state])


def rand_unit(n, rng):
    return normalize([complex(rng.gauss(0, 1), rng.gauss(0, 1))
                      for _ in range(n)])


def shared_direction_family(n, p, rho, seed):
    """Correlated family: every pattern carries energy fraction rho
    in one fixed shared direction c."""
    rng = random.Random(seed)
    c = rand_unit(n, rng)
    scale_c = math.sqrt(rho * 2 * n)   # E||r_i||^2 = 2n
    scale_r = math.sqrt(1.0 - rho)
    pats = []
    for _ in range(p):
        r = [complex(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(n)]
        pats.append(normalize([scale_r * ri + scale_c * ci
                               for ri, ci in zip(r, c)]))
    return pats


def rankr_family(n, p, rank, seed):
    """Patterns confined to one fixed random rank-`rank` subspace."""
    rng = random.Random(seed)
    basis = [rand_unit(n, rng) for _ in range(rank)]
    # orthonormalize (modified Gram-Schmidt)
    Q = []
    for v in basis:
        for q in Q:
            ip = sum(qi.conjugate() * vi for qi, vi in zip(q, v))
            v = [vi - ip * qi for vi, qi in zip(v, q)]
        nv = math.sqrt(sum(abs(x) ** 2 for x in v))
        if nv > 1e-12:
            Q.append([x / nv for x in v])
    pats = []
    for _ in range(p):
        z = [complex(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in Q]
        pats.append(normalize([sum(zi * qi for zi, qi in zip(z, row))
                               for row in zip(*Q)]))
    return pats


def mean_pair_overlap2(patterns):
    """Mean |<a|b>|^2 over distinct pairs."""
    tot, cnt = 0.0, 0
    for i in range(len(patterns)):
        for j in range(i + 1, len(patterns)):
            tot += abs(overlap(patterns[i], patterns[j])) ** 2
            cnt += 1
    return tot / cnt


def theory_shared_overlap2(n, rho):
    """E|<p_i|p_j>|^2 ~= (1-rho^2)/N + rho^2."""
    return (1.0 - rho * rho) / n + rho * rho


class HolographicMembrane:
    def __init__(self, patterns):
        self.patterns = [normalize(p) for p in patterns]

    def project(self, field):
        ms = [overlap(field, p) for p in self.patterns]
        winner = max(range(len(ms)), key=lambda i: abs(ms[i]))
        return winner


def make_family(family, n, p, seed, **kw):
    if family == "shared":
        return shared_direction_family(n, p, kw["rho"], seed)
    if family == "rankR":
        return rankr_family(n, p, kw["rank"], seed)
    raise ValueError(family)


def trial(family, n, p, sigma, seed, **kw):
    rng = random.Random(seed)
    patterns = make_family(family, n, p, rng.randrange(2 ** 30), **kw)
    mem = HolographicMembrane(patterns)
    target = rng.randrange(p)
    probe = noisy(patterns[target], sigma, rng)
    return mem.project(probe) == target


def accuracy(family, n, p, sigma, trials=200, seed0=0, **kw):
    ok = sum(trial(family, n, p, sigma, seed0 + i, **kw)
             for i in range(trials))
    return ok / trials


def p90(curve):
    """Largest P on the [(P, acc)] curve with acc >= 0.90, else None."""
    ok = [p for p, a in curve if a >= 0.90]
    return max(ok) if ok else None
