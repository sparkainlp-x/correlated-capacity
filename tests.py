"""Tests for correlated-capacity. Run: python tests.py"""

from ccapacity import (accuracy, mean_pair_overlap2, p90,
                       rankr_family, shared_direction_family,
                       theory_shared_overlap2)

N = 64


def close(a, b, rel=0.15):
    return abs(a - b) <= rel * b


def test_shared_family_calibration():
    # measured mean |<a|b>|^2 matches theory (1-rho^2)/N + rho^2
    for rho in (0.0, 0.3, 0.6, 0.9):
        pats = shared_direction_family(N, 200, rho, seed=11)
        m = mean_pair_overlap2(pats)
        assert close(m, theory_shared_overlap2(N, rho)), (rho, m)


def test_rho0_is_random():
    pats = shared_direction_family(N, 200, 0.0, seed=11)
    assert close(mean_pair_overlap2(pats), 1.0 / N)


def test_rankr_calibration():
    pats = rankr_family(N, 200, 8, seed=11)
    assert close(mean_pair_overlap2(pats), 1.0 / 8)


def test_determinism():
    a = shared_direction_family(N, 16, 0.6, seed=5)
    b = shared_direction_family(N, 16, 0.6, seed=5)
    assert a == b
    x = accuracy("shared", N, 32, 0.15, trials=50, seed0=3, rho=0.3)
    y = accuracy("shared", N, 32, 0.15, trials=50, seed0=3, rho=0.3)
    assert x == y


def test_anchor_low_noise():
    # reproduces membrane-capacity: sigma=0.15 -> P90 > 1024
    assert accuracy("shared", N, 1024, 0.15, trials=100,
                    seed0=0, rho=0) >= 0.90


def test_anchor_high_noise_breakdown():
    # reproduces membrane-capacity: sigma=0.30 -> tiny capacity
    assert accuracy("shared", N, 4, 0.30, trials=200,
                    seed0=0, rho=0) >= 0.90
    assert accuracy("shared", N, 128, 0.30, trials=100,
                    seed0=0, rho=0) < 0.90


def test_correlation_hurts():
    x = accuracy("shared", N, 64, 0.30, trials=100, seed0=0, rho=0.0)
    y = accuracy("shared", N, 64, 0.30, trials=100, seed0=0, rho=0.9)
    assert y < x - 0.2, (x, y)


def test_p90_unit():
    assert p90([(4, 0.99), (8, 0.95), (16, 0.88)]) == 8
    assert p90([(4, 0.5)]) is None
    assert p90([(4, 0.9), (8, 0.9)]) == 8  # boundary counts


def test_p90_monotone_in_rho():
    # P90 falls as correlation rises (sigma=0.30, coarse sweep)
    curve = {}
    for rho in (0.0, 0.3, 0.6, 0.9):
        c = [(p, accuracy("shared", N, p, 0.30, trials=100,
                          seed0=0, rho=rho))
             for p in (4, 16, 64, 256)]
        curve[rho] = p90(c)
    vals = [curve[r] for r in (0.0, 0.3, 0.6, 0.9)]
    assert all((b is None) or (a is not None and a >= b)
               for a, b in zip(vals, vals[1:])), vals
    assert vals[0] is not None and vals[0] > (vals[-1] or 0)


if __name__ == "__main__":
    for name, fn in sorted(
            [(k, v) for k, v in globals().items() if k.startswith("test_")]):
        fn()
        print(f"PASS {name}")
    print("all tests passed")
