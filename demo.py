"""Demo: the realistic capacity bound.

1. Correlation calibration: measured mean |<a|b>|^2 per rho vs theory.
2. P90 sweep: rho x sigma grid over P = 4..1024.
3. One structured-family data point: rank-8 subspace, R=8, N=64.

Cells run in parallel (multiprocessing); every cell is seeded, so the
table is deterministic regardless of worker count.

Run: python demo.py
"""

from multiprocessing import Pool

from ccapacity import (accuracy, mean_pair_overlap2, p90,
                       rankr_family, shared_direction_family,
                       theory_shared_overlap2)

N = 64
PS = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
RHOS = [0.0, 0.3, 0.6, 0.9]
SIGMAS = [0.15, 0.30]


def cell(args):
    label, family, sigma, p, kw = args
    trials = 200 if p <= 128 else 100
    return (label, family, sigma, p,
            accuracy(family, N, p, sigma, trials=trials,
                     seed0=7000, **kw))


def main():
    print("== 1. correlation calibration (N=64, 200 patterns) ==")
    print(f"  {'rho':>4} {'measured':>9} {'theory':>8}")
    for rho in RHOS:
        pats = shared_direction_family(N, 200, rho, seed=11)
        m = mean_pair_overlap2(pats)
        print(f"  {rho:>4} {m:>9.4f} {theory_shared_overlap2(N, rho):>8.4f}")
    pats = rankr_family(N, 200, 8, seed=11)
    print(f"  {'R=8':>4} {mean_pair_overlap2(pats):>9.4f} {1/8:>8.4f}")

    print("\n== 2. P90 sweep (rows: rho, cols: sigma) ==")
    jobs = [(f"rho={r}", "shared", s, p, {"rho": r})
            for r in RHOS for s in SIGMAS for p in PS]
    jobs += [("rank8", "rankR", s, p, {"rank": 8})
             for s in SIGMAS for p in PS]
    with Pool() as pool:
        results = pool.map(cell, jobs)
    acc = {(label, s, p): a for label, _, s, p, a in results}

    def row(label):
        cells = []
        for s in SIGMAS:
            curve = [(p, acc[(label, s, p)]) for p in PS]
            v = p90(curve)
            cells.append(">1024" if v == PS[-1]
                         else (str(v) if v else "<4"))
        print(f"  {label:>6} " + " ".join(f"{c:>7}" for c in cells))

    print(f"  {'':>6} " + " ".join(f"{s:>7}" for s in SIGMAS))
    for r in RHOS:
        row(f"rho={r}")
    row("rank8")

    print("\n== 3. full curves (P: accuracy) ==")
    for r in RHOS:
        for s in SIGMAS:
            curve = " ".join(f"{p}:{acc[(f'rho={r}', s, p)]:.2f}"
                             for p in PS)
            print(f"  rho={r} s={s}: {curve}")

    print("\nRESULT: random patterns were the best case. Correlation "
          "collapses capacity -- see DESIGN.md.")


if __name__ == "__main__":
    main()
