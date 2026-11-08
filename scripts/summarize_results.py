#!/usr/bin/env python3
"""Aggregate multi-seed results into mean / std / 95% CI and (optionally) compare two methods.

Naming convention for runs:  <group>_s<seed>   e.g. main_nav_ours_s0, main_nav_ours_s1 ...
Run it THROUGH run_logged so the aggregate itself has provenance:

    python3 scripts/run_logged.py --name agg_main --results results/agg_main.json -- \
        python3 scripts/summarize_results.py --prefix main_ --out results/agg_main.json

    python3 scripts/summarize_results.py --prefix main_nav_ --compare main_nav_ours main_nav_vagen --metric success

Output metric keys:  <group>__<metric>__mean / __std / __n / __ci95   (usable as \\res{...})
The comparison prints the difference of means with a Welch 95% CI; p-value only if scipy exists.
Never drop a seed by hand: every <group>_s* file found is used.
"""
import argparse
import hashlib
import glob
import json
import math
import os
import re
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
       9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042}


def t95(df):
    if df <= 0:
        return float("nan")
    if df in T95:
        return T95[df]
    keys = sorted(T95)
    for k in keys:
        if k > df:
            return T95[k]
    return 1.96


def load_groups(prefix):
    groups = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "results", f"{prefix}*.json"))):
        name = os.path.splitext(os.path.basename(path))[0]
        m = re.match(r"^(.*)_s(\d+)$", name)
        if not m:
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        meta = data.get("_meta", {})
        groups.setdefault(m.group(1), []).append(
            {"seed": int(m.group(2)), "metrics": data.get("metrics", {}), "file": os.path.relpath(path, ROOT),
             "logged": meta.get("producer") == "scripts/run_logged.py", "exit_code": meta.get("exit_code"),
             "metrics_sha256": hashlib.sha256(json.dumps(data.get("metrics", {}), sort_keys=True).encode()).hexdigest()})
    return groups


def stats(values):
    n = len(values)
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    ci = t95(n - 1) * std / math.sqrt(n) if n > 1 else float("nan")
    return mean, std, n, ci


def welch(a, b):
    ma, sa, na, _ = stats(a)
    mb, sb, nb, _ = stats(b)
    if na < 2 or nb < 2:
        return ma - mb, float("nan"), None
    va, vb = sa ** 2 / na, sb ** 2 / nb
    se = math.sqrt(va + vb) or 1e-12
    df = (va + vb) ** 2 / ((va ** 2 / (na - 1) if na > 1 else 0) + (vb ** 2 / (nb - 1) if nb > 1 else 0) or 1e-12)
    ci = t95(max(1, int(df))) * se
    p = None
    try:
        from scipy import stats as st  # optional
        p = float(2 * st.t.sf(abs((ma - mb) / se), df))
    except ImportError:
        pass
    return ma - mb, ci, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out", default=None, help="write aggregate JSON (use via run_logged)")
    ap.add_argument("--compare", nargs=2, metavar=("GROUP_A", "GROUP_B"))
    ap.add_argument("--metric", default=None)
    args = ap.parse_args()

    groups = load_groups(args.prefix)
    if not groups:
        sys.exit(f"no results/{args.prefix}*_s<seed>.json files")
    agg, warnings = {}, []
    print(f"{'group':<36} {'metric':<26} {'mean':>9} {'std':>8} {'n':>3} {'ci95':>8}")
    for g, runs in sorted(groups.items()):
        for r in runs:
            if not r["logged"]:
                warnings.append(f"{r['file']}: not produced by run_logged")
            if r["exit_code"] not in (0, None):
                warnings.append(f"{r['file']}: exit_code={r['exit_code']}")
        keys = sorted(set().union(*[r["metrics"].keys() for r in runs]))
        for k in keys:
            vals = [r["metrics"][k] for r in runs if isinstance(r["metrics"].get(k), (int, float))]
            if len(vals) != len(runs):
                warnings.append(f"{g}: metric {k} missing in some seeds ({len(vals)}/{len(runs)})")
            if not vals:
                continue
            mean, std, n, ci = stats(vals)
            base = f"{g}__{k}"
            agg.update({f"{base}__mean": mean, f"{base}__std": std, f"{base}__n": n})
            if not math.isnan(ci):
                agg[f"{base}__ci95"] = ci
            print(f"{g:<36} {k:<26} {mean:>9.4f} {std:>8.4f} {n:>3} {ci:>8.4f}")
        if len(runs) < 3:
            warnings.append(f"{g}: only {len(runs)} seed(s) — say so in the paper")

    if args.compare:
        a, b = args.compare
        if a not in groups or b not in groups:
            sys.exit(f"compare: unknown group(s); have {sorted(groups)}")
        metrics = [args.metric] if args.metric else sorted(groups[a][0]["metrics"])
        print(f"\nWelch comparison {a} - {b}:")
        for k in metrics:
            va = [r["metrics"][k] for r in groups[a] if isinstance(r["metrics"].get(k), (int, float))]
            vb = [r["metrics"][k] for r in groups[b] if isinstance(r["metrics"].get(k), (int, float))]
            if not va or not vb:
                continue
            d, ci, p = welch(va, vb)
            verdict = "CI excludes 0" if not math.isnan(ci) and abs(d) > ci else "not distinguishable at 95%"
            ptxt = f" p={p:.4f}" if p is not None else ""
            print(f"  {k:<26} diff={d:+.4f} ±{ci:.4f}{ptxt}  -> {verdict}")
            agg[f"cmp__{a}__vs__{b}__{k}__diff"] = d
            if not math.isnan(ci):
                agg[f"cmp__{a}__vs__{b}__{k}__ci95"] = ci

    for w in warnings:
        print("WARN", w, file=sys.stderr)
    if args.out:
        out = os.path.join(ROOT, args.out) if not os.path.isabs(args.out) else args.out
        with open(out, "w", encoding="utf-8") as f:
            sources = [{"file": r["file"], "metrics_sha256": r["metrics_sha256"]}
                       for g in sorted(groups) for r in sorted(groups[g], key=lambda r: r["seed"])]
            json.dump({"metrics": agg, "_format": {k: ".0f" for k in agg if k.endswith("__n")},
                       "sources": sources,
                       "notes": f"aggregate of results/{args.prefix}*; warnings: {warnings}"}, f, indent=2)
        print(f"\nwrote {args.out} ({len(agg)} metrics)")


if __name__ == "__main__":
    main()
