"""Figure helper: every figure in the paper is generated from results/ and carries a provenance sidecar.

    from analysis.plotlib import savefig_with_provenance
    fig, ax = plt.subplots()
    ...
    savefig_with_provenance(fig, "figures/main_ood.pdf",
                            sources=["results/agg_main.json"], script=__file__)

Writes figures/main_ood.pdf and figures/main_ood.pdf.meta.json (sources + sha256, script, git commit).
scripts/check_paper.py warns about any \\includegraphics without such a sidecar
(hand-drawn method diagrams: put `% fig-ok: diagram` on the \\includegraphics line).
"""
import datetime as dt
import hashlib
import json
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def savefig_with_provenance(fig, out_path, sources, script, **savefig_kwargs):
    out_abs = out_path if os.path.isabs(out_path) else os.path.join(ROOT, out_path)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    missing = [s for s in sources if not os.path.exists(os.path.join(ROOT, s))]
    if missing:
        raise FileNotFoundError(f"figure sources missing: {missing}")
    savefig_kwargs.setdefault("bbox_inches", "tight")
    fig.savefig(out_abs, **savefig_kwargs)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        commit = None
    meta = {
        "figure": os.path.relpath(out_abs, ROOT),
        "script": os.path.relpath(os.path.abspath(script), ROOT),
        "sources": [{"path": s, "sha256": _sha(os.path.join(ROOT, s))} for s in sources],
        "git_commit": commit,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
    }
    with open(out_abs + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return out_abs
