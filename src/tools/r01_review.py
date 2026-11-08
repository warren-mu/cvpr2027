"""Log the editorial R01 review, retain official evidence, and check quote/coverage.

Judgments are explicitly authored editorial assessments in the input config;
substring checks validate evidence retention, not semantic truth automatically.
"""
import argparse
import json
from pathlib import Path
import re

from src.tools.r01_prepare import ROOT, Evidence, HTMLMetadata, digest


def normalized(text):
    return " ".join(text.replace("\u2019", "'").replace("\u00a0", " ").split())


def citation_lines(files):
    found = {}
    for filename in files:
        for number, line in enumerate((ROOT / filename).read_text().splitlines(), 1):
            line = re.split(r"(?<!\\)%", line, 1)[0]
            for group in re.findall(r"\\cite\w*\{([^}]+)\}", line):
                for key in group.split(","):
                    found.setdefault(key.strip(), []).append({"file": filename, "line": number, "text": line})
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    if not re.fullmatch(r"r01_[a-zA-Z0-9_]+", args.name):
        ap.error("Use a new r01_ name")
    target = ROOT / "results" / (args.name + ".json")
    directory = ROOT / "logs/runs" / args.name
    if directory.exists() or target.exists():
        ap.error("Existing evidence; use a new name")
    config_path = ROOT / "configs/r01_claim_review.json"
    source = json.loads(config_path.read_text())
    evidence = Evidence(directory, 25)
    metrics = {"status": "failed", "reviews": [], "judgments_are_editorial": True}
    config = {"review_config": str(config_path.relative_to(ROOT)), "review_config_sha256": digest(config_path.read_bytes()),
              "scope_sha256": {f: digest((ROOT / f).read_bytes()) for f in source["scope"]}, "policy": source["policy"]}
    code = 1
    try:
        cited = citation_lines(source["scope"])
        reviews = source["reviews"]
        keys = [r["key"] for r in reviews]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate review keys")
        metrics["unreviewed_cited_keys"] = sorted(set(cited) - set(keys))
        for row in reviews:
            observed = {**row, "citation_lines": cited.get(row["key"], []), "quote_found_in_saved_source": False}
            metrics["reviews"].append(observed)
            try:
                if len(row["quote"].split()) > 25:
                    raise ValueError("Quote exceeds short-excerpt bound")
                body = evidence.fetch(row["source_url"], row["key"])
                page = HTMLMetadata()
                page.feed(body.decode("utf-8"))
                observed["source_sha256"] = digest(body)
                observed["quote_found_in_saved_source"] = normalized(row["quote"]) in normalized(" ".join(page.text))
                if not observed["quote_found_in_saved_source"]:
                    observed["evidence_error"] = "Quoted wording not found after whitespace normalization; requires source-version/manual review"
            except Exception as exc:
                observed["evidence_error"] = f"{type(exc).__name__}: {exc}"
        metrics["evidence_unconfirmed_keys"] = [r["key"] for r in metrics["reviews"] if not r["quote_found_in_saved_source"]]
        metrics["editorial_issues"] = [r["key"] for r in reviews if r["verdict"] != "相符"]
        metrics["status"] = "review_recorded" if not metrics["evidence_unconfirmed_keys"] and not metrics["unreviewed_cited_keys"] else "incomplete_evidence"
        lines = ["# R01 描述核对（脚本生成；判断来自已提交审查配置）", "", source["policy"], "",
                 "| Key | 稿件描述 | 判断 | 原文短引及定位 | 限制/建议 |", "|---|---|---|---|---|"]
        for row in metrics["reviews"]:
            values = [row["key"], row["description"], row["verdict"],
                      f'“{row["quote"]}” ({row["quote_kind"]}; [{row["locator"]}]({row["source_url"]})); quote_found={row["quote_found_in_saved_source"]}', row["qualification"]]
            lines.append("| " + " | ".join(v.replace("|", "\\|").replace("\n", " ") for v in values) + " |")
        lines.extend(["", "## 逐处引用覆盖", ""])
        for key, locations in cited.items():
            for location in locations:
                lines.append(f'- `{key}` — `{location["file"]}:{location["line"]}`: {location["text"]}')
        report = directory / "description_review.md"
        report.write_text("\n".join(lines) + "\n")
        metrics["table"] = str(report.relative_to(ROOT))
        metrics["table_sha256"] = digest(report.read_bytes())
        code = int(metrics["status"] != "review_recorded")
    except Exception as exc:
        metrics.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        metrics["source_requests"] = evidence.requests
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"metrics": metrics, "config": config}, ensure_ascii=False, indent=2))
        print("FINAL_METRICS_JSON: " + json.dumps(metrics, ensure_ascii=False, sort_keys=True), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
