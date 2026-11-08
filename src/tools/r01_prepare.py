"""R01 source retrieval, bibliography generation, unchanged verifier, and LaTeX.

Run ONLY inside scripts/run_logged.py after committing this producer. All network
responses and failures are retained in a new, non-overwritable evidence directory.
This does not edit paper/main.tex, sections, or any protected script.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
from html.parser import HTMLParser
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "configs/r01_sources.json"
ALLOWED_HOSTS = {"arxiv.org", "export.arxiv.org", "proceedings.neurips.cc",
                 "proceedings.mlr.press", "openaccess.thecvf.com", "api.crossref.org",
                 "dblp.org", "cvpr.thecvf.com", "api.github.com", "raw.githubusercontent.com"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def protected_parser():
    spec = importlib.util.spec_from_file_location("r01_protected_verify_bib", ROOT / "scripts/verify_bib.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalized_title(value):
    value = re.sub(r"\\[a-zA-Z]+\s*", "", html.unescape(value))
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", value.lower())


def validate_entry(entry, row):
    fields = entry["fields"]
    allowed = {normalized_title(t) for t in [row["title"], *row.get("title_aliases", [])]}
    if normalized_title(fields.get("title", "")) not in allowed:
        raise ValueError(f"Source title mismatch: {fields.get('title')!r}")
    if not fields.get("author") or not re.fullmatch(r"\d{4}", fields.get("year", "")):
        raise ValueError("Source lacks authors or a four-digit publication year")
    entry["key"] = row["key"]
    return entry


class HTMLMetadata(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.meta = {}
        self.text = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "meta" and attrs.get("name") and "content" in attrs:
            self.meta.setdefault(attrs["name"].lower(), []).append(attrs["content"])

    def handle_data(self, data):
        self.text.append(data)


def latex_text(text):
    # Preserve source math/TeX; escape only bare text special characters.
    return re.sub(r"(?<!\\)([&%#_])", r"\\\1", text)


def parse_source(kind, body, row):
    text = body.decode("utf-8")
    if kind in {"bibtex", "html_bibtex"}:
        if kind == "html_bibtex":
            page = HTMLMetadata()
            page.feed(text)
            text = "".join(page.text)
        entries = protected_parser().parse_bib(html.unescape(text))
        matches = []
        for entry in entries:
            try:
                matches.append(validate_entry(entry, row))
            except ValueError:
                continue
        if len(matches) != 1:
            raise ValueError(f"Expected one matching source BibTeX entry, found {len(matches)}")
        return matches[0]
    if kind == "crossref":
        data = json.loads(text)["message"]
        fields = {"title": latex_text(data["title"][0]),
                  "author": " and ".join(latex_text(a.get("given", "") + " " + a["family"]).strip()
                                          for a in data["author"]),
                  "year": str(data["issued"]["date-parts"][0][0]),
                  "doi": data["DOI"], "url": data["URL"]}
        if data.get("container-title"):
            fields["booktitle" if data["type"] in {"proceedings-article", "book-chapter"} else "journal"] = latex_text(data["container-title"][0])
        for source, target in (("page", "pages"), ("volume", "volume"), ("publisher", "publisher")):
            if data.get(source):
                fields[target] = latex_text(str(data[source]))
        typ = "inproceedings" if data["type"] == "proceedings-article" else "incollection" if data["type"] == "book-chapter" else "article"
        return validate_entry({"type": typ, "fields": fields}, row)
    if kind == "arxiv_atom":
        ns = {"a": "http://www.w3.org/2005/Atom"}
        entries = ET.fromstring(text).findall("a:entry", ns)
        if len(entries) != 1:
            raise ValueError("arXiv API did not return exactly one entry")
        item = entries[0]
        identifier = item.findtext("a:id", "", ns)
        if not re.search(r"/" + re.escape(row["arxiv"]) + r"(?:v\d+)?$", identifier):
            raise ValueError("arXiv returned an error or a different identifier")
        fields = {"title": latex_text(" ".join(item.findtext("a:title", "", ns).split())),
                  "author": " and ".join(latex_text(a.findtext("a:name", "", ns)) for a in item.findall("a:author", ns)),
                  "year": item.findtext("a:published", "", ns)[:4]}
    elif kind == "arxiv_html":
        page = HTMLMetadata()
        page.feed(text)
        meta = page.meta
        actual = meta.get("citation_arxiv_id", [""])[0]
        if actual.split("v")[0] != row["arxiv"]:
            raise ValueError("arXiv HTML identifier mismatch")
        fields = {"title": latex_text(meta.get("citation_title", [""])[0]),
                  "author": " and ".join(latex_text(a) for a in meta.get("citation_author", [])),
                  "year": meta.get("citation_date", [""])[0][:4]}
    else:
        raise ValueError(f"Unknown source kind {kind}")
    fields.update({"eprint": row["arxiv"], "archiveprefix": "arXiv", "url": "https://arxiv.org/abs/" + row["arxiv"]})
    return validate_entry({"type": "misc", "fields": fields}, row)


def bibtex(entry):
    fields = {k: v for k, v in entry["fields"].items() if k not in {"abstract", "timestamp", "biburl", "bibsource"}}
    # Outer protection preserves acronyms without changing any source title.
    fields["title"] = "{" + fields["title"] + "}"
    return "@" + entry["type"] + "{" + entry["key"] + ",\n" + "\n".join(
        f"  {k} = {{{v}}}," for k, v in fields.items()) + "\n}\n"


class Evidence:
    def __init__(self, directory, timeout):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        self.timeout = timeout
        self.requests = []
        self.last_arxiv = 0.0

    def fetch(self, url, label):
        if urllib.parse.urlparse(url).hostname not in ALLOWED_HOSTS or not url.startswith("https://"):
            raise ValueError("Unapproved official source URL")
        if "arxiv.org" in urllib.parse.urlparse(url).hostname:
            time.sleep(max(0, 3.1 - (time.monotonic() - self.last_arxiv)))
            self.last_arxiv = time.monotonic()
        record = {"url": url, "label": label, "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
        self.requests.append(record)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "cvpr-agent-kit-r01/1.0 (bibliography verification)"})
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                if urllib.parse.urlparse(response.url).hostname not in ALLOWED_HOSTS:
                    raise ValueError("Redirect left official source allowlist")
                body = response.read(16 * 1024 * 1024 + 1)
                if len(body) > 16 * 1024 * 1024:
                    raise ValueError("Source response exceeded bounded metadata size")
                record.update({"http_status": response.status, "final_url": response.url,
                               "content_type": response.headers.get("Content-Type"), "sha256": digest(body)})
                path = self.directory / (label + ".response")
                path.write_bytes(body)
                record["file"] = str(path.relative_to(ROOT))
                return body
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, urllib.error.HTTPError):
                record["http_status"] = exc.code
            raise
        finally:
            (self.directory / "requests.json").write_text(json.dumps(self.requests, indent=2), encoding="utf-8")


def generate(args, evidence, manifest, metrics):
    rows = []
    generated = []
    for row in manifest["entries"]:
        observation = {"key": row["key"], "attempts": [], "status": "unavailable"}
        rows.append(observation)
        candidates = []
        if row.get("preferred"):
            candidates.append(row["preferred"])
        if row.get("arxiv"):
            candidates.extend([
                {"kind": "arxiv_atom", "url": "https://export.arxiv.org/api/query?id_list=" + row["arxiv"]},
                {"kind": "arxiv_html", "url": "https://arxiv.org/abs/" + row["arxiv"]}])
        for index, source in enumerate(candidates):
            attempt = dict(source)
            observation["attempts"].append(attempt)
            try:
                body = evidence.fetch(source["url"], f"{row['key']}_{index}")
                entry = parse_source(source["kind"], body, row)
                attempt["status"] = "parsed_title_matched"
                observation.update({"status": "generated", "selected": source, "source_sha256": digest(body),
                                    "preferred_source_used": source == row.get("preferred"), "entry": entry})
                generated.append(entry)
                break
            except Exception as exc:
                attempt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        print(json.dumps({"key": row["key"], "status": observation["status"]}), flush=True)
    target = ROOT / "paper/refs.bib"
    if target.exists():
        shutil.copy2(target, evidence.directory / "refs.before.bib")
    text = "% Generated by src/tools/r01_prepare.py from retained official metadata.\n\n" + "\n".join(bibtex(e) for e in generated)
    target.write_text(text, encoding="utf-8")
    shutil.copy2(target, evidence.directory / "refs.generated.bib")
    missing = [r["key"] for r in rows if r["status"] != "generated"]
    degraded = [r["key"] for r in rows if r["status"] == "generated" and next(m for m in manifest["entries"] if m["key"] == r["key"]).get("preferred") and not r["preferred_source_used"]]
    metrics.update(status="complete" if not missing and not degraded else "incomplete", entries=rows,
                   required_keys=[r["key"] for r in manifest["entries"]], missing_keys=missing,
                   conference_source_unavailable_keys=degraded, generated_entry_count=len(generated),
                   refs_bib_sha256=digest(target.read_bytes()), verification_not_yet_run=True)
    return int(bool(missing or degraded))


def verify(args, evidence, manifest, metrics):
    parser = protected_parser()
    script = ROOT / "scripts/verify_bib.py"
    target = ROOT / "paper/refs.bib"
    cache = ROOT / "logs/bib_verification.json"
    if cache.exists():
        shutil.copy2(cache, evidence.directory / "verification.before.json")
    entries = parser.parse_bib(target.read_text(encoding="utf-8"))
    required = [r["key"] for r in manifest["entries"]]
    keys = [r["key"] for r in entries]
    command = [sys.executable, "scripts/verify_bib.py", "paper/refs.bib"]
    if args.recheck:
        command.append("--recheck")
    metrics.update(protected_command=command, protected_script_sha256=digest(script.read_bytes()),
                   refs_bib_sha256=digest(target.read_bytes()), missing_keys=sorted(set(required) - set(keys)),
                   duplicate_keys=sorted(k for k in set(keys) if keys.count(k) > 1), required_keys=required)
    try:
        completed = subprocess.run(command, cwd=ROOT, timeout=args.process_timeout)
        metrics["protected_exit_code"] = completed.returncode
    except subprocess.TimeoutExpired:
        metrics["protected_exit_code"] = None
        metrics["timeout"] = True
    data = json.loads(cache.read_text()) if cache.exists() else {}
    if cache.exists():
        shutil.copy2(cache, evidence.directory / "verification.after.json")
        metrics["cache_sha256"] = digest(cache.read_bytes())
    records = {}
    for entry in entries:
        record = data.get(entry["key"], {})
        records[entry["key"]] = {"record": record, "entry_hash_matches": record.get("entry_hash") == entry["hash"]}
    metrics["entries"] = records
    metrics["unverified_keys"] = [k for k in required if records.get(k, {}).get("record", {}).get("status") != "verified" or not records.get(k, {}).get("entry_hash_matches")]
    good = metrics.get("protected_exit_code") == 0 and not metrics["missing_keys"] and not metrics["duplicate_keys"] and not metrics["unverified_keys"]
    metrics["status"] = "all_required_verified" if good else "incomplete"
    metrics["interpretation"] = "The protected script's exit 0 alone is not all-verified: partial/error entries are explicitly retained here."
    return 0 if good else 1


def template(args, evidence, manifest, metrics):
    settings = manifest["author_kit"]
    evidence.fetch(settings["guidelines_url"], "author_guidelines")
    commit = json.loads(evidence.fetch(settings["commit_api"], "kit_commit"))
    sha = commit["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("GitHub did not supply an immutable commit SHA")
    base = "https://raw.githubusercontent.com/cvpr-org/author-kit/" + sha + "/"
    readme = evidence.fetch(base + "README.md", "kit_readme")
    if settings["required_readme_text"] not in readme.decode("utf-8"):
        raise ValueError("Pinned kit does not explicitly identify CVPR 2027")
    archive = ROOT / "paper/author-kit-2027" / sha
    archive.mkdir(parents=True, exist_ok=False)
    files = []
    for name in dict.fromkeys(settings["install_files"] + settings["archive_files"]):
        body = readme if name == "README.md" else evidence.fetch(base + name, "kit_" + name.replace(".", "_"))
        (archive / name).write_bytes(body)
        row = {"name": name, "sha256": digest(body), "source_url": base + name, "archive": str((archive / name).relative_to(ROOT))}
        if name in settings["install_files"]:
            target = ROOT / "paper" / name
            if target.exists() and target.read_bytes() != body:
                raise ValueError(f"Refusing to replace a different existing template file: {target}")
            target.write_bytes(body)
            row["installed"] = str(target.relative_to(ROOT))
        files.append(row)
    metrics.update(status="installed", commit=sha, files=files,
                   provenance_note="Official cvpr-org repository README explicitly names 2027; the fetched author-guidelines page is retained, without claiming it contains a kit link if absent.",
                   main_and_sections_untouched=True)
    return 0


def compile_paper(args, evidence, manifest, metrics):
    command = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-outdir=" + str(evidence.directory), "main.tex"]
    metrics.update(command=command, main_sha256=digest((ROOT / "paper/main.tex").read_bytes()),
                   source_files={str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in sorted((ROOT / "paper").rglob("*.tex")) if "author-kit-2027" not in p.parts})
    output = evidence.directory / "latexmk.stdout.txt"
    with output.open("w", encoding="utf-8") as stream:
        try:
            proc = subprocess.run(command, cwd=ROOT / "paper", stdout=stream, stderr=subprocess.STDOUT, timeout=args.process_timeout)
            metrics["latexmk_exit_code"] = proc.returncode
        except subprocess.TimeoutExpired:
            metrics.update(latexmk_exit_code=None, timeout=True)
    text = output.read_text(errors="replace")
    print(text, flush=True)
    log = evidence.directory / "main.log"
    logtext = log.read_text(errors="replace") if log.exists() else text
    metrics["warnings"] = [line for line in logtext.splitlines() if "Warning" in line or "Overfull" in line or "Underfull" in line]
    metrics["errors"] = [line for line in logtext.splitlines() if line.startswith("!") or "LaTeX Error" in line]
    pdf = evidence.directory / "main.pdf"
    metrics["pdf"] = str(pdf.relative_to(ROOT)) if pdf.exists() else None
    metrics["pdf_sha256"] = digest(pdf.read_bytes()) if pdf.exists() else None
    matches = re.findall(r"Output written on .*?\((\d+) pages?", logtext)
    metrics["pages"] = int(matches[-1]) if matches and pdf.exists() else None
    metrics["status"] = "compiled" if metrics.get("latexmk_exit_code") == 0 and pdf.exists() else "failed"
    return int(metrics["status"] != "compiled")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["generate", "verify", "template", "compile"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--http-timeout", type=float, default=25)
    ap.add_argument("--process-timeout", type=float, default=900)
    ap.add_argument("--recheck", action="store_true")
    args = ap.parse_args()
    if not re.fullmatch(r"r01_[a-zA-Z0-9_]+", args.name):
        ap.error("Use a unique r01_ name")
    result = ROOT / "results" / (args.name + ".json")
    directory = ROOT / "logs/runs" / args.name
    if directory.exists() or result.exists():
        ap.error("Evidence or result already exists; use a new name")
    evidence = Evidence(directory, args.http_timeout)
    metrics = {"status": "failed"}
    manifest = json.loads(MANIFEST.read_text())
    config = {"mode": args.mode, "name": args.name, "manifest": str(MANIFEST.relative_to(ROOT)),
              "manifest_sha256": digest(MANIFEST.read_bytes()), "evidence_directory": str(directory.relative_to(ROOT)),
              "http_timeout_seconds": args.http_timeout, "process_timeout_seconds": args.process_timeout}
    code = 1
    try:
        code = {"generate": generate, "verify": verify, "template": template, "compile": compile_paper}[args.mode](args, evidence, manifest, metrics)
    except Exception as exc:
        metrics.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        metrics["source_requests"] = evidence.requests
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text(json.dumps({"metrics": metrics, "config": config}, indent=2, ensure_ascii=False), encoding="utf-8")
        print("FINAL_METRICS_JSON: " + json.dumps(metrics, sort_keys=True, ensure_ascii=False), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
