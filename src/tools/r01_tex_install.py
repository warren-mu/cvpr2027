"""Pinned scratch-only paper tools; lock first, commit lock, then install.

No conda operations, shell profile edits, HOME reassignment, path-add, deletion,
or floating tlmgr update. Use run_logged for both commands. The login option is
explicitly authorized for R01's bounded download/unpack and CPU-only TeX tools.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shlex
import shutil
import subprocess
import tarfile
import time
import urllib.request

from src.tools.r01_prepare import ROOT, digest

VERSION = "2026.11"
BASE_URL = "https://github.com/rstudio/tinytex-releases/releases/download/v2026.11/TinyTeX-1-linux-x86_64-v2026.11.tar.xz"
BASE_SHA256 = "5b75befa848deae702ae6fa12ff0eef3f110d7e4e2a979ae99f101eba3c9c70d"
BASE_SHA_SOURCE = "https://github.com/rstudio/tinytex-releases/releases/expanded_assets/v2026.11"
REPOSITORY = "https://mirror.ctan.org/systems/texlive/tlnet"
EXTRA_PACKAGES = ("caption", "cleveref", "enumitem", "lineno", "multirow", "silence")
REQUIRED_FILES = ("article.cls", "times.sty", "xspace.sty", "xcolor.sty", "graphicx.sty",
                  "amsmath.sty", "amssymb.sty", "booktabs.sty", "natbib.sty", "silence.sty",
                  "etoolbox.sty", "caption.sty", "subcaption.sty", "url.sty", "lineno.sty",
                  "cleveref.sty", "enumitem.sty", "multirow.sty", "hyperref.sty", "phvb.tfm", "ptmb.tfm")


def file_hash(path, algorithm="sha256"):
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def parse_tlpdb(text):
    records = {}
    for block in re.split(r"\n\s*\n", text):
        fields = {}
        for line in block.splitlines():
            if line and not line[0].isspace():
                key, _, value = line.partition(" ")
                fields.setdefault(key, []).append(value)
        if "name" in fields:
            name = fields["name"][0]
            if name in records:
                raise ValueError(f"Duplicate TLPDB package: {name}")
            records[name] = fields
    return records


def download(url, path, expected=None, algorithm="sha256", timeout=45):
    if path.exists():
        if expected and file_hash(path, algorithm) == expected:
            return {"url": url, "file": str(path), "sha256": file_hash(path), "reused_verified": True}
        raise ValueError(f"Existing download differs or is not pinned: {path}")
    if not url.startswith("https://"):
        raise ValueError("Only official HTTPS release/repository URLs are used")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    if partial.exists():
        raise ValueError(f"Preserving incomplete download; use a fresh run name: {partial}")
    started = time.monotonic()
    request = urllib.request.Request(url, headers={"User-Agent": "cvpr-agent-kit-r01-tex/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response, partial.open("xb") as stream:
        if not response.url.startswith("https://") or response.status != 200:
            raise ValueError("Unexpected source response")
        final_url = response.url
        total = 0
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            total += len(block)
            if total > 300 * 1024 * 1024 or time.monotonic() - started > 600:
                raise ValueError("Bounded archive download exceeded size/time limit")
            stream.write(block)
    if expected and file_hash(partial, algorithm) != expected:
        raise ValueError(f"Archive {algorithm} mismatch; partial retained: {partial}")
    partial.rename(path)
    return {"url": url, "final_url": final_url, "file": str(path), "sha256": file_hash(path), "bytes": path.stat().st_size}


def safe_members(archive):
    members = archive.getmembers()
    links = {PurePosixPath(m.name) for m in members if m.issym() or m.islnk()}
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("Unsafe archive member path")
        if any(parent in links for parent in path.parents):
            raise ValueError("Archive member traverses a link")
        if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
            raise ValueError("Unsupported archive special file")
        if member.issym() or member.islnk():
            target = PurePosixPath(member.linkname)
            if target.is_absolute():
                raise ValueError("Absolute archive link")
            joined = path.parent / target if member.issym() else target
            depth = 0
            for part in joined.parts:
                if part == "..":
                    depth -= 1
                elif part != ".":
                    depth += 1
                if depth < 0:
                    raise ValueError("Escaping archive link")
    return members


def read_base_database(path):
    with tarfile.open(path) as archive:
        candidates = [m for m in safe_members(archive) if m.name.endswith("/tlpkg/texlive.tlpdb") and m.isfile()]
        if len(candidates) != 1:
            raise ValueError("Expected one TinyTeX package database")
        text = archive.extractfile(candidates[0]).read().decode("utf-8")
        top = PurePosixPath(candidates[0].name).parts[0]
        if top not in {".TinyTeX", "TinyTeX"}:
            raise ValueError("Unexpected TinyTeX archive root")
        return top, text, parse_tlpdb(text)


def check_package(path, name, revision):
    with tarfile.open(path) as archive:
        candidates = [m for m in safe_members(archive) if m.name.endswith("/" + name + ".tlpobj") or m.name == name + ".tlpobj"]
        if len(candidates) != 1 or not candidates[0].isfile():
            raise ValueError("Missing unique package manifest")
        info = parse_tlpdb(archive.extractfile(candidates[0]).read().decode("utf-8"))
        if info.get(name, {}).get("revision") != [str(revision)]:
            raise ValueError("Package revision differs from pinned repository metadata")


def lock_tools(args, scratch, evidence, metrics):
    lock_path = args.lock.resolve()
    if lock_path.exists():
        raise ValueError("Refusing to replace existing lock; use a new lock path")
    cache = scratch / "installers/r01_tex" / args.name
    cache.mkdir(parents=True, exist_ok=False)
    base = cache / "TinyTeX-1-linux-x86_64-v2026.11.tar.xz"
    records = metrics.setdefault("downloads", [])
    records.append(download(BASE_URL, base, BASE_SHA256))
    top, base_text, base_records = read_base_database(base)
    (evidence / "base_texlive.tlpdb").write_text(base_text)
    database = cache / "texlive.tlpdb.xz"
    records.append(download(REPOSITORY + "/tlpkg/texlive.tlpdb.xz", database))
    db_text = lzma.decompress(database.read_bytes()).decode("utf-8")
    (evidence / "repository_texlive.tlpdb").write_text(db_text)
    remote = parse_tlpdb(db_text)
    packages = []
    known = set(base_records) | set(EXTRA_PACKAGES)
    for name in EXTRA_PACKAGES:
        row = remote[name]
        dependencies = row.get("depend", [])
        unresolved = [d for d in dependencies if not d.startswith("setting_") and d not in known]
        if unresolved:
            raise ValueError(f"Unpinned dependency for {name}: {unresolved}")
        checksum = row.get("containerchecksum", [""])[0]
        if not re.fullmatch(r"[a-f0-9]{128}", checksum):
            raise ValueError(f"Missing TeXLive SHA512 container checksum: {name}")
        revision = row["revision"][0]
        path = cache / (name + ".tar.xz")
        url = REPOSITORY + "/archive/" + name + ".tar.xz"
        records.append(download(url, path, checksum, "sha512"))
        check_package(path, name, revision)
        packages.append({"name": name, "revision": revision, "url": url,
                         "sha256": file_hash(path), "sha512": checksum, "dependencies": dependencies})
    locked = {"schema_version": 1, "tinytex_version": VERSION, "platform": "x86_64-linux",
              "base": {"url": BASE_URL, "sha256": BASE_SHA256, "checksum_source": BASE_SHA_SOURCE, "archive_root": top},
              "base_package_revisions": {n: v.get("revision", [None])[0] for n, v in base_records.items()},
              "repository_database_sha256": digest(db_text.encode()), "repository_url": REPOSITORY,
              "packages": packages, "required_files": list(REQUIRED_FILES)}
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("x") as stream:
        json.dump(locked, stream, indent=2)
    metrics.update(status="locked_not_installed", lock=str(lock_path), lock_sha256=file_hash(lock_path), cache=str(cache),
                   next_step="Commit lock, then install with the same --lock and --archive-cache path")


def tex_environment(tree, scratch):
    env = dict(os.environ)
    runtime = scratch / "paper_tex_runtime"
    env["PATH"] = str(tree / "bin/x86_64-linux") + os.pathsep + env.get("PATH", "")
    for key, sub in (("TEXMFHOME", "home"), ("TEXMFVAR", "var"), ("TEXMFCONFIG", "config"),
                     ("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "xdg-cache"), ("XDG_CONFIG_HOME", "xdg-config")):
        path = runtime / sub
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env["TEXMFSYSVAR"] = str(tree / "texmf-var")
    env["TEXMFSYSCONFIG"] = str(tree / "texmf-config")
    return env


def install_tools(args, scratch, evidence, metrics):
    lock_path = args.lock.resolve()
    locked = json.loads(lock_path.read_text())
    if locked.get("schema_version") != 1 or locked.get("base", {}).get("sha256") != BASE_SHA256 or locked["base"].get("url") != BASE_URL:
        raise ValueError("Unexpected base archive lock")
    if {p["name"] for p in locked["packages"]} != set(EXTRA_PACKAGES):
        raise ValueError("Unexpected extra package set")
    cache = args.archive_cache.resolve() if args.archive_cache else scratch / "installers/r01_tex" / args.name
    cache.relative_to(scratch)
    cache.mkdir(parents=True, exist_ok=True)
    downloads = metrics.setdefault("downloads", [])
    base = cache / "TinyTeX-1-linux-x86_64-v2026.11.tar.xz"
    downloads.append(download(BASE_URL, base, BASE_SHA256))
    actual_top, _, actual_base = read_base_database(base)
    if locked["base"].get("archive_root") != actual_top or locked.get("base_package_revisions") != {n: v.get("revision", [None])[0] for n, v in actual_base.items()}:
        raise ValueError("Lock does not match the pinned base archive inventory")
    prefix = args.prefix.resolve() if args.prefix else scratch / "tex" / ("tinytex-" + VERSION + "-" + file_hash(lock_path)[:12])
    prefix.relative_to(scratch)
    if prefix.exists():
        raise ValueError("Preserving existing prefix; source its env.sh or choose a fresh --prefix")
    package_files = []
    for row in locked["packages"]:
        expected_url = REPOSITORY + "/archive/" + row["name"] + ".tar.xz"
        if row["url"] != expected_url:
            raise ValueError("Nonofficial package URL in lock")
        path = cache / (row["name"] + ".tar.xz")
        downloads.append(download(row["url"], path, row["sha256"]))
        if file_hash(path, "sha512") != row["sha512"]:
            raise ValueError("Package SHA512 mismatch")
        check_package(path, row["name"], row["revision"])
        package_files.append(path)
    prefix.mkdir(parents=True, exist_ok=False)
    with tarfile.open(base) as archive:
        members = safe_members(archive)
        archive.extractall(prefix, members=members)
    tree = prefix / locked["base"]["archive_root"]
    env = tex_environment(tree, scratch)
    commands = metrics.setdefault("commands", [])

    def execute(command, label):
        output = evidence / (label + ".txt")
        with output.open("w") as stream:
            result = subprocess.run(command, cwd=tree, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=180)
        text = output.read_text(errors="replace")
        print(text, flush=True)
        commands.append({"command": command, "exit_code": result.returncode, "output": str(output.relative_to(ROOT)), "sha256": file_hash(output)})
        if result.returncode:
            raise RuntimeError(f"TeX setup command failed: {label}")
        return text

    # --file prevents repository fetch; dependencies were checked against base + locked extras.
    execute([str(tree / "bin/x86_64-linux/tlmgr"), "install", "--file", "--no-depends", *map(str, package_files)], "install_locked_packages")
    execute([str(tree / "bin/x86_64-linux/mktexlsr")], "refresh_filename_database")
    package_db = tree / "tlpkg/texlive.tlpdb"
    installed = parse_tlpdb(package_db.read_text())
    for row in locked["packages"]:
        if installed.get(row["name"], {}).get("revision") != [row["revision"]]:
            raise ValueError(f"Installed package revision mismatch: {row['name']}")
    for binary in ("pdflatex", "bibtex", "latexmk", "kpsewhich"):
        execute([str(tree / "bin/x86_64-linux" / binary), "--version" if binary != "latexmk" else "-v"], "version_" + binary)
    resolved = {}
    for name in REQUIRED_FILES:
        result = subprocess.run([str(tree / "bin/x86_64-linux/kpsewhich"), name], cwd=tree, env=env, capture_output=True, text=True, timeout=20)
        path = Path(result.stdout.strip())
        if result.returncode or not path.is_file():
            raise ValueError(f"Required TeX input unavailable: {name}")
        path.resolve().relative_to(prefix.resolve())
        resolved[name] = {"file": str(path), "sha256": file_hash(path)}
    activation = prefix / "env.sh"
    exports = {key: env[key] for key in ("TEXMFHOME", "TEXMFVAR", "TEXMFCONFIG", "TEXMFSYSVAR", "TEXMFSYSCONFIG", "TMPDIR", "XDG_CACHE_HOME", "XDG_CONFIG_HOME")}
    activation.write_text("# Generated paper-only environment; source in the compiling shell.\n" + "\n".join(
        f"export {key}={shlex.quote(value)}" for key, value in exports.items()) + "\nexport PATH=" + shlex.quote(str(tree / "bin/x86_64-linux")) + ':"$PATH"\n')
    shutil.copy2(lock_path, prefix / "installation.lock.json")
    shutil.copy2(package_db, evidence / "installed_texlive.tlpdb")
    metrics.update(status="installed", prefix=str(prefix), activation_file=str(activation),
                   caller_instruction="source " + shlex.quote(str(activation)), lock_sha256=file_hash(lock_path),
                   installed_database_sha256=file_hash(package_db), required_inputs=resolved,
                   home_unchanged=True, training_environment_untouched=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("lock", "install"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--execution-location", required=True, choices=("login", "pbs"))
    ap.add_argument("--lock", type=Path, default=ROOT / "configs/r01_tex_lock.json")
    ap.add_argument("--archive-cache", type=Path)
    ap.add_argument("--prefix", type=Path)
    args = ap.parse_args()
    if not re.fullmatch(r"r01_[A-Za-z0-9_]+", args.name):
        ap.error("Use a unique r01_ name")
    if args.execution_location == "pbs" and not os.environ.get("PBS_JOBID"):
        ap.error("PBS location requires an allocation")
    evidence = ROOT / "logs/runs" / args.name
    target = ROOT / "results" / (args.name + ".json")
    if evidence.exists() or target.exists():
        ap.error("Existing run artifacts; use a new name")
    evidence.mkdir(parents=True, exist_ok=False)
    metrics = {"status": "failed"}
    config = {"mode": args.mode, "execution_location": args.execution_location, "lock": str(args.lock),
              "tinytex_version": VERSION, "extra_packages": list(EXTRA_PACKAGES), "base_url": BASE_URL,
              "base_sha256": BASE_SHA256, "base_checksum_source": BASE_SHA_SOURCE,
              "scope": "R01 CPU-only downloads, unpacking, and TeX tools; no GPU or training environment installation"}
    code = 1
    try:
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise ValueError("Pinned bundle requires Linux x86_64")
        scratch = Path(os.environ["CVPR_SCRATCH"])
        if not scratch.is_absolute():
            raise ValueError("CVPR_SCRATCH must be explicit and absolute")
        scratch = scratch.resolve()
        home = Path.home().resolve()
        if scratch == home or home in scratch.parents or scratch == Path("/"):
            raise ValueError("Paper installation scratch must be outside HOME and not the filesystem root")
        config["scratch"] = str(scratch)
        {"lock": lock_tools, "install": install_tools}[args.mode](args, scratch, evidence, metrics)
        code = 0
    except Exception as exc:
        metrics.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"metrics": metrics, "config": config}, indent=2))
        print("FINAL_METRICS_JSON: " + json.dumps(metrics, sort_keys=True), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
