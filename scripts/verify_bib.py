#!/usr/bin/env python3
"""Verify every BibTeX entry against real databases (stdlib only, needs internet).

    python3 scripts/verify_bib.py paper/refs.bib              # all entries
    python3 scripts/verify_bib.py paper/refs.bib --only-cited # only keys \\cite'd in paper/
    python3 scripts/verify_bib.py paper/refs.bib --recheck    # ignore cache

Lookup order: DOI -> Crossref, arXiv id -> arXiv API, title -> DBLP -> Semantic Scholar
-> arXiv -> Crossref. Title similarity >= 0.92 and matching first author / year (+-1)
=> verified; 0.80-0.92 or metadata mismatch => partial; otherwise not_found.
Results are cached in logs/bib_verification.json (read by scripts/check_paper.py).
Set S2_API_KEY for a higher Semantic Scholar rate limit.
Exit code 1 if any checked entry is not_found.
"""
import argparse
import datetime as dt
import difflib
import glob
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "logs", "bib_verification.json")
UA = "cvpr-agent-kit-verify-bib/1.0"
_last_call = {}


# ----------------------------------------------------------------- BibTeX parsing
def _balanced(text, start, open_ch="{", close_ch="}"):
    depth, p = 0, start
    while p < len(text):
        c = text[p]
        if c == "\\":
            p += 2
            continue
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return p
        p += 1
    return len(text) - 1


def _parse_fields(body):
    fields, p, n = {}, 0, len(body)
    while p < n:
        m = re.compile(r"\s*,?\s*([A-Za-z][\w\-]*)\s*=\s*").match(body, p)
        if not m:
            break
        name, p = m.group(1).lower(), m.end()
        if p >= n:
            break
        if body[p] == "{":
            end = _balanced(body, p)
            value, p = body[p + 1:end], end + 1
        elif body[p] == '"':
            q = p + 1
            depth = 0
            while q < n and not (body[q] == '"' and depth == 0):
                depth += body[q] == "{"
                depth -= body[q] == "}"
                q += 1
            value, p = body[p + 1:q], q + 1
        else:
            m2 = re.compile(r"[^,}\s]+").match(body, p)
            value = m2.group(0) if m2 else ""
            p = m2.end() if m2 else p + 1
        fields[name] = value.strip()
    return fields


def parse_bib(text):
    entries = []
    for m in re.finditer(r"@([A-Za-z]+)\s*\{", text):
        typ = m.group(1).lower()
        if typ in ("comment", "preamble", "string"):
            continue
        end = _balanced(text, m.end() - 1)
        inner = text[m.end():end]
        comma = inner.find(",")
        if comma < 0:
            continue
        key = inner[:comma].strip()
        fields = _parse_fields(inner[comma + 1:])
        raw = text[m.start():end + 1]
        entries.append({"key": key, "type": typ, "fields": fields,
                        "hash": hashlib.sha1(raw.encode()).hexdigest()[:12]})
    return entries


# ----------------------------------------------------------------- normalisation
def strip_latex(s):
    s = re.sub(r"\\[a-zA-Z]+\*?\s*", " ", s or "")
    s = s.replace("{", "").replace("}", "").replace("~", " ")
    s = re.sub(r"\\.", "", s)
    return re.sub(r"\s+", " ", s).strip()


def norm_title(s):
    s = unicodedata.normalize("NFKD", strip_latex(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def surname(name):
    name = strip_latex(name).strip()
    if "," in name:
        name = name.split(",")[0]
    else:
        name = name.split()[-1] if name.split() else name
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", name.lower())


def first_author(fields):
    authors = re.split(r"\s+and\s+", fields.get("author", ""), flags=re.I)
    return surname(authors[0]) if authors and authors[0].strip() else ""


def arxiv_id(fields):
    for k in ("eprint", "arxivid", "journal", "url", "note", "howpublished", "volume"):
        v = fields.get(k, "")
        m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", v)
        if m and (k in ("eprint", "arxivid") or "arxiv" in v.lower()):
            return m.group(1)
    return None


# ----------------------------------------------------------------- HTTP
def http_get(url, headers=None, min_gap=1.0, host_key=None):
    host_key = host_key or urllib.parse.urlparse(url).netloc
    wait = min_gap - (time.time() - _last_call.get(host_key, 0))
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=25) as r:
                _last_call[host_key] = time.time()
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            _last_call[host_key] = time.time()
            if e.code == 404:
                return None
            if e.code in (429, 500, 502, 503) and attempt < 2:
                time.sleep(4 * (attempt + 1))
                continue
            raise
        except urllib.error.URLError:
            if attempt < 2:
                time.sleep(2)
                continue
            raise
    return None


# ----------------------------------------------------------------- sources
def src_crossref_doi(doi):
    txt = http_get("https://api.crossref.org/works/" + urllib.parse.quote(doi), min_gap=0.5)
    if not txt:
        return []
    m = json.loads(txt).get("message", {})
    return [_crossref_item(m)]


def _crossref_item(m):
    year = None
    for k in ("published-print", "published-online", "issued"):
        parts = (m.get(k) or {}).get("date-parts") or [[None]]
        if parts and parts[0] and parts[0][0]:
            year = parts[0][0]
            break
    return {"source": "crossref", "title": (m.get("title") or [""])[0], "year": year,
            "authors": [surname(a.get("family", "")) for a in m.get("author", [])],
            "id": m.get("DOI")}


def src_crossref_title(title):
    q = urllib.parse.urlencode({"query.bibliographic": title, "rows": 5})
    txt = http_get("https://api.crossref.org/works?" + q, min_gap=0.5)
    if not txt:
        return []
    return [_crossref_item(m) for m in json.loads(txt).get("message", {}).get("items", [])]


def src_dblp(title):
    q = urllib.parse.urlencode({"q": title, "format": "json", "h": 5})
    txt = http_get("https://dblp.org/search/publ/api?" + q, min_gap=1.0)
    if not txt:
        return []
    hits = (json.loads(txt).get("result", {}).get("hits", {}) or {}).get("hit", []) or []
    out = []
    for h in hits:
        info = h.get("info", {})
        auth = (info.get("authors") or {}).get("author", [])
        if isinstance(auth, dict):
            auth = [auth]
        out.append({"source": "dblp", "title": info.get("title", ""),
                    "year": int(info["year"]) if str(info.get("year", "")).isdigit() else None,
                    "authors": [surname(re.sub(r"\s\d{4}$", "", a.get("text", ""))) for a in auth],
                    "id": info.get("key"), "venue": info.get("venue")})
    return out


def src_s2(title):
    q = urllib.parse.urlencode({"query": title, "limit": 5, "fields": "title,year,authors,venue,externalIds"})
    headers = {"x-api-key": os.environ["S2_API_KEY"]} if os.environ.get("S2_API_KEY") else None
    txt = http_get("https://api.semanticscholar.org/graph/v1/paper/search?" + q, headers=headers,
                   min_gap=1.1 if headers else 3.0)
    if not txt:
        return []
    return [{"source": "semantic_scholar", "title": p.get("title", ""), "year": p.get("year"),
             "authors": [surname(a.get("name", "")) for a in p.get("authors", [])],
             "id": p.get("paperId"), "venue": p.get("venue")}
            for p in json.loads(txt).get("data", []) or []]


ATOM = "{http://www.w3.org/2005/Atom}"


def _arxiv_parse(txt):
    out = []
    root = ET.fromstring(txt)
    for e in root.findall(ATOM + "entry"):
        title = (e.findtext(ATOM + "title") or "").strip()
        if not title or title.lower() == "error":
            continue
        pub = e.findtext(ATOM + "published") or ""
        out.append({"source": "arxiv", "title": re.sub(r"\s+", " ", title),
                    "year": int(pub[:4]) if pub[:4].isdigit() else None,
                    "authors": [surname(a.findtext(ATOM + "name") or "") for a in e.findall(ATOM + "author")],
                    "id": (e.findtext(ATOM + "id") or "").rsplit("/", 1)[-1]})
    return out


def src_arxiv_id(aid):
    txt = http_get("https://export.arxiv.org/api/query?" + urllib.parse.urlencode({"id_list": aid}), min_gap=3.0)
    return _arxiv_parse(txt) if txt else []


def src_arxiv_title(title):
    words = norm_title(title)
    q = urllib.parse.urlencode({"search_query": f'ti:"{words}"', "max_results": 5})
    txt = http_get("https://export.arxiv.org/api/query?" + q, min_gap=3.0)
    return _arxiv_parse(txt) if txt else []


# ----------------------------------------------------------------- matching
def score(entry_fields, cand):
    t1, t2 = norm_title(entry_fields.get("title", "")), norm_title(cand.get("title", ""))
    if not t1 or not t2:
        return 0.0
    return difflib.SequenceMatcher(None, t1, t2).ratio()


def judge(fields, cands):
    if not cands:
        return {"status": "not_found", "score": 0.0, "notes": "no candidates returned"}
    best = max(cands, key=lambda c: score(fields, c))
    s = score(fields, best)
    res = {"score": round(s, 3), "source": best["source"], "matched_title": best["title"],
           "matched_year": best.get("year"), "matched_id": best.get("id"), "notes": ""}
    if s < 0.80:
        res["status"] = "not_found"
        res["notes"] = "best title match too weak"
        return res
    notes = []
    fa = first_author(fields)
    if fa and best.get("authors") and fa not in best["authors"][:3]:
        notes.append(f"first author '{fa}' not in matched authors {best['authors'][:3]}")
    y = fields.get("year", "")
    if y.isdigit() and best.get("year") and abs(int(y) - int(best["year"])) > 1:
        notes.append(f"year {y} vs {best['year']}")
    res["status"] = "verified" if s >= 0.92 and not notes else "partial"
    res["notes"] = "; ".join(notes)
    return res


NET_ERRORS = (urllib.error.URLError, ET.ParseError, json.JSONDecodeError, OSError, ValueError)


def verify_entry(e):
    """Never report not_found unless >= 2 databases actually answered (avoid false accusations)."""
    f = e["fields"]
    cands, answered, errors = [], 0, []
    lookups = []
    if f.get("doi"):
        lookups.append((src_crossref_doi, f["doi"].strip()))
    aid = arxiv_id(f)
    if aid:
        lookups.append((src_arxiv_id, aid))
    if f.get("title"):
        t = strip_latex(f["title"])
        lookups += [(src_dblp, t), (src_s2, t), (src_arxiv_title, t), (src_crossref_title, t)]
    if not lookups:
        return {"status": "not_found", "score": 0.0, "notes": "entry has no title/doi/arXiv id"}
    for fn, arg in lookups:
        try:
            cands += fn(arg)
            answered += 1
        except NET_ERRORS as ex:
            errors.append(f"{fn.__name__}: {ex}")
            continue
        res = judge(f, cands)
        if res["status"] == "verified":
            return res
    res = judge(f, cands)
    if res["status"] == "not_found" and answered < 2:
        return {"status": "error", "score": res.get("score", 0.0),
                "notes": f"only {answered} database(s) reachable; rerun later. " + "; ".join(errors)[:300]}
    if errors:
        res["notes"] = (res.get("notes", "") + f" [{len(errors)} source(s) unreachable]").strip()
    return res


# ----------------------------------------------------------------- main
def cited_keys():
    keys = set()
    for path in glob.glob(os.path.join(ROOT, "paper", "**", "*.tex"), recursive=True):
        with open(path, encoding="utf-8", errors="ignore") as fh:
            for m in re.finditer(r"\\[a-zA-Z]*cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", fh.read()):
                keys.update(k.strip() for k in m.group(1).split(",") if k.strip())
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bib", nargs="?", default=os.path.join(ROOT, "paper", "refs.bib"))
    ap.add_argument("--only-cited", action="store_true")
    ap.add_argument("--recheck", action="store_true")
    args = ap.parse_args()

    with open(args.bib, encoding="utf-8", errors="ignore") as fh:
        entries = parse_bib(fh.read())
    try:
        with open(CACHE, encoding="utf-8") as fh:
            cache = json.load(fh)
    except (OSError, json.JSONDecodeError):
        cache = {}

    only = cited_keys() if args.only_cited else None
    bad = 0
    for e in entries:
        if only is not None and e["key"] not in only:
            continue
        prev = cache.get(e["key"])
        if (not args.recheck and prev and prev.get("entry_hash") == e["hash"]
                and prev.get("status") in ("verified", "partial", "not_found")):
            res = prev
        else:
            res = verify_entry(e)
            res["entry_hash"] = e["hash"]
            res["checked_at"] = dt.datetime.now().isoformat(timespec="seconds")
            res["bib_title"] = strip_latex(e["fields"].get("title", ""))
            cache[e["key"]] = res
            os.makedirs(os.path.dirname(CACHE), exist_ok=True)
            with open(CACHE, "w", encoding="utf-8") as fh:
                json.dump(cache, fh, indent=2, ensure_ascii=False)
        bad += res["status"] == "not_found"
        mark = {"verified": "OK ", "partial": "?? ", "not_found": "XX ", "error": "!! "}[res["status"]]
        print(f"{mark}{e['key']:<32} {res['status']:<9} score={res.get('score', 0):<5} "
              f"{res.get('source', '')} {res.get('notes', '')}")

    counts = {}
    for k, v in cache.items():
        counts[v["status"]] = counts.get(v["status"], 0) + 1
    print("\nsummary:", counts, "->", os.path.relpath(CACHE, ROOT))
    print("XX = 数据库里找不到：删除或改成真实存在的论文；?? = 需要人工核对标题/作者/年份；!! = 网络错误，稍后重跑。")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
