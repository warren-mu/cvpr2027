#!/usr/bin/env python3
"""Paper gate: catches the mistakes that get CVPR 2027 papers desk-rejected or that
make an LLM-written paper untrustworthy.

    python3 scripts/check_paper.py --fast     # daily (Stop hook runs this)
    python3 scripts/check_paper.py --final    # before submission: warnings become errors

Checks
  LLM   LLM Usage Statement section present (missing = desk reject)
  NUM   hard-coded decimals inside tabular (must be \\res{key}); numeric claims in prose
  RES   every \\res{key} defined; defined from logged, successful runs; numbers.tex in sync
  CITE  every \\cite key exists in .bib and is verified by scripts/verify_bib.py
  INJ   hidden text / instructions aimed at reviewers or AI tools (desk reject)
  ANON  links, acknowledgements, non-review mode
  TODO  placeholders left in the text
  FIG   \\includegraphics without a provenance sidecar (figures must come from results/)
  CLAIM CLAIMS.md rows not yet verified by the auditor
  PAGE  body longer than 8 pages (needs paper/main.pdf and pdftotext)
Escape hatch: put `% claim-ok: <reason>` at the end of a source line to skip NUM checks
on that line (e.g. hyper-parameters). The auditor reviews every claim-ok line.
"""
import argparse
import glob
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Report:
    def __init__(self, final):
        self.final, self.items = final, []

    def add(self, level, code, where, msg):
        if level == "WARN+" :  # warning now, error at --final
            level = "ERROR" if self.final else "WARN"
        self.items.append({"level": level, "code": code, "where": where, "msg": msg})

    def errors(self):
        return [i for i in self.items if i["level"] == "ERROR"]


def strip_comment(line):
    out, i = [], 0
    while i < len(line):
        if line[i] == "\\" and i + 1 < len(line):
            out.append(line[i:i + 2])
            i += 2
            continue
        if line[i] == "%":
            break
        out.append(line[i])
        i += 1
    return "".join(out)


def tex_files():
    files = []
    for p in glob.glob(os.path.join(PAPER, "**", "*.tex"), recursive=True):
        rel = os.path.relpath(p, ROOT).replace(os.sep, "/")
        if rel.startswith("paper/generated/"):
            continue
        files.append((rel, p))
    return sorted(files)


UNIT = r"\s*(em|ex|pt|cm|mm|in|bp|\\(line|text|column)width|\\(text|column)height|\\baselineskip)"
DECIMAL = re.compile(r"(?<![\w.])(\d+\.\d+)(?!\d)(?!" + UNIT + r")")
PROSE_NUM = re.compile(r"(?<![\w.\\])(\d+(?:\.\d+)?)\s*(\\%|%|\s(?:points?|pp|dB|mIoU|mAP|AP|FID|PSNR|SSIM|FPS)\b|\s*\\times|\s*×)")
TAB_BEGIN = re.compile(r"\\begin\{(tabular\*?|tabularx|longtable|tabu)\}")
TAB_END = re.compile(r"\\end\{(tabular\*?|tabularx|longtable|tabu)\}")
RES_USE = re.compile(r"\\res\{([^}]*)\}")
CITE = re.compile(r"\\[a-zA-Z]*cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}")
INJECTION = [
    (re.compile(r"\\(text)?color\s*(\[[^\]]*\])?\s*\{\s*white\s*\}", re.I), "white text"),
    (re.compile(r"\\color\s*\[rgb\]\s*\{\s*1\s*,\s*1\s*,\s*1\s*\}", re.I), "white text (rgb)"),
    (re.compile(r"\\fontsize\s*\{\s*0*\.?0*[0-2](pt)?\s*\}", re.I), "near-zero font size"),
    (re.compile(r"\\scalebox\s*\{\s*0(\.0+)?\s*\}", re.I), "zero scalebox"),
    (re.compile(r"ignore\s+(all\s+|any\s+)?(previous|prior|above)\s+instructions", re.I), "instruction to ignore instructions"),
    (re.compile(r"(as an?|you are an?)\s+(ai|llm|language model)?\s*reviewer", re.I), "text addressing a reviewer/LLM"),
    (re.compile(r"give\s+(this\s+paper\s+)?(a\s+)?(positive|high|strong)\s+(review|score|rating)", re.I), "request for a positive review"),
]
TODO = re.compile(r"\bTODO\b|\\todo\b|\bXXX\b|\?\?|\[citation needed\]|\\lipsum", re.I)
GRAPHICS = re.compile(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}")
IMG_EXT = ("", ".pdf", ".png", ".jpg", ".jpeg", ".eps")


def resolve_figure(rel_tex, name):
    base_dirs = [os.path.dirname(os.path.join(ROOT, rel_tex)), PAPER, os.path.join(ROOT, "figures")]
    for d in base_dirs:
        for ext in IMG_EXT:
            p = os.path.normpath(os.path.join(d, name + ext))
            if os.path.isfile(p):
                return p
    return None
LINK = re.compile(r"\\url\{|\\href\{|https?://|github\.com", re.I)


def check_tex(rep, defined_keys, manifest):
    full_text, used_keys, cited = "", set(), {}
    for rel, path in tex_files():
        with open(path, encoding="utf-8", errors="ignore") as fh:
            lines = fh.read().splitlines()
        in_tab = 0
        for n, raw in enumerate(lines, 1):
            where = f"{rel}:{n}"
            claim_ok = "claim-ok" in raw
            line = strip_comment(raw)
            full_text += line + "\n"
            in_tab += len(TAB_BEGIN.findall(line))

            for m in RES_USE.finditer(line):
                used_keys.add(m.group(1).strip())
            body = RES_USE.sub(" ", line)

            if not claim_ok:
                if in_tab > 0:
                    for m in DECIMAL.finditer(body):
                        rep.add("ERROR", "NUM", where,
                                f"表格里手写了数字 {m.group(1)}：请改用 \\res{{key}}（来自 results/*.json）")
                else:
                    for m in PROSE_NUM.finditer(body):
                        rep.add("WARN+", "NUM", where,
                                f"正文中的数值 '{m.group(0).strip()}' 没有用 \\res{{}}；若是超参数请在行尾加 % claim-ok: 原因")
            in_tab = max(0, in_tab - len(TAB_END.findall(line)))

            for m in CITE.finditer(line):
                for k in m.group(1).split(","):
                    if k.strip():
                        cited.setdefault(k.strip(), where)
            for pat, label in INJECTION:
                if pat.search(raw):
                    rep.add("ERROR", "INJ", where, f"疑似隐藏文字/提示注入（{label}）——CVPR 2027 直接 desk reject")
            for m in GRAPHICS.finditer(body):
                fig = resolve_figure(rel, m.group(1).strip())
                if fig is None:
                    rep.add("WARN+", "FIG", where, f"找不到图像文件 '{m.group(1)}'")
                elif "fig-ok" not in raw and not os.path.exists(fig + ".meta.json"):
                    rep.add("WARN+", "FIG", where,
                            f"图 '{m.group(1)}' 没有溯源 sidecar：数据图请用 analysis/plotlib.savefig_with_provenance 生成；"
                            "手绘方法图在该行加 % fig-ok: diagram")
            if TODO.search(body):
                rep.add("WARN+", "TODO", where, "存在占位符（TODO / ?? / lipsum）")
            if LINK.search(body):
                rep.add("WARN+", "ANON", where, "存在外链/URL：CVPR 2027 禁止扩展内容的外链，拿不准就删掉")
            if re.search(r"\\section\*?\{\s*acknowledg", body, re.I):
                rep.add("WARN+", "ANON", where, "审稿版不应包含致谢（可能泄露身份）")

    if not re.search(r"\\(sub)?section\*?\{\s*LLM Usage Statement\s*\}", full_text, re.I):
        rep.add("ERROR", "LLM", "paper/", "缺少 'LLM Usage Statement' 小节（缺失 = desk reject）。"
                "\\input{sections/llm_usage} 放在正文之后、参考文献之前")
    if full_text and not re.search(r"\\usepackage\[[^\]]*review[^\]]*\]\{cvpr\}", full_text):
        rep.add("WARN+", "ANON", "paper/main.tex", "没找到 \\usepackage[review]{cvpr}：确认用审稿模式编译（匿名+行号）")

    for k in sorted(used_keys - defined_keys):
        rep.add("ERROR", "RES", "paper/", f"\\res{{{k}}} 未定义：results/*.json 里没有这个指标（不要手填，去开 [EXP] 任务）")
    for k in sorted(used_keys & defined_keys):
        info = manifest.get(k, {})
        if info.get("unlogged"):
            rep.add("WARN+", "RES", info.get("source", "?"), f"\\res{{{k}}} 来自没有 run_logged 记录的结果文件")
        if info.get("exit_code") not in (0, None):
            rep.add("WARN+", "RES", info.get("source", "?"), f"\\res{{{k}}} 来自失败的运行（exit_code={info.get('exit_code')}）")
    return cited


def check_numbers_sync(rep):
    try:
        exp = load_module("export_numbers")
        defs, manifest, problems = exp.collect()
        tex = exp.render(defs)
    except Exception as e:  # noqa: BLE001
        rep.add("WARN", "RES", "scripts/export_numbers.py", f"无法运行：{e}")
        return set(), {}
    path = os.path.join(PAPER, "generated", "numbers.tex")
    current = open(path, encoding="utf-8").read() if os.path.exists(path) else ""
    if current != tex:
        rep.add("WARN+", "RES", "paper/generated/numbers.tex", "与 results/*.json 不同步：运行 python3 scripts/export_numbers.py")
    for p in problems:
        rep.add("ERROR" if p.startswith("ERROR") else "WARN", "RES", "results/", p.split(None, 1)[1])
    defined = set(re.findall(r"\\@namedef\{res@([^}]*)\}", current))
    return defined, manifest


def check_citations(rep, cited):
    if not cited:
        return
    bibs = glob.glob(os.path.join(PAPER, "**", "*.bib"), recursive=True)
    vb = load_module("verify_bib")
    entries = {}
    for b in bibs:
        with open(b, encoding="utf-8", errors="ignore") as fh:
            for e in vb.parse_bib(fh.read()):
                entries[e["key"]] = e
    try:
        with open(os.path.join(ROOT, "logs", "bib_verification.json"), encoding="utf-8") as fh:
            cache = json.load(fh)
    except (OSError, json.JSONDecodeError):
        cache = {}
    unchecked = []
    for key, where in sorted(cited.items()):
        if key not in entries:
            rep.add("ERROR", "CITE", where, f"引用键 '{key}' 不在任何 .bib 中")
            continue
        c = cache.get(key)
        if not c or c.get("entry_hash") != entries[key]["hash"]:
            unchecked.append(key)
            continue
        st = c.get("status")
        if st == "not_found":
            rep.add("ERROR", "CITE", where, f"'{key}' 在数据库中找不到（疑似幻觉引用，ICLR/CVPR 会 desk reject）")
        elif st == "partial":
            rep.add("WARN+", "CITE", where, f"'{key}' 只部分匹配（{c.get('notes') or c.get('matched_title')}），需人工核对")
        elif st == "error":
            rep.add("WARN+", "CITE", where, f"'{key}' 上次校验网络出错，重跑 verify_bib.py")
    if unchecked:
        rep.add("ERROR", "CITE", "paper/refs.bib",
                f"{len(unchecked)} 条被引用的条目还没校验（或校验后被改动）：{', '.join(unchecked[:8])}"
                f"{' …' if len(unchecked) > 8 else ''}。运行 python3 scripts/verify_bib.py --only-cited")


def check_claims(rep):
    path = os.path.join(ROOT, "CLAIMS.md")
    if not os.path.exists(path):
        return
    counts = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            cells = [c.strip().lower() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 5 and re.match(r"^c\d+", cells[0]):
                counts[cells[-1]] = counts.get(cells[-1], 0) + 1
    open_ = {k: v for k, v in counts.items() if k != "verified"}
    if open_:
        rep.add("WARN+", "CLAIM", "CLAIMS.md", f"还有未被 auditor 核验的主张：{open_}")


def check_pages(rep):
    pdf = os.path.join(PAPER, "main.pdf")
    if not os.path.exists(pdf) or not shutil.which("pdftotext"):
        if rep.final:
            rep.add("WARN", "PAGE", "paper/main.pdf", "未检查页数（需要编译出 paper/main.pdf 且安装 pdftotext）")
        return
    page = 1
    while page <= 30:
        out = subprocess.run(["pdftotext", "-f", str(page), "-l", str(page), "-layout", pdf, "-"],
                             capture_output=True, text=True)
        if out.returncode != 0 or not out.stdout.strip():
            break
        lines = [l.strip() for l in out.stdout.splitlines() if l.strip()]
        for idx, l in enumerate(lines):
            if re.match(r"^(\d+\.?\s*)?(References|LLM Usage Statement)\b", l):
                at_top = idx <= 2
                body_pages = page - 1 + (0 if at_top else 1)
                if body_pages > 8:
                    rep.add("ERROR", "PAGE", "paper/main.pdf",
                            f"正文约 {body_pages} 页（'{l[:30]}' 出现在第 {page} 页），超过 8 页上限")
                return
        page += 1


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--fast", action="store_true")
    g.add_argument("--final", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    rep = Report(final=args.final)

    if not tex_files():
        print("paper/ 下还没有 .tex 文件，跳过。")
        sys.exit(0)
    defined, manifest = check_numbers_sync(rep)
    cited = check_tex(rep, defined, manifest)
    check_citations(rep, cited)
    check_claims(rep)
    if args.final:
        check_pages(rep)

    errs = rep.errors()
    warns = [i for i in rep.items if i["level"] == "WARN"]
    for i in rep.items:
        print(f"{i['level']:<5} [{i['code']}] {i['where']}: {i['msg']}")
    mode = "final" if args.final else "fast"
    print(f"\n[{mode}] {len(errs)} error(s), {len(warns)} warning(s)")
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    with open(os.path.join(ROOT, "logs", "check_report.json"), "w", encoding="utf-8") as fh:
        json.dump({"mode": mode, "items": rep.items}, fh, indent=2, ensure_ascii=False)
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
