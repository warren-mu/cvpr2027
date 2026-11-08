#!/usr/bin/env bash
# 跨模型审稿备用方案：没配置 Codex MCP 时，直接调用 Codex CLI 做一次只读评审。
#   bash scripts/codex_review.sh <轮次N>
# 需要：npm i -g @openai/codex 并已登录。不同版本参数可能不同，以 `codex exec --help` 为准。
set -euo pipefail
cd "$(dirname "$0")/.."
ROUND="${1:-1}"
OUT="reviews/ROUND_${ROUND}_codex.md"
command -v codex >/dev/null || { echo "未找到 codex CLI"; exit 1; }
mkdir -p reviews

read -r -d '' PROMPT <<'EOF' || true
You are a strict but fair reviewer for CVPR 2027. You did not take part in writing this paper.
Read the LaTeX sources under paper/ (or paper/main.pdf if present), results/*.json, CLAIMS.md and
lit/related_work.md. Do NOT read anything under reviews/. Do NOT modify any file.

Write your review in Markdown with these sections:
1. Summary (2-3 sentences)
2. Overall score (1-6, CVPR scale) and confidence (1-5)
3. Fatal issues that would cause rejection, ranked; for each give file:line evidence, the minimal fix, and an effort estimate
4. Major issues
5. Minor issues
6. Questions for the authors

Check specifically: whether each claimed contribution is supported by an experiment; baseline strength and
fairness; missing recent related work (last 12 months); consistency between numbers in tables and in the text;
over-claiming wording; and desk-reject risks (missing "LLM Usage Statement", de-anonymization, external links,
more than 8 pages, hidden text).
EOF

codex exec --sandbox read-only "$PROMPT" > "$OUT"
echo "saved $OUT"
