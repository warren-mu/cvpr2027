---
name: gate
description: 提交前终检（G4）：重新生成数字、校验所有被引用文献、编译 PDF、跑 check_paper --final，给出逐项红绿清单。用户输入 /gate 或说"终检"时使用。
---

# /gate —— 提交前终检

依次执行，任何一步失败都要如实报告，不要绕过：

```bash
python3 scripts/export_numbers.py
python3 scripts/verify_bib.py paper/refs.bib --only-cited
(cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex)
python3 scripts/check_paper.py --final
```

然后输出清单（✅ / ❌ + 证据）：
- [ ] LLM Usage Statement 存在，且如实列出用过的工具（Claude Code、Codex 等）与用途
- [ ] 正文 ≤ 8 页（参考文献和 LLM Usage Statement 不计）
- [ ] 所有 `\res{}` 有定义，且来源 run 的 exit_code = 0
- [ ] 所有被引用文献 verified（partial 已人工核对）
- [ ] CLAIMS.md 全部 verified
- [ ] 匿名：无作者/单位/致谢/GitHub/外链；`\usepackage[review]{cvpr}`
- [ ] 无隐藏文字 / 提示注入
- [ ] 补充材料同样匿名、无外链（截止 2026-11-23 AoE）
- [ ] OpenReview：已注册（11-10 AoE 前）、全部作者档案完整

全部 ✅ 后，请人类亲自做最后一次通读，并亲自点击提交。
