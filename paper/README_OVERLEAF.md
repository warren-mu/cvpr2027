# CVPR 2027 初稿：上传到 Overleaf 与合作者协作

## 上传
Overleaf → New Project → Upload Project → 选这个 zip。Compiler 用 pdfLaTeX，主文件是 `main.tex`。

## 文件
- `cvpr.sty`、`ieeenat_fullname.bst`：官方 CVPR 2027 author kit（github.com/cvpr-org/author-kit，"updated for CVPR 2027"），**不要改**：改模板、页边距或字体会被 desk reject。
- `main.tex`：审稿模式（匿名、带行号）。
- `sections/`：摘要、引言、相关工作、方法、数据构造、实验设置、结果、局限与结论、LLM Usage Statement。
- `refs.bib`：27 条引用，全部经过项目的文献核验（arXiv / DBLP / Semantic Scholar / Crossref 或官方论文集）。
- `generated/numbers.tex`：由项目仓库里的脚本从实验日志自动生成，提供 `\res{key}`。

## 写作规则（合作者请务必遵守）
1. **实验数字只能用 `\res{key}`**，不要手打。红色的 `[??]` 是还没有结果的数字。结果出来后，用仓库重新生成的 `generated/numbers.tex` 整个替换这里的同名文件。
2. **新加引用之前，先把条目发回来做核验**。引用不存在的论文会被 desk reject。
3. **只写看到的**：用 "we observe"，不写 "we prove"；没有消融支撑的因果解释不要写；负结果如实写。
4. **匿名**：正文和补充材料不能出现作者、单位、致谢、GitHub 链接；外链一律不放。
5. **不能有任何隐藏文字**，也不能有写给 AI 审稿的指令（CVPR 2027 明文规定 desk reject）。红色 `[TODO: ...]` 是可见的草稿标记，提交前必须全部删掉。
6. **LLM Usage Statement** 必须保留，放在正文之后、参考文献之前，内容要和 OpenReview 上的声明一致。
7. 正文 8 页（不含参考文献和 LLM Usage Statement）。目前正文约 6.7 页，结果部分写完后要压缩。

## 提交前
把 Overleaf 的最终版本同步回项目仓库的 `paper/`，跑 `python3 scripts/check_paper.py --final`，必须全绿。
