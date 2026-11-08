# paper/

1. 下载官方 CVPR 2027 author kit（LaTeX），解压到本目录，主文件命名为 `main.tex`，审稿模式：`\usepackage[review]{cvpr}`。
2. 在 `main.tex` 导言区加入：
   ```latex
   \input{generated/numbers}   % 由 scripts/export_numbers.py 生成，提供 \res{key}
   ```
3. 在正文结束、`\bibliography` 之前加入：
   ```latex
   \input{sections/llm_usage}  % 缺失 = desk reject
   ```
4. 表格里的数字一律写 `\res{key}`；参考 `docs/example_usage.tex`。
5. 引用文件统一用 `refs.bib`，由 scout 维护并经 `scripts/verify_bib.py` 校验。
