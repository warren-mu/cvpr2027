# CLAIMS 账本

规则：论文里每一句"我们更好 / 我们首次 / 我们提出的 X 带来 Y"都要在这里有一行。
writer 登记（状态写 `pending`），只有 auditor 能改成 `verified` / `weaken` / `failed`（hook 强制）。

| ID | 论文中的说法（位置） | 证据（results / 图 / 文献） | 备注 | 状态 |
|---|---|---|---|---|
| C1 | 例：在 ADE20K 上 mIoU 比最强 baseline 高 \res{gain_miou_ade20k}（abstract, §4.2 Tab.1） | results/exp01_main.json | 3 seeds | pending |
