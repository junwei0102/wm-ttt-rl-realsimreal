#!/bin/bash
# 按 related_work/index.json 把所有论文的 PDF + LaTeX 源码抓回来。
# paper.pdf 和 source/ 不进版本库（体积 + 版权），clone 之后跑一次这个就行。
#
#   bash scripts/fetch-all.sh             # 全部（已有 paper.pdf 的跳过）
#   bash scripts/fetch-all.sh 04_rl       # 只抓某个主题
#
# 故意串行：arXiv 要求礼貌抓取，fetch-paper.sh 每次请求后会 sleep。全量约 54 篇、1.2 GB。

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

FILTER="${1:-}"
LIST=$(mktemp)
python3 - "$FILTER" <<'EOF' > "$LIST" || exit 1
import json, sys
f = sys.argv[1]
d = json.load(open("related_work/index.json"))
for p in d["surveys"] + d["papers"]:
    if not f or p["topic"] == f:
        print(f'{p["arxiv_id"]}\t{p["topic"]}/{p["slug"]}')
EOF

OK=0; SKIP=0; FAIL=0
while IFS=$'\t' read -r ID DEST; do
  if [ -s "related_work/$DEST/paper.pdf" ]; then SKIP=$((SKIP+1)); continue; fi
  if bash scripts/fetch-paper.sh "$ID" "$DEST"; then OK=$((OK+1)); else FAIL=$((FAIL+1)); echo "  ✗ $ID $DEST"; fi
done < "$LIST"
rm -f "$LIST"

echo "完成：新抓 $OK，跳过 $SKIP，失败 $FAIL"
