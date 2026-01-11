import csv
import re
from pathlib import Path

src = Path(r"../../../output/train.txt")
dst = src.with_suffix(".csv")

kv_re = re.compile(r"(\w+)=(-?\d+(?:\.\d+)?)")

rows = []
all_keys = set()

for line in src.read_text(encoding="utf-8", errors="ignore").splitlines():
    kvs = dict((k, v) for k, v in kv_re.findall(line))
    if not kvs or "epoch" not in kvs:
        continue

    row = {}
    for k, v in kvs.items():
        row[k] = int(v) if (k == "epoch" and v.isdigit()) else float(v)

    rows.append(row)
    all_keys.update(row.keys())

preferred = ["critic_loss", "fake_loss", "generator_loss", "penalty", "real_loss"]
rest = [k for k in preferred if k in all_keys] + sorted(
    k for k in all_keys if k not in {"epoch", *preferred}
)
fieldnames = ["epoch"] + rest

rows.sort(key=lambda r: r["epoch"])

with dst.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow(r)

print(f"Wrote {len(rows)} rows -> {dst}")
