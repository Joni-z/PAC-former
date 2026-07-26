"""Scrape every finished run's test line out of logs/ into one table.

The logs are tqdm-polluted (carriage returns) and named by config, so reading
results by hand is error-prone -- which is exactly how the sec. 13.16 mask-shape
results sat undiscovered. This regenerates the evidence ledger's numbers from
the logs on demand.

    python scripts/collect_results.py                # all runs
    python scripts/collect_results.py tuab           # filter by substring
    python scripts/collect_results.py --csv out.csv
"""

import argparse
import glob
import os
import re
import sys

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")

# `[probe] test | k=v ...`      pretrain.py
# `[biot] test@best | (...) k=v ...` / `test@last`   baseline_biot.py
# `test | k=v ...`              train.py (supervised from scratch)
TEST_RE = re.compile(
    r"^\[?(?:probe|biot)?\]?\s*test(?P<tag>@\w+)?\s*\|(?:\s*\([^)]*\))?\s*(?P<kv>.+)$"
)
KV_RE = re.compile(r"(\w+)=([-\d.naN]+)")


def parse(path):
    """Return list of (tag, {metric: value}) for every test line in the log."""
    out = []
    try:
        with open(path, "r", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return out
    for line in text.replace("\r", "\n").split("\n"):
        line = line.strip()
        m = TEST_RE.match(line)
        if not m:
            continue
        kv = {k: float(v) for k, v in KV_RE.findall(m.group("kv"))}
        if kv:
            out.append((m.group("tag") or "", kv))
    return out


# Job ids at/after this were launched with eval.select_key() corrected to match
# BIOT (multiclass selects on cohen_kappa, not balanced_accuracy -- AGENT.md 13.31).
# ANY multiclass run below it picked its best epoch on the wrong metric and its test
# number is not comparable. Binary runs (TUAB/TUSZ/CHB-MIT) always selected on AUROC
# and are unaffected at every job id.
SELECT_FIX_JOBID = 14686664
MULTICLASS = ("tuev", "sleep")           # 6-class and 5-class; the affected tasks


def is_stale(run: str, jobid: int | None) -> bool:
    if jobid is None or jobid >= SELECT_FIX_JOBID:
        return False
    return any(d in run for d in MULTICLASS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("filter", nargs="?", default="", help="substring of the run name")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--all", action="store_true",
                    help="also show runs invalidated by the sec. 13.31 selection-metric "
                         "fix (hidden by default so they cannot be misread as current)")
    args = ap.parse_args()

    rows, stale = [], []
    # named logs only (the %j.log copies are duplicates of these)
    for path in sorted(glob.glob(os.path.join(LOG_DIR, "*-*.log"))):
        name = os.path.basename(path)
        if re.fullmatch(r"\d+\.log", name):
            continue
        m = re.search(r"-(\d+)\.log$", name)
        jobid = int(m.group(1)) if m else None
        run = re.sub(r"-\d+\.log$", "", name)
        if args.filter and args.filter not in run:
            continue
        bad = is_stale(run, jobid)
        for tag, kv in parse(path):
            (stale if bad else rows).append((run + tag + ("  [STALE]" if bad else ""), kv))

    if stale and not args.all:
        print(f"note: {len(stale)} multiclass result(s) hidden — selected on the wrong "
              f"validation metric before job {SELECT_FIX_JOBID} (AGENT.md 13.31). "
              f"Pass --all to see them.", file=sys.stderr)
    elif stale:
        rows += stale

    if not rows:
        print("no finished runs matched", file=sys.stderr)
        return 1

    metrics = []
    for _, kv in rows:
        for k in kv:
            if k not in metrics:
                metrics.append(k)
    w = max(len(r) for r, _ in rows)
    print(f"{'run':<{w}} " + " ".join(f"{m:>18}" for m in metrics))
    for run, kv in sorted(rows, key=lambda r: r[0]):
        print(f"{run:<{w}} " + " ".join(
            f"{kv[m]:>18.4f}" if m in kv else " " * 18 for m in metrics))

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["run"] + metrics)
            for run, kv in sorted(rows, key=lambda r: r[0]):
                wr.writerow([run] + [kv.get(m, "") for m in metrics])
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
