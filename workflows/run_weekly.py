#!/usr/bin/env python3
"""
Run the full weekly update pipeline end-to-end.

Steps (in order):
  1. Deal processing          — parse deal emails, update early_deals.csv + DB
  2. Tracking updates         — check Notion tracked companies for news alerts
  3. Pipeline news search     — search Parallel API for pipeline company news
  4. Pipeline articles → DB   — verify & add pipeline news articles to tracking
  5. Weekly update email      — generate HTML, run all checks, send email

Each step is run as a subprocess. A failed step logs a warning but does not
abort the pipeline — later steps that depend on fresh data will gracefully
fall back to whatever data is already on disk.

Usage:
    python workflows/run_weekly.py                   # full pipeline
    python workflows/run_weekly.py --skip-deals      # skip step 1
    python workflows/run_weekly.py --dry-run         # run everything except sending the email
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime

PYTHON = sys.executable
WORKFLOWS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(WORKFLOWS)


def run_step(label, cmd, cwd=ROOT):
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {label}")
    print(f"  {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'=' * width}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"\n⚠️  '{label}' exited with code {result.returncode} — continuing pipeline")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Run the full weekly update pipeline")
    parser.add_argument("--skip-deals", action="store_true",
                        help="Skip step 1 (deal processing)")
    parser.add_argument("--skip-tracking", action="store_true",
                        help="Skip steps 2–4 (tracking + pipeline news)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run all data steps but skip sending the email")
    args = parser.parse_args()

    start = datetime.now()
    print(f"\n🚀  Weekly update pipeline starting — {start.strftime('%Y-%m-%d %H:%M')}")

    steps = []

    if not args.skip_deals:
        steps.append((
            "Step 1 — Deal processing",
            [PYTHON, os.path.join(WORKFLOWS, "deals.py")]
        ))
    else:
        print("\n⏭  Skipping step 1 (--skip-deals)")

    if not args.skip_tracking:
        steps.append((
            "Step 2 — Tracking updates (Notion alerts)",
            [PYTHON, os.path.join(WORKFLOWS, "tracking.py")]
        ))
        steps.append((
            "Step 3 — Pipeline news search",
            [PYTHON, os.path.join(WORKFLOWS, "tracking.py"), "--pipeline-news"]
        ))
        steps.append((
            "Step 4 — Add pipeline articles to tracking DB",
            [PYTHON, os.path.join(WORKFLOWS, "add_pipeline_articles_to_tracking.py")]
        ))
    else:
        print("\n⏭  Skipping steps 2–4 (--skip-tracking)")

    weekly_cmd = [PYTHON, os.path.join(WORKFLOWS, "weekly_update.py")]
    if args.dry_run:
        weekly_cmd.append("--no-send")
    steps.append(("Step 5 — Weekly update email", weekly_cmd))

    results = {}
    for label, cmd in steps:
        results[label] = run_step(label, cmd)

    elapsed = int((datetime.now() - start).total_seconds() // 60)
    width = 60
    print(f"\n{'=' * width}")
    print(f"  Pipeline complete — {elapsed}m elapsed")
    print(f"{'=' * width}")
    for label, ok in results.items():
        icon = "✅" if ok else "⚠️ "
        print(f"  {icon}  {label}")
    print()

    failed = [label for label, ok in results.items() if not ok]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
