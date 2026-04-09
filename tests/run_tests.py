#!/usr/bin/env python3
"""
CLI entry point: orchestrates test execution, LLM judging, and result output.

Usage:
    python tests/run_tests.py                    # Run all tests
    python tests/run_tests.py --category A       # Run only category A
    python tests/run_tests.py --file zero_shot   # Run single file (partial match)
    python tests/run_tests.py --skip-judge       # Skip LLM judging
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

# Ensure the tests/ directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_config import get_test_entries, REPO_ROOT
from runner import run_test, stop_mcp_server
from judge import judge_result


def parse_args():
    p = argparse.ArgumentParser(description="Workshop test harness with LLM-as-Judge")
    p.add_argument("--category", choices=["A", "B", "C", "D", "E"],
                    help="Run only this category")
    p.add_argument("--file", type=str, default=None,
                    help="Run a single file (partial name match)")
    p.add_argument("--skip-judge", action="store_true",
                    help="Skip LLM judging, just capture output")
    p.add_argument("--output-dir", type=str,
                    default=os.path.join(REPO_ROOT, "tests"),
                    help="Directory for results files")
    return p.parse_args()


def print_header():
    print()
    print("=" * 90)
    print("  WORKSHOP TEST HARNESS — LLM-as-Judge")
    print("=" * 90)
    print()


def print_summary(results):
    print()
    print("=" * 90)
    print("  TEST RESULTS SUMMARY")
    print("=" * 90)
    print(f"  {'File':<45} {'Cat':>3}  {'Status':<10} {'Pass':>5}  {'Score':>5}  {'Time':>6}")
    print("-" * 90)

    for r in results:
        name = r["file_name"]
        if len(name) > 44:
            name = name[:41] + "..."
        cat = r["category"]
        status = r["status"]
        passed = r.get("pass")
        score = r.get("score", "")
        dur = f"{r['duration_s']:.0f}s"

        if passed is True:
            pass_str = "PASS"
        elif passed is False:
            pass_str = "FAIL"
        else:
            pass_str = "SKIP"

        print(f"  {name:<45} {cat:>3}  {status:<10} {pass_str:>5}  {str(score):>5}  {dur:>6}")

    print("-" * 90)

    total = len(results)
    passed = sum(1 for r in results if r.get("pass") is True)
    failed = sum(1 for r in results if r.get("pass") is False)
    skipped = sum(1 for r in results if r.get("pass") is None)
    timed_out = sum(1 for r in results if r.get("status") == "timeout")
    total_time = sum(r.get("duration_s", 0) for r in results)

    print(f"  Total: {total} | Passed: {passed} | Failed: {failed} | "
          f"Skipped: {skipped} | Timeout: {timed_out} | "
          f"Total time: {total_time:.0f}s")
    print("=" * 90)
    print()


def save_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # -- JSON (full details) --
    json_path = os.path.join(output_dir, f"results_{ts}.json")
    with open(json_path, "w") as f:
        json.dump({
            "timestamp": ts,
            "total": len(results),
            "passed": sum(1 for r in results if r.get("pass") is True),
            "failed": sum(1 for r in results if r.get("pass") is False),
            "skipped": sum(1 for r in results if r.get("pass") is None),
            "results": results,
        }, f, indent=2, default=str)
    print(f"  JSON results: {json_path}")

    # -- CSV (summary) --
    csv_path = os.path.join(output_dir, f"results_{ts}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "category", "status", "exit_code", "duration_s",
                          "pass", "score", "reasoning"])
        for r in results:
            writer.writerow([
                r["file_name"],
                r["category"],
                r["status"],
                r.get("exit_code", ""),
                r.get("duration_s", 0),
                r.get("pass", ""),
                r.get("score", ""),
                r.get("reasoning", ""),
            ])
    print(f"  CSV results:  {csv_path}")

    # -- Also write latest symlink-style files --
    latest_json = os.path.join(output_dir, "results.json")
    latest_csv = os.path.join(output_dir, "results.csv")
    with open(latest_json, "w") as f:
        json.dump({
            "timestamp": ts,
            "total": len(results),
            "passed": sum(1 for r in results if r.get("pass") is True),
            "failed": sum(1 for r in results if r.get("pass") is False),
            "skipped": sum(1 for r in results if r.get("pass") is None),
            "results": results,
        }, f, indent=2, default=str)
    with open(latest_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "category", "status", "exit_code", "duration_s",
                          "pass", "score", "reasoning"])
        for r in results:
            writer.writerow([
                r["file_name"], r["category"], r["status"],
                r.get("exit_code", ""), r.get("duration_s", 0),
                r.get("pass", ""), r.get("score", ""), r.get("reasoning", ""),
            ])
    print(f"  Latest:       {latest_json}")
    print(f"                {latest_csv}")


def main():
    args = parse_args()
    print_header()

    entries = get_test_entries()

    # -- Filter --
    if args.category:
        entries = [e for e in entries if e["category"] == args.category]
        print(f"  Filtering: category {args.category} ({len(entries)} tests)")
    if args.file:
        entries = [e for e in entries if args.file.lower() in os.path.basename(e["file_path"]).lower()]
        print(f"  Filtering: file match '{args.file}' ({len(entries)} tests)")

    if not entries:
        print("  No tests matched the filter. Exiting.")
        return

    total = len(entries)
    print(f"  Running {total} tests {'(no LLM judge)' if args.skip_judge else '(with LLM judge)'}")
    print()

    results = []
    for i, entry in enumerate(entries):
        name = os.path.basename(entry["file_path"])
        print(f"  [{i+1}/{total}] {name} (cat={entry['category']}) ...", end=" ", flush=True)

        result = run_test(entry)

        if not args.skip_judge and result["status"] != "skipped":
            verdict = judge_result(result, entry["task_description"])
            result.update(verdict)
        elif result["status"] == "skipped":
            result["pass"] = None
            result["score"] = 0
            result["reasoning"] = result.get("skip_reason", "Skipped")
        else:
            # skip-judge mode: mark based on exit code
            result["pass"] = result.get("exit_code") == 0
            result["score"] = 10 if result["pass"] else 0
            result["reasoning"] = "Judge skipped"

        status = result["status"]
        p = "PASS" if result.get("pass") is True else ("FAIL" if result.get("pass") is False else "SKIP")
        score = result.get("score", "")
        dur = f"{result['duration_s']:.0f}s"
        print(f"{status} | {p} | score={score} | {dur}")

        results.append(result)

        # Brief pause between tests to avoid API rate limits
        if i < total - 1 and result["status"] != "skipped":
            time.sleep(1)

    # -- Cleanup --
    stop_mcp_server()

    print_summary(results)
    save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
