"""Run many commands in parallel (one per line in a batch file).

Each command's output goes to logs/<batch name>/<line number>.log. Lines that
are empty or start with # are skipped. Commands whose --out folder already has
a summary.json are skipped, so a stopped batch can simply be started again.

Usage:
    python scripts/run_many.py batches/A_capability.txt --parallel 4
"""
import argparse
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def out_dir(cmd):
    parts = shlex.split(cmd)
    return Path(parts[parts.index("--out") + 1]) if "--out" in parts else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("batch")
    ap.add_argument("--parallel", type=int, default=4)
    args = ap.parse_args()
    lines = [(i, ln.strip()) for i, ln in enumerate(Path(args.batch).read_text().splitlines(), 1)]
    cmds = [(i, c) for i, c in lines if c and not c.startswith("#")]
    logs = Path("logs") / Path(args.batch).stem
    logs.mkdir(parents=True, exist_ok=True)
    todo = [(i, c) for i, c in cmds if not (out_dir(c) and (out_dir(c) / "summary.json").exists())]
    print(f"{len(cmds)} commands, {len(cmds) - len(todo)} already done, running {len(todo)} "
          f"with {args.parallel} in parallel. Logs: {logs}/", flush=True)
    t0 = time.time()

    def run(item):
        i, cmd = item
        cmd = cmd.replace("python ", f"{shlex.quote(sys.executable)} ", 1) if cmd.startswith("python ") else cmd
        with open(logs / f"{i:03d}.log", "w") as f:
            code = subprocess.call(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
        status = "ok" if code == 0 else f"FAILED (exit {code}, see {logs}/{i:03d}.log)"
        print(f"[{(time.time() - t0) / 60:6.1f} min] line {i}: {status}", flush=True)
        return code

    with ThreadPoolExecutor(args.parallel) as pool:
        codes = list(pool.map(run, todo))
    failed = sum(c != 0 for c in codes)
    print(f"Finished in {(time.time() - t0) / 60:.1f} min; {failed} failed.")


if __name__ == "__main__":
    main()
