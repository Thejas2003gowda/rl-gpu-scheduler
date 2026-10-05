"""Parse Philly's cluster_job_log into data/philly_jobs.csv and print workload stats.

Usage:
    python scripts/prepare_philly.py --job-log /path/to/trace-data/cluster_job_log
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpusched import add_duration_estimates, parse_philly_job_log  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job-log", required=True, help="path to Philly cluster_job_log (JSON)")
    ap.add_argument("--out", default="data/philly_jobs.csv")
    ap.add_argument("--estimate", choices=["mean", "geomean"], default="mean",
                    help="how to average a user's past job lengths")
    args = ap.parse_args()

    df, dropped = parse_philly_job_log(args.job_log)
    df = add_duration_estimates(df, kind=args.estimate)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    hours = df["duration"] / 3600
    print(f"Kept {len(df):,} jobs from {df['user'].nunique()} users in {df['vc'].nunique()} virtual clusters")
    print(f"Dropped: {dropped}")
    print(f"Span: {df['submit_time'].max() / 86400:.1f} days")
    print("Status:", df["status"].value_counts().to_dict())
    q = np.percentile(hours, [25, 50, 75, 90, 99, 100])
    print("Duration (hours) p25/p50/p75/p90/p99/max:", " / ".join(f"{v:.3g}" for v in q))
    print(f"Jobs finishing within 2.3 minutes: {(df['duration'] <= 138).mean():.1%}")
    print("GPU request shares:", (df["gpus"].value_counts(normalize=True).sort_index().round(3)).to_dict())
    print("Estimate source:", df["est_source"].value_counts().to_dict())
    err = np.abs(np.log(df["est_duration"] / df["duration"]))
    print(f"Estimate error |log(est/true)| median: {np.median(err):.2f} "
          f"(0 = perfect, 0.69 = off by 2x, 2.3 = off by 10x)")
    print("\nLargest virtual clusters (jobs):")
    print(df["vc"].value_counts().head(10).to_string())
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
