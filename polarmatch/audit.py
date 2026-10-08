from __future__ import annotations

import argparse
import json
from pathlib import Path


from polarmatch.protocol import audit_polar_holdout, load_pair_table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the public polar-holdout pair split without training."
    )
    parser.add_argument("--train-pairs", required=True, type=Path)
    parser.add_argument("--test-pairs", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    audit = audit_polar_holdout(
        load_pair_table(args.train_pairs),
        load_pair_table(args.test_pairs),
    )
    report = {
        "mode": "read-only polar-holdout audit; no training",
        "split": audit.to_dict(),
        "passed": audit.polar_holdout_passed,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    if not audit.polar_holdout_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
