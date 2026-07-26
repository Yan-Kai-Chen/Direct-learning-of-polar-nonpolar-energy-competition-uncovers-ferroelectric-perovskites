from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
SRC = MODULE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ae3gnn.data import audit_training_table, select_numeric_features


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the fixed public training table without training."
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    frame, audit = audit_training_table(args.data)
    numeric = select_numeric_features(frame)
    report = {
        "mode": "read-only data and split audit; no training",
        "data": audit.to_dict(),
        "numeric_feature_count": len(numeric.columns),
        "row_idx_used_as_feature": "row_idx" in numeric.columns,
        "split_used_as_feature": "split" in numeric.columns,
        "passed": audit.passed,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    if not audit.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
