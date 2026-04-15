from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from extraction import DATASET_ANALYZERS
    from exploration import DATASET_EXPLORERS
else:
    from .extraction import DATASET_ANALYZERS
    from .exploration import DATASET_EXPLORERS


def main() -> int:
    dataset_choices = sorted(set(DATASET_ANALYZERS) | set(DATASET_EXPLORERS))
    parser = argparse.ArgumentParser(description="Write per-dataset data quality and exploration outputs.")
    parser.add_argument("--dataset", choices=["all", *dataset_choices], default="all")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    selected_names = dataset_choices if args.dataset == "all" else [args.dataset]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in selected_names:
        dataset_output = args.output_dir / dataset_name
        dataset_output.mkdir(parents=True, exist_ok=True)
        if dataset_name in DATASET_ANALYZERS:
            report = DATASET_ANALYZERS[dataset_name](args.data_dir / dataset_name)
            report_path = dataset_output / "data_quality.md"
            report_path.write_text(report, encoding="utf-8")
            print(f"Wrote {report_path}")
        if dataset_name in DATASET_EXPLORERS:
            exploration_report = DATASET_EXPLORERS[dataset_name](args.data_dir / dataset_name, dataset_output)
            exploration_path = dataset_output / "exploration_findings.md"
            exploration_path.write_text(exploration_report, encoding="utf-8")
            print(f"Wrote {exploration_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
