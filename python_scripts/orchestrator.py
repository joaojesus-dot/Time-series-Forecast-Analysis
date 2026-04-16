from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from extraction import DATASET_ANALYZERS
    from exploration import write_boiler_outputs
else:
    from .extraction import DATASET_ANALYZERS
    from .exploration import write_boiler_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Write boiler-only KDD and physical-analysis outputs.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    dataset_name = "chinese_boiler_dataset"
    dataset_output = args.output_dir / dataset_name
    dataset_output.mkdir(parents=True, exist_ok=True)

    quality_report = DATASET_ANALYZERS[dataset_name](args.data_dir / dataset_name)
    quality_path = dataset_output / "data_quality.md"
    quality_path.write_text(quality_report, encoding="utf-8")
    print(f"Wrote {quality_path}")

    write_boiler_outputs(args.data_dir / dataset_name, dataset_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
