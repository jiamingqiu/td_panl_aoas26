"""Prepare the CGM input used by the TD-PANL example.

The source archive must be downloaded manually from the Jaeb public data site
after accepting its terms of use. This script does not download or redistribute
the source data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

import pandas as pd

from pyseqdx.utilities.gen_data_cgm import (
    bin_and_interp_segment,
    label_hypoevent,
    label_segment,
)


CGM_MEMBER = "Data Tables/BDataCGM.txt"
ROSTER_MEMBER = "Data Tables/BPtRoster.txt"
DEFAULT_OUTPUT = Path(__file__).with_name("CGM.csv")
DEFAULT_LABELED_OUTPUT = Path(__file__).with_name("cgm_interp_label.csv")


def prepare_cgm(archive: Path) -> pd.DataFrame:
    with zipfile.ZipFile(archive) as source_zip:
        missing = {
            CGM_MEMBER,
            ROSTER_MEMBER,
        }.difference(source_zip.namelist())
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise FileNotFoundError(
                f"Required file(s) not found in {archive}: {missing_list}"
            )

        with source_zip.open(CGM_MEMBER) as cgm_file:
            measurements = pd.read_csv(cgm_file, sep="|")
        with source_zip.open(ROSTER_MEMBER) as roster_file:
            roster = pd.read_csv(roster_file, sep="|")

    measurements = measurements.drop(
        columns=["RecID", "BCGMDeviceType", "BFileType", "CalBG"]
    ).rename(
        columns={
            "PtID": "id",
            "DeviceDaysFromEnroll": "day",
            "DeviceTm": "time",
            "Glucose": "gl",
        }
    )
    measurements["dummy_datetime"] = (
        pd.Timestamp("2000-01-01")
        + pd.to_timedelta(measurements["day"], unit="D")
        + pd.to_timedelta(measurements["time"])
    )

    roster = roster.drop(columns=["RecID"]).rename(
        columns={"PtID": "id", "BCaseControlStatus": "label"}
    )
    roster["y"] = roster["label"].map({"Case": 1, "Control": 0})
    if roster["y"].isna().any():
        unknown = sorted(roster.loc[roster["y"].isna(), "label"].unique())
        raise ValueError(f"Unknown case/control label(s): {unknown}")

    prepared = pd.merge(measurements, roster, on="id")
    prepared = prepared[
        ["id", "day", "time", "gl", "dummy_datetime", "label", "y"]
    ]
    return prepared


def prepare_labeled_cgm(prepared: pd.DataFrame) -> pd.DataFrame:
    labeled = label_segment(prepared.dropna().copy(), gap_sec=3600)
    labeled = bin_and_interp_segment(labeled)
    labeled = label_hypoevent(labeled, low_val=60, duration_sec=1200)
    return labeled


def write_csv(data: pd.DataFrame, output: Path, description: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False)

    print(
        f"Wrote {description}: {len(data):,} rows for "
        f"{data['id'].nunique()} participants"
    )
    print(f"Output: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "archive",
        type=Path,
        help="Path to the manually downloaded SevereHypoDataset.zip archive.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"CGM CSV path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--labeled-output",
        type=Path,
        default=DEFAULT_LABELED_OUTPUT,
        help=(
            "Interpolated, event-labeled CSV path "
            f"(default: {DEFAULT_LABELED_OUTPUT})."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive = args.archive.expanduser().resolve()
    output = args.output.expanduser().resolve()
    labeled_output = args.labeled_output.expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {archive}")
    if output == labeled_output:
        raise ValueError("--output and --labeled-output must be different paths")

    existing = [path for path in (output, labeled_output) if path.exists()]
    if existing and not args.force:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output already exists: {paths}; use --force to replace")

    prepared = prepare_cgm(archive)
    write_csv(prepared, output, "CGM data")
    labeled = prepare_labeled_cgm(prepared)
    write_csv(labeled, labeled_output, "interpolated, event-labeled CGM data")


if __name__ == "__main__":
    main()
