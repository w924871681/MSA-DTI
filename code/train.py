"""Prepare Bitbrains source cohorts and run native source training (no target future access)."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from inputs import CODE, WORK, cases, file_hash, materialize, read

RUN = WORK / "training"
ORIGINAL = RUN / "original"


def _write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pending")
    tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def prepare(raw: Path) -> None:
    if not raw.is_file():
        raise FileNotFoundError(raw)
    contract = read(CODE / "native/protocol/source_development.json")
    if file_hash(raw) != contract["archive_sha256"]:
        raise ValueError(f"Expected fastStorage.zip SHA-256 {contract['archive_sha256']}")

    shutil.copytree(CODE / "native", ORIGINAL, dirs_exist_ok=True)
    target_zip = ORIGINAL / "data/raw/fastStorage.zip"
    target_zip.parent.mkdir(parents=True, exist_ok=True)
    if not target_zip.is_file() or file_hash(target_zip) != contract["archive_sha256"]:
        shutil.copyfile(raw, target_zip)

    env = dict(
        os.environ,
        PYTHONPATH=str(ORIGINAL / "src"),
        MSADTI_OUTPUT=str(ORIGINAL / "results"),
        PYTHONUTF8="1",
    )
    if not (ORIGINAL / "results/BITBRAINS_INPUT_LOCK.json").exists():
        subprocess.run(
            [sys.executable, "-B", "-m", "msa_dti_native.prepare"],
            cwd=ORIGINAL,
            env=env,
            check=True,
        )

    for stage in ("A", "B", "C", "D", "E"):
        dest = RUN / "cohorts" / stage
        shutil.copytree(CODE / "protocol" / stage, dest, dirs_exist_ok=True)
        materialize(stage, raw, dest)

    _write_json(
        RUN / "prepared.json",
        {"archive_sha256": file_hash(raw), "groups": {s: len(cases(s)) for s in "ABCDE"}},
    )
    print(f"Prepared Bitbrains cohorts under {RUN.resolve()}")


def run_native_source(stage: str) -> None:
    if not (RUN / "prepared.json").is_file():
        raise RuntimeError("Run `python code/train.py prepare --bitbrains PATH` first.")
    sys.path.insert(0, str(ORIGINAL / "src"))
    os.environ["MSADTI_OUTPUT"] = str(ORIGINAL / "results")
    from msa_dti_native import source_training as st  # noqa: E402

    if stage == "c1":
        for h in (1, 4):
            st.train_c1({"H": h, "id": f"c1_H{h}"})
    elif stage == "medet":
        for h in (1, 4):
            st.medet({"H": h, "id": f"medet_H{h}"})
    elif stage == "hmeta":
        st.hmeta({"id": "hmeta"})
    elif stage == "teacher":
        st.teacher({"id": "teacher"})
    else:
        raise ValueError(f"Unsupported stage: {stage}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=["prepare", "c1", "medet", "hmeta", "teacher", "all"],
        help="prepare: build verified Bitbrains cohorts; c1/medet/hmeta/teacher/all: source training.",
    )
    parser.add_argument("--bitbrains", type=Path, help="Path to GWA-T-12 fastStorage.zip")
    args = parser.parse_args()

    if args.stage in ("prepare", "all"):
        if args.bitbrains is None:
            parser.error("--bitbrains PATH_TO_FASTSTORAGE_ZIP is required for prepare/all")
        prepare(args.bitbrains.resolve())
        if args.stage == "prepare":
            return 0
    if args.stage == "all":
        for sub in ("c1", "teacher", "medet", "hmeta"):
            run_native_source(sub)
        return 0
    run_native_source(args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
