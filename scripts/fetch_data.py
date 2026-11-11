"""Download the public cohort used for development.

The Metastatic Prostate Cancer Project (provisional, June 2021), as published
in the cBioPortal datahub. Licence: Open Database License (ODbL) 1.0. The files
are not committed to this repository; this script fetches them.

    python scripts/fetch_data.py
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

STUDY = "mpcproject_broad_2021"
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / STUDY
# Large files are stored with Git LFS and served from a different host.
LFS = f"https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/{STUDY}/"
PLAIN = f"https://raw.githubusercontent.com/cBioPortal/datahub/master/public/{STUDY}/"
FILES = {
    "LICENSE": PLAIN,
    "meta_study.txt": PLAIN,
    "data_clinical_patient.txt": LFS,
    "data_clinical_sample.txt": LFS,
    "data_mutations.txt": LFS,
    "data_timeline_treatment.txt": LFS,
    "data_timeline_psa.txt": LFS,
    "data_timeline_sample.txt": LFS,
    "data_timeline_mets.txt": LFS,
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, base in FILES.items():
        target = OUT / name
        if target.exists() and target.stat().st_size > 0:
            print(f"have  {name}")
            continue
        print(f"fetch {name} ...", end=" ", flush=True)
        with urllib.request.urlopen(base + name, timeout=120) as response:
            content = response.read()
        if not content or content.startswith(b"version https://git-lfs"):
            raise SystemExit(f"\n{name}: got an empty file or a Git LFS pointer from {base}")
        target.write_bytes(content)
        print(f"{len(content):,} bytes, sha256 {hashlib.sha256(content).hexdigest()[:12]}")
    print(f"Done. Study files are in {OUT}")


if __name__ == "__main__":
    main()
