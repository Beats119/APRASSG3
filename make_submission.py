"""Build the Google Form ZIP: exactly one notebook and one PDF report.

Usage (from the repository root):
    python make_submission.py

Output: submission/2611AI26_2611AI44_2611AI59.zip, containing only
  Assignment3_Neural_Networks.ipynb
  Assignment3_Report.pdf
at the top level of the ZIP (no folders inside it).
"""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "submission" / "2611AI26_2611AI44_2611AI59.zip"
FILES = [ROOT / "notebook" / "Assignment3_Neural_Networks.ipynb",
         ROOT / "report" / "Assignment3_Report.pdf"]

if __name__ == "__main__":
    for path in FILES:
        if not path.exists():
            raise FileNotFoundError(f"Missing submission file: {path}")
    OUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in FILES:
            zf.write(path, arcname=path.name)
    with zipfile.ZipFile(OUT) as zf:
        names = zf.namelist()
    assert names == ["Assignment3_Neural_Networks.ipynb",
                     "Assignment3_Report.pdf"], names
    print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes) containing:")
    for name in names:
        print(f"  {name}")
